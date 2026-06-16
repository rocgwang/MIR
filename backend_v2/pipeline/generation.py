"""LoRA-fine-tuned MusicGen-Medium generation (v2).

Text-only generation (no melody conditioning). A 4-bar loop is generated
and cross-fade tiled to fill the full track length.

Adapter location is resolved from env vars:
  LORA_ADAPTER_PATH       — local directory (default /tmp/lora_adapters)
  LORA_ADAPTER_GDRIVE_URL — Google Drive folder URL, downloaded on first run
"""

import os

import librosa
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

MODEL_NAME = "facebook/musicgen-medium"
LOOP_BARS = 4
OUTPUT_SR = 44100

_GDRIVE_URL = "https://drive.google.com/drive/folders/1cxKcfxQB7GTl4IEX71e09wVE0oTXPwCU?usp=sharing"


class MusicGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        adapter_path = self._resolve_adapter()
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        base = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        self.model = PeftModel.from_pretrained(base, adapter_path).eval()

    def _resolve_adapter(self) -> str:
        path = os.getenv("LORA_ADAPTER_PATH", "/tmp/lora_adapters")
        if not os.path.exists(path):
            import gdown
            url = os.getenv("LORA_ADAPTER_GDRIVE_URL", _GDRIVE_URL)
            gdown.download_folder(url, output=path, quiet=False)
        return path

    def generate(self, prompt: str, target_bpm: int, total_len: int) -> np.ndarray:
        sec_per_beat = 60.0 / target_bpm
        loop_sec = LOOP_BARS * 4 * sec_per_beat
        loop_len = int(loop_sec * OUTPUT_SR)
        max_new_tokens = int(min(loop_sec, 30) * 50)

        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                temperature=0.9,
                top_k=200,
                max_new_tokens=max_new_tokens,
            )

        out_sr = self.model.config.audio_encoder.sampling_rate
        raw = audio_values[0, 0].cpu().float().numpy()
        loop = librosa.resample(raw, orig_sr=out_sr, target_sr=OUTPUT_SR)
        loop = loop[:loop_len]
        loop = loop / (np.max(np.abs(loop)) + 1e-9) * 0.95
        return _tile_xfade(loop, total_len, int(0.012 * OUTPUT_SR))


def _tile_xfade(loop: np.ndarray, total: int, xf: int) -> np.ndarray:
    out = np.zeros(total)
    L = len(loop)
    if L == 0:
        return out
    xf = min(xf, L // 2)
    pos = 0
    while pos < total:
        end = min(pos + L, total)
        chunk = loop[: end - pos].copy()
        if pos > 0 and xf > 0:
            n = min(xf, len(chunk))
            f = np.linspace(0, 1, n)
            out[pos : pos + n] = out[pos : pos + n] * (1 - f) + chunk[:n] * f
            out[pos + n : end] += chunk[n:]
        else:
            out[pos:end] += chunk
        pos += L - xf
    return out
