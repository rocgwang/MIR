"""LoRA-fine-tuned MusicGen-Melody generation (v2).

Loads LoRA adapter weights on top of facebook/musicgen-melody. The melody
condition is the no_vocals (instrumental) stem so MusicGen tracks the
original harmonic/chroma content rather than the vocal melody.

Adapter location is resolved from env vars:
  LORA_ADAPTER_PATH       — local directory (default /tmp/lora_adapters)
  LORA_ADAPTER_GDRIVE_URL — Google Drive folder URL, downloaded on first run
"""

import os
from pathlib import Path

import librosa
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

MODEL_NAME = "facebook/musicgen-melody"
MELODY_SR = 32000
MELODY_SECONDS = 15
N_BARS = 8
OUTPUT_SR = 44100

_GDRIVE_URL = "https://drive.google.com/drive/folders/1cxKcfxQB7GTl4IEX71e09wVE0oTXPwCU?usp=sharing"


class MusicGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        adapter_path = self._resolve_adapter()
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        base = MusicgenMelodyForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        self.model = PeftModel.from_pretrained(base, adapter_path).eval()

    def _resolve_adapter(self) -> str:
        path = os.getenv("LORA_ADAPTER_PATH", "/tmp/lora_adapters")
        if not os.path.exists(path):
            import gdown
            url = os.getenv("LORA_ADAPTER_GDRIVE_URL", _GDRIVE_URL)
            gdown.download_folder(url, output=path, quiet=False)
        return path

    def generate(
        self, no_vocals_path: Path, prompt: str, target_bpm: int
    ) -> np.ndarray:
        sec_per_beat = 60.0 / target_bpm
        total_sec = N_BARS * 4 * sec_per_beat
        max_new_tokens = int(min(total_sec, 30) * 50)

        melody, _ = librosa.load(
            str(no_vocals_path), sr=MELODY_SR, mono=True, duration=MELODY_SECONDS
        )

        inputs = self.processor(
            audio=melody,
            sampling_rate=MELODY_SR,
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=4.0,
                temperature=0.8,
                top_k=200,
                max_new_tokens=max_new_tokens,
            )

        out_sr = self.model.config.audio_encoder.sampling_rate
        raw = audio_values[0, 0].cpu().float().numpy()
        return librosa.resample(raw, orig_sr=out_sr, target_sr=OUTPUT_SR)
