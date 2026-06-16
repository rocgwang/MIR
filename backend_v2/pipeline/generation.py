"""LoRA-fine-tuned MusicGen-Melody generation (v2).

Text-only generation (no melody audio conditioning). A 4-bar loop is
generated and cross-fade tiled to fill the full track length.

Adapter weights are resolved from env vars:
  LORA_ADAPTER_PATH        — local directory (default /tmp/lora_adapters)
  LORA_ADAPTER_FILE_ID     — Google Drive file ID for adapter_model.safetensors
"""

import json
import os

import librosa
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

MODEL_NAME = "facebook/musicgen-small"
LOOP_BARS = 4
OUTPUT_SR = 44100

_ADAPTER_FILE_ID = "1SVLKKA5PAoKFHzmMYKMTmwPbzKqbzhQQ"

_ADAPTER_CONFIG = {
    "alora_invocation_tokens": None,
    "alpha_pattern": {},
    "arrow_config": None,
    "auto_mapping": {
        "base_model_class": "MusicgenMelodyForConditionalGeneration",
        "parent_library": "transformers.models.musicgen_melody.modeling_musicgen_melody",
    },
    "base_model_name_or_path": "facebook/musicgen-small",
    "bias": "none",
    "corda_config": None,
    "ensure_weight_tying": False,
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 16,
    "lora_bias": False,
    "lora_dropout": 0.05,
    "lora_ga_config": None,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "peft_version": "0.19.1",
    "qalora_group_size": 16,
    "r": 8,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ["v_proj", "q_proj"],
    "target_parameters": None,
    "task_type": None,
    "trainable_token_indices": None,
    "use_bdlora": None,
    "use_dora": False,
    "use_qalora": False,
    "use_rslora": False,
}


class MusicGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        adapter_path = self._resolve_adapter()
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        base = MusicgenMelodyForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        self.model = PeftModel.from_pretrained(base, adapter_path).eval()

    def _resolve_adapter(self) -> str:
        path = os.getenv("LORA_ADAPTER_PATH", "/tmp/lora_adapters")
        safetensors = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(safetensors):
            import gdown
            os.makedirs(path, exist_ok=True)
            file_id = os.getenv("LORA_ADAPTER_FILE_ID", _ADAPTER_FILE_ID)
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output=safetensors,
                quiet=False,
            )
            with open(os.path.join(path, "adapter_config.json"), "w") as f:
                json.dump(_ADAPTER_CONFIG, f, indent=2)
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
