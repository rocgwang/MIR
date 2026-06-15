"""MusicGen-Melody generation.

Ported from chant2tech's melody-conditioned generation step: the isolated
vocal stem is fed in as the melody condition together with a concept-specific
text prompt, so the generated techno follows the original vocal's melodic
contour.
"""

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

MODEL_NAME = "facebook/musicgen-melody"
MELODY_SR = 32000
MELODY_SECONDS = 15


class MusicGenerator:
    """Loads MusicGen-Melody once and reuses it across requests."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).to(device)
        self.model.generation_config.guidance_scale = 3.0

    def generate(
        self, melody_waveform: torch.Tensor, melody_sr: int, prompt: str
    ) -> tuple[np.ndarray, int]:
        waveform = melody_waveform.mean(dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(melody_sr, MELODY_SR)
        waveform = resampler(waveform)
        waveform = waveform[:, : MELODY_SR * MELODY_SECONDS]

        inputs = self.processor(
            audio=waveform.numpy(),
            sampling_rate=MELODY_SR,
            text=[prompt],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if self.model.dtype == torch.float16:
            inputs = {
                k: v.to(torch.float16) if torch.is_floating_point(v) else v
                for k, v in inputs.items()
            }

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=800,
                guidance_scale=2.5,
                do_sample=True,
                temperature=0.9,
                top_k=150,
                top_p=0.9,
            )

        output = audio_values[0, 0].cpu().float().numpy()
        return output, self.model.config.audio_encoder.sampling_rate
