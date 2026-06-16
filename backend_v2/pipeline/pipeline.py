"""End-to-end Sacred -> Techno v2 conversion pipeline.

1. Crop input to 3 min and run Demucs two-stem split.
2. Extract key from cropped audio for concept-specific prompt.
3. Generate techno instrumental via LoRA MusicGen-Medium (text-only, 4-bar
   loop tiled to full length).
4. Synthesize a harmony-tracking bass line from the instrumental's chroma.
5. Build vocal chop track from onset-sliced vocal fragments.
6. Synthesize a four-on-the-floor kick + hihat drum pattern.
7. Sidechain-mix drums + bass + instrumental + chops into the final WAV.
"""

import random
from pathlib import Path

import librosa
import numpy as np
import torch

from .bass import build_bass_track
from .drums import build_drum_track
from .features import extract_features
from .generation import MusicGenerator
from .mixing import SR, create_mix
from .prompts import CONCEPT_PROMPTS, build_prompt
from .separation import separate
from .vocal_chop import build_chop_track

N_BARS = 8


def convert_to_techno(
    input_path: Path,
    work_dir: Path,
    concept_id: str,
    generator: MusicGenerator,
) -> Path:
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    vocals_path, no_vocals_path = separate(input_path, work_dir / "demucs")

    y, sr = librosa.load(str(work_dir / "demucs" / "input_cropped.wav"), sr=SR, mono=True)
    features = extract_features(y, sr)

    prompt = build_prompt(concept_id, features["key"])
    target_bpm = CONCEPT_PROMPTS[concept_id].target_bpm
    sec_per_beat = 60.0 / target_bpm
    total_len = int(N_BARS * 4 * sec_per_beat * SR)

    techno_instr = generator.generate(prompt, target_bpm, total_len)
    bass = build_bass_track(techno_instr, total_len, sec_per_beat, N_BARS)
    chop_track = build_chop_track(vocals_path, total_len, sec_per_beat, N_BARS)
    drums = build_drum_track(total_len, sec_per_beat, N_BARS)

    output_path = work_dir / f"sacred-techno-v2-{concept_id}.wav"
    return create_mix(
        drums=drums,
        techno_instr=techno_instr,
        chop_track=chop_track,
        bass=bass,
        total_len=total_len,
        sec_per_beat=sec_per_beat,
        n_bars=N_BARS,
        output_path=output_path,
    )
