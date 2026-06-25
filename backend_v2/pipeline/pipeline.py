"""End-to-end Sacred -> Techno v2 conversion pipeline."""

from pathlib import Path

import librosa

from .bass import build_bass_track
from .drums import build_drum_track
from .generation import MusicGenerator
from .mixing import SR, create_mix
from .prompts import CONCEPT_PROMPTS, build_prompt
from .vocal_chop import build_chop_track

N_BARS = 8
SEED = 42
CROP_SECONDS = 180


def convert_to_techno(
    input_path: Path,
    work_dir: Path,
    concept_id: str,
    generator: MusicGenerator,
) -> Path:
    input_audio, _ = librosa.load(str(input_path), sr=SR, mono=True, duration=CROP_SECONDS)

    prompt = build_prompt(concept_id)
    target_bpm = CONCEPT_PROMPTS[concept_id].target_bpm
    sec_per_beat = 60.0 / target_bpm
    total_len = int(N_BARS * 4 * sec_per_beat * SR)

    techno_instr = generator.generate(prompt, target_bpm, total_len, seed=SEED)
    bass = build_bass_track(input_audio, total_len, sec_per_beat, N_BARS)
    chop_track = build_chop_track(input_audio, total_len, sec_per_beat, N_BARS)
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
