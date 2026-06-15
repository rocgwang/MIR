"""End-to-end Sacred -> Techno conversion pipeline.

1. Extract BPM/key from the uploaded clip.
2. Isolate the vocal stem with Demucs.
3. Build a concept-specific MusicGen prompt from the extracted key + the
   concept's target BPM.
4. Generate a techno instrumental conditioned on the vocal melody.
5. Mix the original vocal (time-stretched to the target BPM) with the
   generated beat.
"""

from pathlib import Path

import librosa
import torchaudio

from .features import extract_features
from .generation import MusicGenerator
from .mixing import create_hybrid_mix
from .prompts import CONCEPT_PROMPTS, build_prompt
from .separation import separate_vocals

ANALYSIS_SR = 32000


def convert_to_techno(
    input_path: Path, work_dir: Path, concept_id: str, generator: MusicGenerator
) -> Path:
    y, sr = librosa.load(str(input_path), sr=ANALYSIS_SR, mono=True)
    features = extract_features(y, sr)

    vocal_path = separate_vocals(input_path, work_dir / "demucs")

    prompt = build_prompt(concept_id, features["key"])
    target_bpm = CONCEPT_PROMPTS[concept_id].target_bpm

    melody_waveform, melody_sr = torchaudio.load(str(vocal_path))
    beat, beat_sr = generator.generate(melody_waveform, melody_sr, prompt)

    output_path = work_dir / f"techno-{concept_id}.wav"
    create_hybrid_mix(
        vocal_path=vocal_path,
        beat=beat,
        beat_sr=beat_sr,
        output_path=output_path,
        target_bpm=target_bpm,
        original_bpm=features["bpm"],
    )
    return output_path
