"""Hybrid mix: original chant vocal (time-stretched) over the generated beat.

Ported from chant2tech's create_hybrid_mix.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def create_hybrid_mix(
    vocal_path: Path,
    beat: np.ndarray,
    beat_sr: int,
    output_path: Path,
    target_bpm: int,
    original_bpm: int,
    vocal_gain: float = 1.8,
    beat_gain: float = 0.4,
) -> Path:
    vocal_y, sr = librosa.load(str(vocal_path), sr=beat_sr)

    # Time-stretch the vocal from its original tempo to the generated beat's tempo.
    rate = target_bpm / original_bpm
    vocal_stretched = librosa.effects.time_stretch(y=vocal_y, rate=rate)

    n = min(len(vocal_stretched), len(beat))
    mixed = vocal_stretched[:n] * vocal_gain + beat[:n] * beat_gain

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = (mixed / peak) * 0.95

    sf.write(str(output_path), mixed, sr)
    return output_path
