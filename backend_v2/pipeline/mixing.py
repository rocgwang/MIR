"""Sidechain mix: drums + MusicGen instrumental + vocal chops.

A kick-triggered sidechain envelope ducks the instrumental and chop tracks
on each beat, giving the characteristic techno "pumping" feel. The final
mix goes through a soft tanh limiter and peak normalization.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def create_mix(
    drums: np.ndarray,
    techno_instr: np.ndarray,
    chop_track: np.ndarray,
    total_len: int,
    sec_per_beat: float,
    n_bars: int,
    output_path: Path,
) -> Path:
    sc = _sidechain_env(total_len, sec_per_beat, n_bars)

    mix = (
        0.9 * _fit(drums, total_len)
        + 0.7 * (_fit(techno_instr, total_len) * sc)
        + 0.6 * (_fit(chop_track, total_len) * sc)
    )
    mix = np.tanh(mix * 1.2)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.97

    sf.write(str(output_path), mix, SR)
    return output_path


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def _sidechain_env(
    total_len: int,
    sec_per_beat: float,
    n_bars: int,
    depth: float = 0.65,
    recover: float = 5.0,
) -> np.ndarray:
    env = np.ones(total_len)
    L = int(sec_per_beat * SR)
    dip = (1 - depth) + depth * (1 - np.exp(-np.linspace(0, recover, L)))
    for beat in range(n_bars * 4):
        pos = int(beat * sec_per_beat * SR)
        end = min(pos + L, total_len)
        env[pos:end] = dip[: end - pos]
    return env
