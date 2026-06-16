"""Sidechain mix: drums + bass + MusicGen instrumental + vocal chops.

A kick-triggered sidechain envelope ducks the bass, instrumental, and chop
tracks on each beat. The sidechain ramp includes a short attack to prevent
clicks at beat boundaries. Final mix goes through a soft tanh limiter and
peak normalization.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def create_mix(
    drums: np.ndarray,
    techno_instr: np.ndarray,
    chop_track: np.ndarray,
    bass: np.ndarray,
    total_len: int,
    sec_per_beat: float,
    n_bars: int,
    output_path: Path,
) -> Path:
    sc = _sidechain_env(total_len, sec_per_beat, n_bars)

    def _norm(x: np.ndarray, peak: float) -> np.ndarray:
        return x / (np.max(np.abs(x)) + 1e-9) * peak

    drm = _norm(_fit(drums,        total_len), 0.95)
    bas = _norm(_fit(bass,         total_len), 0.90)
    ins = _norm(_fit(techno_instr, total_len), 0.90)
    chp = _norm(_fit(chop_track,   total_len), 0.90)

    mix = 0.95 * drm + 0.20 * (bas * sc) + 0.75 * (ins * sc) + 0.80 * (chp * sc)
    mix = _norm(mix, 0.9)
    mix = np.tanh(mix * 1.1)
    mix = _norm(mix, 0.97)

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
    depth: float = 0.6,
    recover: float = 5.0,
    atk_ms: float = 3.0,
) -> np.ndarray:
    env = np.ones(total_len)
    L = int(sec_per_beat * SR)
    a = int(SR * atk_ms / 1000)
    dip = (1 - depth) + depth * (1 - np.exp(-np.linspace(0, recover, L)))
    for beat in range(n_bars * 4):
        pos = int(beat * sec_per_beat * SR)
        end = min(pos + L, total_len)
        seg = dip[: end - pos].copy()
        if a > 1 and pos > 0:
            r = min(a, len(seg))
            seg[:r] = np.linspace(env[pos - 1], seg[0], r)
        env[pos:end] = seg
    return env
