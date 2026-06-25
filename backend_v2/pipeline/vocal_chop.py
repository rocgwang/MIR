"""Vocal chop synthesis.

Detects onsets directly in the (mixed) input audio, slices them into short
chops with a fast-attack / exponential-decay envelope, then places them on
the sparse 16th-note grid positions that leave room for the kick — a
percussive stab effect.
"""

import numpy as np
import librosa

SR = 44100
CHOP_PATTERN = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]


def build_chop_track(
    input_audio: np.ndarray,
    total_len: int,
    sec_per_beat: float,
    n_bars: int = 8,
) -> np.ndarray:
    chops = _extract_chops(input_audio)
    return _place_on_grid(chops, total_len, sec_per_beat, n_bars)


def _extract_chops(v: np.ndarray) -> list[np.ndarray]:
    onsets = librosa.onset.onset_detect(y=v, sr=SR, backtrack=True, units="samples")
    bounds = list(onsets) + [len(v)]
    chops = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        seg = v[s:e]
        if len(seg) < int(0.05 * SR):
            continue
        seg = seg[: int(0.35 * SR)]
        env = np.ones(len(seg))
        a = max(1, int(0.004 * SR))
        env[:a] = np.linspace(0, 1, a)
        env[a:] = np.exp(-np.linspace(0, 6, len(seg) - a))
        seg = seg * env
        peak = np.max(np.abs(seg)) + 1e-9
        chops.append(seg / peak)
    return chops


def _place_on_grid(
    chops: list[np.ndarray],
    total_len: int,
    sec_per_beat: float,
    n_bars: int,
) -> np.ndarray:
    step_len = sec_per_beat / 4
    steps_bar = 16
    track = np.zeros(total_len)
    if not chops:
        return track
    ci = 0
    for bar in range(n_bars):
        for stp in range(steps_bar):
            if not CHOP_PATTERN[stp]:
                continue
            chop = chops[ci % len(chops)]
            ci += 1
            pos = int(((bar * steps_bar) + stp) * step_len * SR)
            end = min(pos + len(chop), total_len)
            if pos < total_len:
                track[pos:end] += chop[: end - pos]
    peak = np.max(np.abs(track)) + 1e-9
    return track / peak
