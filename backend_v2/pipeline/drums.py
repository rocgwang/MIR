"""Four-on-the-floor drum synthesis (no samples needed).

Kick: pitch-dropping sine wave.
Hihat: decaying white noise on the offbeats (8th notes).
"""

import numpy as np

SR = 44100


def build_drum_track(total_len: int, sec_per_beat: float, n_bars: int = 8) -> np.ndarray:
    kick = _synth_kick()
    hat = _synth_hat()
    drums = np.zeros(total_len)

    for beat in range(n_bars * 4):
        pos = int(beat * sec_per_beat * SR)
        end = min(pos + len(kick), total_len)
        if pos < total_len:
            drums[pos:end] += kick[: end - pos]

    for eighth in range(n_bars * 8):
        if eighth % 2 == 1:
            pos = int(eighth * (sec_per_beat / 2) * SR)
            end = min(pos + len(hat), total_len)
            if pos < total_len:
                drums[pos:end] += hat[: end - pos]

    peak = np.max(np.abs(drums)) + 1e-9
    return drums / peak


def _synth_kick(dur: float = 0.35, f0: float = 130.0, f1: float = 45.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    f = f1 + (f0 - f1) * np.exp(-t * 35)
    phase = 2 * np.pi * np.cumsum(f) / SR
    k = np.sin(phase) * np.exp(-t * 7)
    fo = int(SR * 0.005)
    k[-fo:] *= np.linspace(1, 0, fo)
    return k


def _synth_hat(dur: float = 0.05) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    h = np.random.randn(len(t)) * np.exp(-t * 90) * 0.4
    fo = int(SR * 0.003)
    h[-fo:] *= np.linspace(1, 0, fo)
    return h
