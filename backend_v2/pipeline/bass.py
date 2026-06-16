"""Bass line synthesizer.

Tracks chord roots from the techno instrumental's chroma, then synthesizes
an 8th-note sawtooth-sine bass loop that follows the harmony bar by bar.
"""

import librosa
import numpy as np

SR = 44100


def build_bass_track(
    techno_instr: np.ndarray,
    total_len: int,
    sec_per_beat: float,
    n_bars: int = 8,
    loop_bars: int = 4,
) -> np.ndarray:
    loop_len = int(loop_bars * 4 * sec_per_beat * SR)
    roots = _bar_roots(techno_instr, SR, loop_bars)
    eighth = sec_per_beat / 2
    bass_loop = np.zeros(loop_len)
    for bar in range(loop_bars):
        for k in range(8):
            pos = int((bar * 4 * sec_per_beat + k * eighth) * SR)
            note = _synth_bass_note(roots[bar], eighth)
            end = min(pos + len(note), loop_len)
            if pos < loop_len:
                bass_loop[pos:end] += note[: end - pos]
    bass_loop = bass_loop / (np.max(np.abs(bass_loop)) + 1e-9) * 0.9
    return _tile_xfade(bass_loop, total_len, int(0.012 * SR))


def _bar_roots(sig: np.ndarray, sr: int, n_bars: int) -> list:
    ch = librosa.feature.chroma_cqt(y=sig, sr=sr)
    T = ch.shape[1]
    roots = []
    for b in range(n_bars):
        s = int(b * T / n_bars)
        e = max(int((b + 1) * T / n_bars), s + 1)
        pc = int(np.argmax(ch[:, s:e].mean(axis=1)))
        roots.append(librosa.midi_to_hz(36 + pc))
    return roots


def _synth_bass_note(freq: float, dur: float) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    saw = 2 * (t * freq - np.floor(0.5 + t * freq))
    tone = 0.7 * saw + 0.3 * np.sin(2 * np.pi * freq * t)
    return tone * np.exp(-t * 5) * 0.7


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
