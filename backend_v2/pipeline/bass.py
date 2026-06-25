"""Bass line synthesizer.

Tracks chord roots from the original input audio's chroma, then synthesizes
a 16th-note square-saw bass loop that gallops around the vocal-chop
off-beats, bar by bar.
"""

import librosa
import numpy as np

SR = 44100
BASS_PATTERN = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]


def build_bass_track(
    input_audio: np.ndarray,
    total_len: int,
    sec_per_beat: float,
    n_bars: int = 8,
    loop_bars: int = 4,
) -> np.ndarray:
    loop_len = int(loop_bars * 4 * sec_per_beat * SR)
    analysis_signal = input_audio[:loop_len] if len(input_audio) >= loop_len else input_audio
    roots = _bar_roots(analysis_signal, SR, loop_bars)
    sixteenth = sec_per_beat / 4
    bass_loop = np.zeros(loop_len)

    for bar in range(loop_bars):
        bar_start = int(bar * 4 * sec_per_beat * SR)
        root = roots[bar]
        if bar == 1 and roots[1] == roots[0]:
            root *= 1.25
        if bar == 2 and roots[2] == roots[0]:
            root *= 1.5
        if bar == 3 and roots[3] == roots[0]:
            root *= 1.12

        for step in range(16):
            if not BASS_PATTERN[step]:
                continue
            pos = bar_start + int(step * sixteenth * SR)
            note = _synth_bass_note(root, sixteenth * 1.25)
            end = min(pos + len(note), loop_len)
            if pos < loop_len:
                bass_loop[pos:end] += note[: end - pos]

    bass_loop = bass_loop / (np.max(np.abs(bass_loop)) + 1e-9) * 0.85
    return _tile_xfade(bass_loop, total_len, int(0.015 * SR))


def _bar_roots(sig: np.ndarray, sr: int, n_bars: int) -> list:
    ch = librosa.feature.chroma_stft(y=sig, sr=sr)
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
    square = np.sign(np.sin(2 * np.pi * freq * t))
    saw = 2 * (t * freq - np.floor(0.5 + t * freq))
    tone = 0.7 * square + 0.3 * saw

    atk_len = max(1, int(SR * 0.002))
    env = np.exp(-t * 14)
    env[:atk_len] *= np.linspace(0, 1, atk_len)
    return tone * env * 0.5


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
