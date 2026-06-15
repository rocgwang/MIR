"""Audio feature extraction used to build the MusicGen prompt.

Ported from chant2tech's AudioFeatureExtractor (BPM + key only — those are
the two values referenced by the MusicGen prompt templates).
"""

import numpy as np
import librosa

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def extract_features(y: np.ndarray, sr: int) -> dict:
    return {
        "bpm": _extract_bpm(y, sr),
        "key": _extract_key(y, sr),
    }


def _extract_bpm(y: np.ndarray, sr: int) -> int:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo[0]) if isinstance(tempo, np.ndarray) else int(tempo)
        return bpm if bpm > 0 else 120
    except Exception:
        return 120


def _extract_key(y: np.ndarray, sr: int) -> str:
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096)
        chroma_mean = np.mean(chroma, axis=1)

        key_idx = int(np.argmax(chroma_mean))
        root_note = NOTE_NAMES[key_idx]

        minor_idx = (key_idx + 3) % 12
        major_idx = (key_idx + 4) % 12

        if chroma_mean[minor_idx] > chroma_mean[major_idx]:
            return f"{root_note} Minor"
        return f"{root_note} Major"
    except Exception:
        return "A Minor"
