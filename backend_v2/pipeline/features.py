"""Audio feature extraction (key) for building the MusicGen prompt."""

import numpy as np
import librosa

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def extract_features(y: np.ndarray, sr: int) -> dict:
    return {"key": _extract_key(y, sr)}


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
