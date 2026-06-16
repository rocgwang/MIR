"""Vocal/instrumental isolation via Demucs.

Crops the input to crop_seconds first (memory + speed), then runs a
two-stem htdemucs split. Returns (vocals_path, no_vocals_path) — both
stems are needed: vocals for the chop track, no_vocals for MusicGen melody
conditioning.
"""

import subprocess
from pathlib import Path

import librosa
import soundfile as sf


def separate(
    input_path: Path, out_dir: Path, crop_seconds: int = 180
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped_path = _crop(input_path, out_dir, crop_seconds)
    return _demucs(cropped_path, out_dir)


def _crop(input_path: Path, out_dir: Path, crop_seconds: int) -> Path:
    y, sr = librosa.load(str(input_path), sr=44100, mono=True, duration=crop_seconds)
    cropped = out_dir / "input_cropped.wav"
    sf.write(str(cropped), y, sr)
    return cropped


def _demucs(input_path: Path, out_dir: Path) -> tuple[Path, Path]:
    subprocess.run(
        ["demucs", "--two-stems", "vocals", "--out", str(out_dir), str(input_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stem_dir = out_dir / "htdemucs" / input_path.stem
    vocals_path = stem_dir / "vocals.wav"
    no_vocals_path = stem_dir / "no_vocals.wav"
    if not vocals_path.exists() or not no_vocals_path.exists():
        raise RuntimeError(f"Demucs did not produce expected stems in {stem_dir}")
    return vocals_path, no_vocals_path
