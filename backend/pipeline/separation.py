"""Vocal isolation via Demucs.

Ported from the AudioPreprocessor.run_demucs step in the preprocessing
notebook: a two-stem (vocals / accompaniment) htdemucs split is enough — the
vocal stem is used both as the MusicGen melody condition and as the vocal
track in the final hybrid mix.
"""

import subprocess
from pathlib import Path


def separate_vocals(input_path: Path, out_dir: Path) -> Path:
    """Run Demucs and return the path to the isolated vocal stem."""
    subprocess.run(
        [
            "demucs",
            "-n", "htdemucs",
            "--two-stems=vocals",
            "--out", str(out_dir),
            str(input_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    vocal_path = out_dir / "htdemucs" / input_path.stem / "vocals.wav"
    if not vocal_path.exists():
        raise RuntimeError(f"Demucs did not produce a vocal stem for {input_path}")
    return vocal_path
