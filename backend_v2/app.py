"""FastAPI service — Sacred -> Techno v2 pipeline.

Same /convert endpoint contract as backend_v1 (audio upload + concept form
field, returns audio/wav), so the frontend works with either backend.

Run with (GPU required):
    uvicorn app:app --host 0.0.0.0 --port 8000

LoRA adapter weights are resolved from env vars at startup:
  LORA_ADAPTER_PATH       local directory (default /tmp/lora_adapters)
  LORA_ADAPTER_GDRIVE_URL Google Drive folder URL (auto-downloaded if path missing)
"""

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pipeline.generation import MusicGenerator
from pipeline.pipeline import convert_to_techno
from pipeline.prompts import CONCEPT_IDS

ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["generator"] = MusicGenerator()
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/convert")
async def convert(audio: UploadFile = File(...), concept: str = Form(...)):
    if concept not in CONCEPT_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown concept: {concept}")

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        input_path = work_dir / (audio.filename or "input.wav")

        with input_path.open("wb") as f:
            shutil.copyfileobj(audio.file, f)

        try:
            output_path = convert_to_techno(
                input_path=input_path,
                work_dir=work_dir,
                concept_id=concept,
                generator=ml_models["generator"],
            )
            output_bytes = output_path.read_bytes()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=output_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="sacred-techno-v2-{concept}.wav"'},
    )
