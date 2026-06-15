# Sufi → Techno conversion backend

FastAPI service that implements the actual conversion pipeline (Demucs vocal
separation -> feature extraction -> MusicGen-Melody generation -> hybrid mix),
adapted from the `dl_for_music_오디오_전처리.py` and `chant2tech` notebooks.

This needs PyTorch + a CUDA GPU, so it runs as a separate process from the
Next.js app — `app/api/convert/route.ts` proxies requests to it.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

`demucs` also requires `ffmpeg` to be available on `PATH`.

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The first request is slow because `facebook/musicgen-melody` (and the Demucs
model weights) are downloaded and loaded into GPU memory on startup.

### Running on Colab

If you're running this on a Colab GPU notebook instead of a local machine,
start uvicorn and expose port 8000 with a tunnel (e.g. `ngrok http 8000`),
then point the frontend at the resulting `https://*.ngrok.io` URL.

## Wiring up the frontend

In the `mir-web` root, set:

```bash
# .env.local
ML_BACKEND_URL=http://localhost:8000
```

`app/api/convert/route.ts` will then forward uploads to
`${ML_BACKEND_URL}/convert` and stream the generated `audio/wav` back to the
browser.

## Pipeline overview (`pipeline/`)

- `features.py` — extracts BPM and key (Chroma-based) from the uploaded clip.
- `separation.py` — Demucs two-stem split, returns the isolated vocal/chant stem.
- `prompts.py` — maps each concept id (`powerful`, `groove`, `hypnotic`, `dark`,
  `melodic`, `minimal` — must match `lib/concepts.ts`) to a MusicGen prompt
  template and target BPM.
- `generation.py` — loads MusicGen-Melody once at startup and generates a
  techno instrumental conditioned on the vocal melody + concept prompt.
- `mixing.py` — time-stretches the original vocal to the target BPM and mixes
  it with the generated instrumental.
- `pipeline.py` — orchestrates the steps above into `convert_to_techno()`.
