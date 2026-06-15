# Sufi → Techno conversion backend

FastAPI service that implements the actual conversion pipeline (Demucs vocal
separation -> feature extraction -> MusicGen-Melody generation -> hybrid mix),
adapted from the `dl_for_music_오디오_전처리.py` and `chant2tech` notebooks.

This needs PyTorch + a CUDA GPU, so it runs as a separate process from the
Next.js app. The browser uploads audio **directly** to this server (CORS is
enabled in `app.py`) rather than through a Next.js API route — that avoids
Vercel's ~4.5MB serverless function request body limit, which audio files
exceed easily.

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
NEXT_PUBLIC_ML_BACKEND_URL=http://localhost:8000
```

The page uploads audio directly to `${NEXT_PUBLIC_ML_BACKEND_URL}/convert`
and plays back the returned `audio/wav`.

> `NEXT_PUBLIC_*` env vars are inlined into the JS bundle at build time. On
> Vercel, set this in the project's environment variables and redeploy
> whenever the backend URL changes (e.g. a new ngrok tunnel) — a free ngrok
> URL changes on every restart, so for anything beyond quick testing use a
> stable hostname (paid ngrok domain, Cloudflare Tunnel, or a hosted GPU box).

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
