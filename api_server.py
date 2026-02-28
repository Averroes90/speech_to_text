"""
FastAPI wrapper for the transcription/translation pipeline.
Bearer token auth. Mirrors backend/main.py but adds auth + production hardening.
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_BEARER_TOKEN:
    raise RuntimeError("API_BEARER_TOKEN not set in .env")

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials


# ---------------------------------------------------------------------------
# Pipeline imports (your existing code)
# ---------------------------------------------------------------------------
import app as pipeline
from utils.utils import translate_srt
from handlers_and_protocols.handlers import get_environmet_handler

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("API server starting up")
    yield
    logger.info("API server shutting down")


app = FastAPI(
    title="Transcribe & Translate API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Public health check — no auth required."""
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    video: UploadFile = File(...),
    source_language: str = Form("it"),
    target_language: str = Form("en"),
    services: str = Form('["openai"]'),
    server_region: str = Form("us-central1"),
    translate: bool = Form(True),
    _token: str = Depends(verify_token),
):
    """
    Upload an audio/video file for transcription (and optional translation).

    Parameters:
    - video: Audio or video file to transcribe
    - source_language: Source language code (default: "it")
    - target_language: Target language code (default: "en")
    - services: JSON array of service names, e.g. '["openai"]' or '["openai", "google"]'
    - server_region: Google Cloud region for Chirp (default: "us-central1")
    - translate: Whether to translate (default: true). Auto-disabled if source == target.
    """
    service_list = json.loads(services)

    suffix = Path(video.filename).suffix if video.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        needs_translation = translate and (source_language != target_language)

        if needs_translation:
            srt_responses = pipeline.multi_transcribe(
                file_path=tmp_path,
                service_names=service_list,
                source_language=source_language,
                target_language=target_language,
                audio_output_extension=".flac",
                server_region=server_region,
            )
        else:
            srt_responses = pipeline.multi_transcribe_only(
                file_path=tmp_path,
                service_names=service_list,
                source_language=source_language,
                audio_output_extension=".flac",
                server_region=server_region,
            )

        return {
            "success": True,
            "results": srt_responses,
            "filename": video.filename,
            "services_used": service_list,
            "translated": needs_translation,
        }

    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/translate")
def translate_text_endpoint(
    text: str = Form(...),
    source_language: str = Form("it"),
    target_language: str = Form("en"),
    service: str = Form("google"),
    _token: str = Depends(verify_token),
):
    """
    Translate SRT text or plain text.

    Parameters:
    - text: The text (or SRT content) to translate
    - source_language: Source language code
    - target_language: Target language code
    - service: Translation service to use (default: "google")
    """
    try:
        env_handler = get_environmet_handler(service=service)
        env_handler.load_environment()

        translated = translate_srt(
            srt_content=text,
            service=service,
            source_language=source_language,
            target_language=target_language,
            env_loaded=True,
        )
        return {
            "success": True,
            "translated": translated,
            "source_language": source_language,
            "target_language": target_language,
        }
    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run directly for local dev
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
