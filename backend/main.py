from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import sys
import os
from pathlib import Path
import tempfile
import json

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import app

app_instance = FastAPI()

# Configure CORS
app_instance.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app_instance.post("/transcribe")
async def transcribe_video(
    video: UploadFile = File(...),
    source_language: str = Form("it"),
    target_language: str = Form("en"),
    services: str = Form('["openai"]'),
    server_region: str = Form("us-central1")
):
    """
    Upload a video file and get SRT transcription/translation from multiple services.

    Parameters:
    - video: Video file to transcribe
    - source_language: Source language code (default: "it")
    - target_language: Target language code (default: "en")
    - services: JSON array of service names - e.g. '["openai", "google"]' (default: '["openai"]')
    - server_region: Google Cloud region for Chirp (default: "us-central1")

    Returns:
    - JSON with SRT content from each service (uses multithreading for concurrent transcription)
    """
    # Parse services list
    service_list = json.loads(services)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as temp_file:
        content = await video.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Call existing multi_transcribe function (already handles multithreading)
        srt_responses = app.multi_transcribe(
            file_path=temp_file_path,
            service_names=service_list,
            source_language=source_language,
            target_language=target_language,
            audio_output_extension=".flac",
            server_region=server_region
        )

        return {
            "success": True,
            "results": srt_responses,
            "filename": video.filename,
            "services_used": service_list
        }

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app_instance.get("/")
async def root():
    return {"message": "Transcription API is running", "available_services": ["openai", "google"]}
