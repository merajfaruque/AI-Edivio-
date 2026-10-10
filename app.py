
import os
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Edivio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path("/tmp/ai_edivio")
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return {
        "app": "AI Edivio",
        "status": "running",
        "message": "Video API is ready"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    allowed = {".mp4", ".mov", ".webm", ".m4v"}

    ext = Path(video.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Please upload an MP4, MOV, WEBM or M4V video."
        )

    job_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{job_id}{ext}"

    size = 0
    limit = 200 * 1024 * 1024

    try:
        with saved_path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail="Video limit is 200 MB."
                    )
                out.write(chunk)
    except Exception:
        saved_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    return {
        "status": "uploaded",
        "job_id": job_id,
        "filename": video.filename,
        "size_bytes": size,
        "message": "Video uploaded successfully."
    }


@app.post("/process/{job_id}")
def process_video(job_id: str):
    matches = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video not found.")

    source = matches[0]
    output = OUTPUT_DIR / f"{job_id}_processed.mp4"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-movflags", "+faststart",
                str(output)
            ],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="Video processing failed. Check server FFmpeg setup."
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed on the server yet."
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Processing timed out. Try a shorter video."
        )

    return {
        "status": "processed",
        "job_id": job_id,
        "download_url": f"/download/{job_id}"
    }


@app.get("/download/{job_id}")
def download_video(job_id: str):
    output = OUTPUT_DIR / f"{job_id}_processed.mp4"
    if not output.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed video not found."
        )

    return FileResponse(
        output,
        media_type="video/mp4",
        filename="ai_edivio_processed.mp4"
    )
