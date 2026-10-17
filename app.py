import os
import re
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

# First real editing feature: detect long quiet sections and remove them.
# These are conservative defaults; they do NOT detect repeated words or bad takes.
SILENCE_NOISE_DB = "-35dB"
MIN_SILENCE_SECONDS = 0.65
KEEP_MARGIN_SECONDS = 0.12
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def run_command(command, timeout=600):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed on the server yet.",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Processing timed out. Try a shorter video.",
        )

    if result.returncode != 0:
        # Avoid returning potentially huge FFmpeg logs to the browser.
        detail = (result.stderr or "Unknown FFmpeg error")[-1800:]
        raise HTTPException(status_code=500, detail=f"FFmpeg failed: {detail}")
    return result


def get_duration(path: Path) -> float:
    result = run_command(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=60,
    )
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        raise HTTPException(status_code=500, detail="Could not read video duration.")


def detect_silences(path: Path):
    # FFmpeg's silencedetect reports intervals in its stderr log.
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-i", str(path),
                "-vn", "-af",
                f"silencedetect=noise={SILENCE_NOISE_DB}:d={MIN_SILENCE_SECONDS}",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed on the server yet.",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Silence analysis timed out. Try a shorter video.",
        )

    # FFmpeg normally returns 0 for this analysis. Parse stderr even if it does not.
    log = result.stderr or ""
    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log)]
    duration = get_duration(path)

    silences = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else duration
        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        if end - start >= MIN_SILENCE_SECONDS:
            silences.append((start, end))
    return duration, silences


def build_keep_segments(duration, silences):
    # Keep a small margin around detected silence boundaries to avoid clipping speech.
    keep = []
    cursor = 0.0

    for silence_start, silence_end in silences:
        cut_start = max(cursor, silence_start - KEEP_MARGIN_SECONDS)
        cut_end = min(duration, silence_end + KEEP_MARGIN_SECONDS)

        if cut_start > cursor:
            keep.append((cursor, cut_start))
        cursor = max(cursor, cut_end)

    if cursor < duration:
        keep.append((cursor, duration))

    # Avoid generating a malformed filter for an empty/near-empty result.
    return [(start, end) for start, end in keep if end - start >= 0.08]


@app.get("/")
def home():
    return {
        "app": "AI Edivio",
        "status": "running",
        "message": "Video API is ready",
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
            detail="Please upload an MP4, MOV, WEBM or M4V video.",
        )

    job_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{job_id}{ext}"
    size = 0

    try:
        with saved_path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Video limit is 200 MB.",
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
        "message": "Video uploaded successfully.",
    }


@app.post("/process/{job_id}")
def process_video(job_id: str):
    matches = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video not found.")

    source = matches[0]
    output = OUTPUT_DIR / f"{job_id}_processed.mp4"

    duration, silences = detect_silences(source)
    keep_segments = build_keep_segments(duration, silences)

    # If silence detection finds nothing, still create a valid MP4 transcode.
    if not keep_segments:
        keep_segments = [(0.0, duration)]

    if not silences:
        run_command(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-movflags", "+faststart",
                str(output),
            ],
            timeout=600,
        )
        removed_seconds = 0.0
    else:
        # Trim the same time ranges from video and audio, then concatenate them.
        filters = []
        concat_inputs = []
        for index, (start, end) in enumerate(keep_segments):
            filters.append(
                f"[0:v]trim=start={start:.3f}:end={end:.3f},"
                f"setpts=PTS-STARTPTS[v{index}]"
            )
            filters.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
            concat_inputs.append(f"[v{index}][a{index}]")

        filters.append(
            "".join(concat_inputs)
            + f"concat=n={len(keep_segments)}:v=1:a=1[vout][aout]"
        )

        # Some source videos have no audio stream. Handle them without audio filters.
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0", str(source),
            ],
            capture_output=True, text=True, timeout=60,
        )
        has_audio = bool((probe.stdout or "").strip())

        if has_audio:
            run_command(
                [
                    "ffmpeg", "-y", "-i", str(source),
                    "-filter_complex", ";".join(filters),
                    "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-movflags", "+faststart",
                    str(output),
                ],
                timeout=600,
            )
        else:
            video_filters = []
            video_inputs = []
            for index, (start, end) in enumerate(keep_segments):
                video_filters.append(
                    f"[0:v]trim=start={start:.3f}:end={end:.3f},"
                    f"setpts=PTS-STARTPTS[v{index}]"
                )
                video_inputs.append(f"[v{index}]")
            video_filters.append(
                "".join(video_inputs)
                + f"concat=n={len(keep_segments)}:v=1:a=0[vout]"
            )
            run_command(
                [
                    "ffmpeg", "-y", "-i", str(source),
                    "-filter_complex", ";".join(video_filters),
                    "-map", "[vout]",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-movflags", "+faststart",
                    str(output),
                ],
                timeout=600,
            )

        kept_seconds = sum(end - start for start, end in keep_segments)
        removed_seconds = max(0.0, duration - kept_seconds)

    return {
        "status": "processed",
        "job_id": job_id,
        "download_url": f"/download/{job_id}",
        "editing": "silence_cut",
        "original_duration_seconds": round(duration, 2),
        "removed_silence_seconds": round(removed_seconds, 2),
        "silence_intervals_detected": len(silences),
        "message": (
            "Long quiet sections were detected and removed where found. "
            "Repeated words and bad takes are not detected yet."
        ),
    }


@app.get("/download/{job_id}")
def download_video(job_id: str):
    output = OUTPUT_DIR / f"{job_id}_processed.mp4"
    if not output.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed video not found.",
        )

    return FileResponse(
        output,
        media_type="video/mp4",
        filename="ai_edivio_processed.mp4",
    )
