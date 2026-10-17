import os
import re
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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

# Conservative first-pass settings. These improve common cases but cannot
# perfectly remove room echo, every kind of noise, or retakes.
SILENCE_NOISE_DB = "-35dB"
MIN_SILENCE_SECONDS = 0.65
KEEP_MARGIN_SECONDS = 0.12
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# FFmpeg filters: light broadband noise reduction + speech-friendly loudness.
# These are defaults, not a guarantee that every recording will sound better.
AUDIO_FILTERS = (
    "afftdn=nr=10:nf=-50:tn=1,"
    "loudnorm=I=-16:LRA=11:TP=-1.5"
)


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
            detail="FFmpeg/ffprobe is not installed on the server.",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Processing timed out. Try a shorter video.",
        )

    if result.returncode != 0:
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
        duration = float(result.stdout.strip())
    except (TypeError, ValueError):
        raise HTTPException(status_code=500, detail="Could not read video duration.")
    if duration <= 0:
        raise HTTPException(status_code=400, detail="The uploaded video has no usable duration.")
    return duration


def has_audio_stream(path: Path) -> bool:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", str(path),
        ],
        timeout=60,
    )
    return bool((result.stdout or "").strip())


def detect_silences(path: Path):
    """Return (duration, [(silence_start, silence_end), ...])."""
    duration = get_duration(path)
    if not has_audio_stream(path):
        return duration, []

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
            detail="FFmpeg is not installed on the server.",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Silence analysis timed out. Try a shorter video.",
        )

    log = result.stderr or ""
    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log)]

    silences = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else duration
        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        if end - start >= MIN_SILENCE_SECONDS:
            silences.append((start, end))

    # Keep intervals ordered and merge any overlaps caused by log rounding.
    silences.sort(key=lambda item: item[0])
    merged = []
    for start, end in silences:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return duration, merged


def build_keep_segments(duration, silences):
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

    keep = [(start, end) for start, end in keep if end - start >= 0.08]
    # If the entire clip is detected as silence, preserve the original clip
    # rather than accidentally creating an empty/invalid output.
    return keep or [(0.0, duration)]


def make_video_only_filter(keep_segments):
    filters = []
    video_inputs = []
    for index, (start, end) in enumerate(keep_segments):
        filters.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        video_inputs.append(f"[v{index}]")
    filters.append(
        "".join(video_inputs)
        + f"concat=n={len(keep_segments)}:v=1:a=0[vout]"
    )
    return ";".join(filters)


def make_av_filter(keep_segments):
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
        + f"concat=n={len(keep_segments)}:v=1:a=1[vout][acut]"
    )
    # Apply enhancement after joining segments to keep filter behavior consistent.
    filters.append(f"[acut]{AUDIO_FILTERS}[aenh]")
    return ";".join(filters)


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
                    raise HTTPException(status_code=413, detail="Video limit is 200 MB.")
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
    audio_present = has_audio_stream(source)
    keep_segments = build_keep_segments(duration, silences) if silences else [(0.0, duration)]

    # Don't trim silence on a video with no audio; still transcode to a compatible MP4.
    if silences and audio_present:
        filter_complex = make_av_filter(keep_segments)
        run_command(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-filter_complex", filter_complex,
                "-map", "[vout]", "-map", "[aenh]",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(output),
            ],
            timeout=900,
        )
    elif silences and not audio_present:
        run_command(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-filter_complex", make_video_only_filter(keep_segments),
                "-map", "[vout]",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                "-movflags", "+faststart",
                str(output),
            ],
            timeout=900,
        )
    elif audio_present:
        run_command(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-map", "0:v:0", "-map", "0:a:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                "-af", AUDIO_FILTERS,
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(output),
            ],
            timeout=900,
        )
    else:
        run_command(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-map", "0:v:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                "-movflags", "+faststart",
                str(output),
            ],
            timeout=900,
        )

    output_duration = get_duration(output)
    removed_seconds = max(0.0, duration - output_duration) if silences and audio_present else 0.0

    return {
        "status": "processed",
        "job_id": job_id,
        "download_url": f"/download/{job_id}",
        "editing": ["long_silence_cut", "audio_noise_reduction", "loudness_normalization"]
        if audio_present and silences
        else (["audio_noise_reduction", "loudness_normalization"] if audio_present else ["video_transcode"]),
        "original_duration_seconds": round(duration, 2),
        "output_duration_seconds": round(output_duration, 2),
        "removed_silence_seconds": round(removed_seconds, 2),
        "silence_intervals_detected": len(silences) if audio_present else 0,
        "audio_enhancement": {
            "enabled": audio_present,
            "filters": ["afftdn (light noise reduction)", "loudnorm (loudness normalization)"] if audio_present else [],
            "note": "This first-pass preset can reduce steady background noise and even out loudness; it does not reliably remove strong echo or identify repeated words/bad takes."
        },
        "message": (
            "Long quiet sections were cut where detected; light noise reduction and loudness normalization were applied. "
            "Repeated words, bad takes, and strong room echo are not automatically fixed yet."
            if audio_present and silences
            else (
                "Light noise reduction and loudness normalization were applied. "
                "Repeated words, bad takes, and strong room echo are not automatically fixed yet."
                if audio_present
                else "No audio stream was found; the video was converted to MP4 without audio enhancement."
            )
        ),
    }


@app.get("/download/{job_id}")
def download_video(job_id: str):
    output = OUTPUT_DIR / f"{job_id}_processed.mp4"
    if not output.is_file():
        raise HTTPException(status_code=404, detail="Processed video not found.")

    return FileResponse(
        output,
        media_type="video/mp4",
        filename="ai_edivio_processed.mp4",
    )
