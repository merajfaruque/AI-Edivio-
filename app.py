import os, re, uuid, subprocess, json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="AI Edivio API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
BASE = Path(os.getenv("AI_EDIVIO_DIR", "/tmp/ai_edivio"))
UPLOADS, OUTPUTS, ASSETS = BASE/"uploads", BASE/"outputs", BASE/"assets"
for folder in (UPLOADS, OUTPUTS, ASSETS): folder.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_BYTES", str(200*1024*1024)))
SILENCE_DB = os.getenv("SILENCE_NOISE_DB", "-35dB")
MIN_SILENCE = float(os.getenv("MIN_SILENCE_SECONDS", "0.65"))
MARGIN = float(os.getenv("KEEP_MARGIN_SECONDS", "0.12"))


def run(cmd, timeout=900):
    try: p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError: raise HTTPException(500, "FFmpeg/ffprobe is not installed on the server.")
    except subprocess.TimeoutExpired: raise HTTPException(504, "Processing timed out. Try a shorter video.")
    if p.returncode != 0: raise HTTPException(500, "FFmpeg failed: " + (p.stderr or "unknown error")[-1500:])
    return p


def duration(path: Path) -> float:
    p = run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)], 60)
    try: d=float(p.stdout.strip())
    except Exception: raise HTTPException(400,"Could not read this video's duration.")
    if d <= 0: raise HTTPException(400,"Video duration is invalid.")
    return d


def has_audio(path: Path) -> bool:
    p = run(["ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=codec_type","-of","csv=p=0",str(path)],60)
    return bool(p.stdout.strip())


def detect_silences(path: Path):
    total=duration(path)
    if not has_audio(path): return total, []
    p=subprocess.run(["ffmpeg","-hide_banner","-i",str(path),"-vn","-af",f"silencedetect=noise={SILENCE_DB}:d={MIN_SILENCE}","-f","null","-"],capture_output=True,text=True,timeout=600)
    log=p.stderr or ""
    starts=[float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)",log)]
    ends=[float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)",log)]
    spans=[]
    for i,s in enumerate(starts):
        e=ends[i] if i<len(ends) else total
        s=max(0,min(s,total)); e=max(s,min(e,total))
        if e-s >= MIN_SILENCE: spans.append((s,e))
    merged=[]
    for s,e in sorted(spans):
        if merged and s <= merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],e))
        else: merged.append((s,e))
    return total,merged


def keep_segments(total, silences):
    if not silences: return [(0.0,total)]
    out=[]; cursor=0.0
    for s,e in silences:
        cut_s=max(cursor,s-MARGIN); cut_e=min(total,e+MARGIN)
        if cut_s-cursor >= .08: out.append((cursor,cut_s))
        cursor=max(cursor,cut_e)
    if total-cursor >= .08: out.append((cursor,total))
    return out or [(0.0,total)]

async def save_upload(file: UploadFile, path: Path, allowed_exts):
    ext=Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts: raise HTTPException(400,"Unsupported file type.")
    size=0
    try:
        with path.open("wb") as f:
            while True:
                chunk=await file.read(1024*1024)
                if not chunk: break
                size += len(chunk)
                if size > MAX_UPLOAD: raise HTTPException(413,"File limit is 200 MB.")
                f.write(chunk)
    except Exception:
        path.unlink(missing_ok=True); raise
    finally: await file.close()
    return size, ext

@app.get("/")
def home(): return {"app":"AI Edivio","status":"running","message":"Video API is ready","version":"2.0.0","features":{"silence_cut":True,"audio_cleanup":"FFmpeg basic denoise + loudness normalization","music_overlay":True,"green_screen_background_replace":True,"captions":"SRT upload/burn-in supported","speech_transcription":"not built into lightweight Render service"}}

@app.get("/health")
def health(): return {"status":"ok","version":"2.0.0"}

@app.post("/upload")
async def upload_video(video: UploadFile=File(...)):
    job=uuid.uuid4().hex
    size,ext=await save_upload(video,UPLOADS/f"{job}{ext_for(video.filename)}",{".mp4",".mov",".webm",".m4v"})
    return {"status":"uploaded","job_id":job,"filename":video.filename,"size_bytes":size,"message":"Video uploaded successfully."}

def ext_for(filename): return Path(filename or "").suffix.lower()

@app.post("/assets/{job_id}/music")
async def upload_music(job_id: str, music: UploadFile=File(...)):
    if not list(UPLOADS.glob(f"{job_id}.*")): raise HTTPException(404,"Upload the video first.")
    size,ext=await save_upload(music,ASSETS/f"{job_id}_music{ext_for(music.filename)}",{".mp3",".wav",".m4a",".aac",".ogg",".flac"})
    return {"status":"uploaded","asset":"music","size_bytes":size}

@app.post("/assets/{job_id}/background")
async def upload_background(job_id: str, background: UploadFile=File(...)):
    if not list(UPLOADS.glob(f"{job_id}.*")): raise HTTPException(404,"Upload the video first.")
    size,ext=await save_upload(background,ASSETS/f"{job_id}_background{ext_for(background.filename)}",{".png",".jpg",".jpeg",".webp"})
    return {"status":"uploaded","asset":"background","size_bytes":size}

@app.post("/assets/{job_id}/captions")
async def upload_captions(job_id: str, captions: UploadFile=File(...)):
    if not list(UPLOADS.glob(f"{job_id}.*")): raise HTTPException(404,"Upload the video first.")
    size,ext=await save_upload(captions,ASSETS/f"{job_id}_captions.srt",{".srt"})
    return {"status":"uploaded","asset":"captions_srt","size_bytes":size}

@app.post("/process/{job_id}")
def process_video(job_id: str, cut_silence: bool=Form(True), clean_audio: bool=Form(True), music_volume: float=Form(0.12), background_mode: str=Form("none"), aspect_ratio: str=Form("original"), caption_style: str=Form("white")):
    matches=list(UPLOADS.glob(f"{job_id}.*"))
    if not matches: raise HTTPException(404,"Video not found. Please upload again.")
    source=matches[0]; total=duration(source); audio=has_audio(source)
    music_candidates=list(ASSETS.glob(f"{job_id}_music.*")); bg_candidates=list(ASSETS.glob(f"{job_id}_background.*")); srt=ASSETS/f"{job_id}_captions.srt"
    if not 0 <= music_volume <= 1: raise HTTPException(400,"Music volume must be between 0 and 1.")
    if background_mode not in {"none","green"}: raise HTTPException(400,"Background mode must be none or green.")
    if background_mode=="green" and not bg_candidates: raise HTTPException(400,"Upload a background image first for green-screen replacement.")
    if aspect_ratio not in {"original","16:9","9:16","1:1"}: raise HTTPException(400,"Unsupported aspect ratio.")
    silences=[]
    if cut_silence and audio: total,silences=detect_silences(source)
    segments=keep_segments(total,silences) if cut_silence else [(0.0,total)]
    out=OUTPUTS/f"{job_id}_processed.mp4"
    # Build synchronized trim/concat graph. With no silence, this still safely transcodes.
    graph=[]; video_labels=[]; audio_labels=[]
    for i,(start,end) in enumerate(segments):
        graph.append(f"[0:v]trim=start={start:.4f}:end={end:.4f},setpts=PTS-STARTPTS[v{i}]")
        video_labels.append(f"[v{i}]")
        if audio:
            graph.append(f"[0:a]atrim=start={start:.4f}:end={end:.4f},asetpts=PTS-STARTPTS[a{i}]")
            audio_labels.append(f"[a{i}]")
    graph.append("".join(video_labels)+f"concat=n={len(segments)}:v=1:a=0[vbase]")
    if audio:
        graph.append("".join(audio_labels)+f"concat=n={len(segments)}:v=0:a=1[abase]")
        if clean_audio: graph.append("[abase]afftdn=nr=10:nf=-50:tn=1,loudnorm=I=-16:LRA=11:TP=-1.5[aclean]")
        else: graph.append("[abase]anull[aclean]")
    # Background replacement works only with a genuinely green/solid keyed backdrop.
    cmd=["ffmpeg","-y","-i",str(source)]
    if background_mode=="green": cmd += ["-loop","1","-i",str(bg_candidates[0])]
    if music_candidates: cmd += ["-stream_loop","-1","-i",str(music_candidates[0])]
    if background_mode=="green":
        graph.append("[1:v][vbase]scale2ref[bg][vref]")
        graph.append("[vref]chromakey=0x00FF00:0.16:0.08[fg]")
        graph.append("[bg][fg]overlay=shortest=1[vkey]")
        vlabel="vkey"
    else: vlabel="vbase"
    if aspect_ratio=="16:9": graph.append(f"[{vlabel}]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[vfinal]"); vlabel="vfinal"
    elif aspect_ratio=="9:16": graph.append(f"[{vlabel}]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[vfinal]"); vlabel="vfinal"
    elif aspect_ratio=="1:1": graph.append(f"[{vlabel}]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[vfinal]"); vlabel="vfinal"
    # Add uploaded SRT if present. Escape path for FFmpeg filter syntax.
    if srt.exists():
        safe=str(srt).replace("\\","/").replace(":","\\:").replace("'","\\'")
        graph.append(f"[{vlabel}]subtitles='{safe}':force_style='FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1'[vcap]")
        vlabel="vcap"
    if music_candidates:
        music_index=2 if background_mode=="green" else 1
        graph.append(f"[{music_index}:a]volume={music_volume:.3f},asetpts=PTS-STARTPTS[amusic]")
        if audio: graph.append("[aclean][amusic]amix=inputs=2:duration=first:dropout_transition=2[aout]")
        else: graph.append("[amusic]anull[aout]")
        alabel="aout"
    else: alabel="aclean" if audio else None
    cmd += ["-filter_complex",";".join(graph),"-map",f"[{vlabel}]"]
    if alabel: cmd += ["-map",f"[{alabel}]"]
    cmd += ["-c:v","libx264","-preset","ultrafast","-crf","22","-pix_fmt","yuv420p"]
    if alabel: cmd += ["-c:a","aac","-b:a","192k"]
    cmd += ["-movflags","+faststart","-t",f"{sum(e-s for s,e in segments)+0.25:.3f}",str(out)]
    run(cmd, timeout=1200)
    out_d=duration(out)
    expected=sum(e-s for s,e in segments)
    if out_d > expected + 1.0:
        out.unlink(missing_ok=True)
        raise HTTPException(500,f"Output duration check failed ({out_d:.2f}s vs expected {expected:.2f}s). Please retry with a shorter clip.")
    return {"status":"processed","job_id":job_id,"download_url":f"/download/{job_id}","original_duration_seconds":round(total,2),"output_duration_seconds":round(out_d,2),"removed_seconds_estimate":round(max(0,total-expected),2),"silence_segments_detected":len(silences),"features":{"silence_cut":bool(cut_silence),"audio_cleanup":bool(clean_audio and audio),"music_added":bool(music_candidates),"green_screen_replaced":background_mode=="green","captions_burned_in":srt.exists(),"aspect_ratio":aspect_ratio},"notes":["Silence detection cannot identify repeated words or decide which retakes are best.","Green-screen replacement requires a green backdrop; it does not remove arbitrary backgrounds.","Audio cleanup is FFmpeg-based basic denoising, not neural voice isolation."]}

@app.get("/download/{job_id}")
def download_video(job_id: str):
    path=OUTPUTS/f"{job_id}_processed.mp4"
    if not path.exists(): raise HTTPException(404,"Processed video not found.")
    return FileResponse(path,media_type="video/mp4",filename="ai_edivio_processed.mp4")

@app.get("/job/{job_id}")
def job_status(job_id: str):
    uploaded=bool(list(UPLOADS.glob(f"{job_id}.*"))); output=OUTPUTS/f"{job_id}_processed.mp4"
    return {"job_id":job_id,"uploaded":uploaded,"processed":output.exists(),"download_url":f"/download/{job_id}" if output.exists() else None}
