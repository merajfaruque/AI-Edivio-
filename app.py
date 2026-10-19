import os, re, uuid, subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="AI Edivio API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
BASE = Path(os.getenv("AI_EDIVIO_DIR", "/tmp/ai_edivio"))
UPLOADS, OUTPUTS, ASSETS = BASE/"uploads", BASE/"outputs", BASE/"assets"
for p in (UPLOADS, OUTPUTS, ASSETS): p.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_BYTES", str(200*1024*1024)))
SILENCE_DB = os.getenv("SILENCE_NOISE_DB", "-35dB")
MIN_SILENCE = float(os.getenv("MIN_SILENCE_SECONDS", "0.65"))
MARGIN = float(os.getenv("KEEP_MARGIN_SECONDS", "0.12"))
VIDEO_EXTS={".mp4",".mov",".webm",".m4v"}
AUDIO_EXTS={".mp3",".wav",".m4a",".aac",".ogg",".flac"}
IMAGE_EXTS={".png",".jpg",".jpeg",".webp"}

def run(cmd, timeout=900):
    try: p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    except FileNotFoundError: raise HTTPException(500,"FFmpeg/ffprobe is not installed.")
    except subprocess.TimeoutExpired: raise HTTPException(504,"Processing timed out. Try a shorter video.")
    if p.returncode: raise HTTPException(500,"Media processing failed: "+(p.stderr or "unknown error")[-1500:])
    return p

def jobfile(folder, job, pattern=None):
    if not re.fullmatch(r"[a-f0-9]{32}",job): raise HTTPException(400,"Invalid job ID.")
    found=list(folder.glob(pattern or f"{job}.*"))
    return found[0] if found else None

def duration(path):
    p=run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)],60)
    try: d=float(p.stdout.strip())
    except: raise HTTPException(400,"Could not read video duration.")
    if d<=0: raise HTTPException(400,"Video duration is invalid.")
    return d

def has_audio(path):
    return bool(run(["ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=codec_type","-of","csv=p=0",str(path)],60).stdout.strip())

def detect_silences(path):
    total=duration(path)
    if not has_audio(path): return total,[]
    try:
        p=subprocess.run(["ffmpeg","-hide_banner","-i",str(path),"-vn","-af",f"silencedetect=noise={SILENCE_DB}:d={MIN_SILENCE}","-f","null","-"],capture_output=True,text=True,timeout=600)
    except subprocess.TimeoutExpired: raise HTTPException(504,"Silence detection timed out.")
    log=p.stderr or ""
    starts=[float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)",log)]
    ends=[float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)",log)]
    spans=[]
    for i,s in enumerate(starts):
        e=ends[i] if i<len(ends) else total
        if min(total,e)-max(0,s)>=MIN_SILENCE: spans.append((max(0,s),min(total,e)))
    merged=[]
    for s,e in sorted(spans):
        if merged and s<=merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],e))
        else: merged.append((s,e))
    return total,merged

def keep_segments(total, silences):
    if not silences: return [(0.0,total)]
    out=[]; cursor=0.0
    for s,e in silences:
        a=max(cursor,s-MARGIN); b=min(total,e+MARGIN)
        if a-cursor>=.08: out.append((cursor,a))
        cursor=max(cursor,b)
    if total-cursor>=.08: out.append((cursor,total))
    return out or [(0.0,total)]

async def save_upload(file,path,allowed):
    ext=Path(file.filename or "").suffix.lower()
    if ext not in allowed: raise HTTPException(400,"Unsupported file type.")
    size=0
    try:
        with path.open("wb") as f:
            while True:
                chunk=await file.read(1024*1024)
                if not chunk: break
                size+=len(chunk)
                if size>MAX_UPLOAD: raise HTTPException(413,f"Upload limit is {MAX_UPLOAD//1048576} MB.")
                f.write(chunk)
    except Exception:
        path.unlink(missing_ok=True); raise
    finally: await file.close()
    return size

@app.get("/")
def home():
    return {"app":"AI Edivio","status":"running","version":"3.0.0","features":{"silence_cut":True,"basic_audio_cleanup":True,"music":True,"green_screen":True,"srt_captions":True,"color_lighting":True,"speech_transcription":False,"ai_background_removal":False}}

@app.get("/health")
def health(): return {"status":"ok","version":"3.0.0"}

@app.post("/upload")
async def upload_video(video: UploadFile=File(...)):
    ext=Path(video.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS: raise HTTPException(400,"Use MP4, MOV, WEBM, or M4V.")
    job=uuid.uuid4().hex
    size=await save_upload(video,UPLOADS/f"{job}{ext}",VIDEO_EXTS)
    return {"status":"uploaded","job_id":job,"filename":video.filename,"size_bytes":size}

@app.post("/assets/{job_id}/music")
async def upload_music(job_id:str,music:UploadFile=File(...)):
    if not jobfile(UPLOADS,job_id): raise HTTPException(404,"Upload video first.")
    ext=Path(music.filename or "").suffix.lower()
    if ext not in AUDIO_EXTS: raise HTTPException(400,"Unsupported audio format.")
    size=await save_upload(music,ASSETS/f"{job_id}_music{ext}",AUDIO_EXTS)
    return {"status":"uploaded","size_bytes":size}

@app.post("/assets/{job_id}/background")
async def upload_background(job_id:str,background:UploadFile=File(...)):
    if not jobfile(UPLOADS,job_id): raise HTTPException(404,"Upload video first.")
    ext=Path(background.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS: raise HTTPException(400,"Use PNG, JPG, or WEBP.")
    size=await save_upload(background,ASSETS/f"{job_id}_background{ext}",IMAGE_EXTS)
    return {"status":"uploaded","size_bytes":size}

@app.post("/assets/{job_id}/captions")
async def upload_captions(job_id:str,captions:UploadFile=File(...)):
    if not jobfile(UPLOADS,job_id): raise HTTPException(404,"Upload video first.")
    if Path(captions.filename or "").suffix.lower()!=".srt": raise HTTPException(400,"Captions must be .srt.")
    size=await save_upload(captions,ASSETS/f"{job_id}_captions.srt",{".srt"})
    return {"status":"uploaded","size_bytes":size}

@app.post("/process/{job_id}")
async def process_video(job_id:str,cut_silence:bool=Form(True),clean_audio:bool=Form(True),
 music_volume:float=Form(.12),background_mode:str=Form("none"),aspect_ratio:str=Form("original"),
 auto_enhance:bool=Form(False),brightness:float=Form(0),contrast:float=Form(1),
 saturation:float=Form(1),sharpen:float=Form(0)):
    source=jobfile(UPLOADS,job_id)
    if not source: raise HTTPException(404,"Video not found. Upload again.")
    if not 0<=music_volume<=1: raise HTTPException(400,"Music volume must be 0-1.")
    if background_mode not in {"none","green"}: raise HTTPException(400,"Invalid background mode.")
    if aspect_ratio not in {"original","16:9","9:16","1:1"}: raise HTTPException(400,"Invalid aspect ratio.")
    if auto_enhance: brightness,contrast,saturation,sharpen=.02,1.08,1.08,.30
    if not -.5<=brightness<=.5: raise HTTPException(400,"Brightness range is -0.5 to 0.5.")
    if not 0<=contrast<=2: raise HTTPException(400,"Contrast range is 0 to 2.")
    if not 0<=saturation<=3: raise HTTPException(400,"Saturation range is 0 to 3.")
    if not 0<=sharpen<=2: raise HTTPException(400,"Sharpen range is 0 to 2.")
    total=duration(source); audio=has_audio(source); silences=[]
    if cut_silence and audio: total,silences=await run_in_threadpool(detect_silences,source)
    segments=keep_segments(total,silences) if cut_silence else [(0.0,total)]
    music=next(iter(ASSETS.glob(f"{job_id}_music.*")),None)
    bg=next(iter(ASSETS.glob(f"{job_id}_background.*")),None)
    srt=ASSETS/f"{job_id}_captions.srt"
    if background_mode=="green" and not bg: raise HTTPException(400,"Upload a background image for green-screen replacement.")
    cmd=["ffmpeg","-y","-i",str(source)]; bgidx=None; musicidx=None
    if background_mode=="green": bgidx=1; cmd += ["-loop","1","-i",str(bg)]
    if music: musicidx=2 if bgidx is not None else 1; cmd += ["-stream_loop","-1","-i",str(music)]
    graph=[]; vs=[]; auds=[]
    for i,(s,e) in enumerate(segments):
        graph.append(f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}]"); vs.append(f"[v{i}]")
        if audio:
            graph.append(f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]"); auds.append(f"[a{i}]")
    graph.append("".join(vs)+f"concat=n={len(segments)}:v=1:a=0[vbase]")
    alabel=None
    if audio:
        graph.append("".join(auds)+f"concat=n={len(segments)}:v=0:a=1[abase]")
        graph.append("[abase]"+("afftdn=nr=10:nf=-50:tn=1," if clean_audio else "")+"loudnorm=I=-16:LRA=11:TP=-1.5[aclean]")
        alabel="aclean"
    vlabel="vbase"
    if background_mode=="green":
        graph.append("[1:v][vbase]scale2ref[bgscaled][vref]")
        graph.append("[vref]chromakey=0x00FF00:0.16:0.08[fg]")
        graph.append("[bgscaled][fg]overlay=shortest=1[vkey]"); vlabel="vkey"
    if brightness!=0 or contrast!=1 or saturation!=1:
        graph.append(f"[{vlabel}]eq=brightness={brightness:.4f}:contrast={contrast:.4f}:saturation={saturation:.4f}[vcolor]"); vlabel="vcolor"
    if sharpen>0:
        graph.append(f"[{vlabel}]unsharp=5:5:{sharpen:.3f}:5:5:0[vsharp]"); vlabel="vsharp"
    if aspect_ratio=="16:9":
        graph.append(f"[{vlabel}]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[vfmt]"); vlabel="vfmt"
    elif aspect_ratio=="9:16":
        graph.append(f"[{vlabel}]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[vfmt]"); vlabel="vfmt"
    elif aspect_ratio=="1:1":
        graph.append(f"[{vlabel}]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[vfmt]"); vlabel="vfmt"
    if srt.exists():
        sp=str(srt).replace("\\","/").replace(":","\\:").replace("'","\\'")
        graph.append(f"[{vlabel}]subtitles='{sp}':force_style='FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1'[vcap]"); vlabel="vcap"
    if music:
        graph.append(f"[{musicidx}:a]volume={music_volume:.3f},asetpts=PTS-STARTPTS[amusic]")
        if audio: graph.append("[aclean][amusic]amix=inputs=2:duration=first:dropout_transition=2[aout]")
        else: graph.append(f"[amusic]atrim=duration={sum(e-s for s,e in segments):.3f},asetpts=PTS-STARTPTS[aout]")
        alabel="aout"
    out=OUTPUTS/f"{job_id}_processed.mp4"
    cmd += ["-filter_complex",";".join(graph),"-map",f"[{vlabel}]"]
    if alabel: cmd += ["-map",f"[{alabel}]"]
    cmd += ["-c:v","libx264","-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"]
    if alabel: cmd += ["-c:a","aac","-b:a","160k"]
    expected=sum(e-s for s,e in segments)
    cmd += ["-movflags","+faststart","-t",f"{expected+.25:.3f}",str(out)]
    await run_in_threadpool(run,cmd,1200)
    outdur=duration(out)
    if outdur>expected+1.5:
        out.unlink(missing_ok=True); raise HTTPException(500,"Output duration check failed; try a shorter clip.")
    return {"status":"processed","job_id":job_id,"download_url":f"/download/{job_id}",
      "original_duration_seconds":round(total,2),"output_duration_seconds":round(outdur,2),
      "removed_seconds_estimate":round(max(0,total-expected),2),"silence_segments_detected":len(silences),
      "features":{"silence_cut":bool(cut_silence),"audio_cleanup":bool(clean_audio and audio),
      "music_added":bool(music),"green_screen_replaced":background_mode=="green",
      "captions_burned_in":srt.exists(),"aspect_ratio":aspect_ratio,"auto_enhance":auto_enhance,
      "brightness":brightness,"contrast":contrast,"saturation":saturation,"sharpen":sharpen},
      "notes":["Silence cutting does not detect repeated words or select retakes.",
      "Green-screen replacement requires green-screen footage.","Audio cleanup is basic FFmpeg processing, not neural voice isolation.",
      "Auto Enhance is a fixed preset, not scene-aware AI."]}

@app.get("/download/{job_id}")
def download_video(job_id:str):
    path=OUTPUTS/f"{job_id}_processed.mp4"
    if not re.fullmatch(r"[a-f0-9]{32}",job_id) or not path.exists(): raise HTTPException(404,"Processed video not found.")
    return FileResponse(path,media_type="video/mp4",filename="ai_edivio_processed.mp4")

@app.get("/job/{job_id}")
def job_status(job_id:str):
    uploaded=bool(jobfile(UPLOADS,job_id)); output=OUTPUTS/f"{job_id}_processed.mp4"
    return {"job_id":job_id,"uploaded":uploaded,"processed":output.exists(),"download_url":f"/download/{job_id}" if output.exists() else None}
