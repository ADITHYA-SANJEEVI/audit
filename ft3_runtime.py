import os, re, json, time, hashlib, platform, subprocess, sys, gc
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from jiwer import wer
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    VitsModel,
    AutoTokenizer,
)

OUT = Path("ft3_output")
OUT.mkdir(exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

REGNO = "RA2311003010745"
SOURCE_COMMIT = "797ce2f3132b8cb9cc17ff12b3d0e9d977a2f432"
SOURCE_AUDIO_PATH = "Audios/596658_english_3_paragraph_blob.wav"
SOURCE_TRANSCRIPT_PATH = "Transcripts/9_596658_E3_7.txt"
RAW_BASE = f"https://raw.githubusercontent.com/projectboli/Project_Boli_Dataset/{SOURCE_COMMIT}"
SOURCE_AUDIO_URL = f"{RAW_BASE}/{SOURCE_AUDIO_PATH}"
SOURCE_TRANSCRIPT_URL = f"{RAW_BASE}/{SOURCE_TRANSCRIPT_PATH}"

RAW_AUDIO = OUT / "ProjectBoli_596658_E3_source_blob.wav"
RAW_TRANSCRIPT = OUT / "ProjectBoli_596658_E3_annotations.txt"
AUDIO_PATH = OUT / f"{REGNO}_original_disfluent.wav"
TTS_PATH = OUT / f"{REGNO}_TTS_output.wav"

VERBATIM = (
    "throughout the c c centuries people have explained the rainbow "
    "uh in v various ways some have accepted it as a miracle without "
    "the physical explanation"
)
CLEAN = (
    "throughout the centuries people have explained the rainbow in "
    "various ways some have accepted it as a miracle without the physical explanation"
)

EVENTS = [
    {"type": "Sound repetition (SR)", "start": 1.990308, "end": 6.352452, "surface": "c c centuries"},
    {"type": "Interjection (IN)", "start": 9.506422, "end": 10.275838, "surface": "uh in"},
    {"type": "Sound repetition (SR)", "start": 10.275838, "end": 11.394989, "surface": "v various"},
]

def sh(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def alignment(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    back = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        dp[i][0]=i; back[i][0]="D"
    for j in range(1,m+1):
        dp[0][j]=j; back[0][j]="I"
    for i in range(1,n+1):
        for j in range(1,m+1):
            if r[i-1] == h[j-1]:
                dp[i][j]=dp[i-1][j-1]; back[i][j]="C"
            else:
                opts=[(dp[i-1][j-1]+1,"S"),(dp[i-1][j]+1,"D"),(dp[i][j-1]+1,"I")]
                dp[i][j],back[i][j]=min(opts,key=lambda x:x[0])
    rows=[]; i=n; j=m
    counts={"C":0,"S":0,"D":0,"I":0}
    while i>0 or j>0:
        op=back[i][j]
        if op in ("C","S"):
            rows.append({"reference":r[i-1],"hypothesis":h[j-1],"op":op}); i-=1; j-=1
        elif op=="D":
            rows.append({"reference":r[i-1],"hypothesis":"—","op":"D"}); i-=1
        else:
            rows.append({"reference":"—","hypothesis":h[j-1],"op":"I"}); j-=1
        counts[op]+=1
    rows.reverse()
    counts["N"]=n
    counts["WER_manual"]=(counts["S"]+counts["D"]+counts["I"])/n if n else 0.0
    return rows, counts

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

# 1) Download exact public dataset assets, pinned to a commit.
for url, dest in [(SOURCE_AUDIO_URL, RAW_AUDIO), (SOURCE_TRANSCRIPT_URL, RAW_TRANSCRIPT)]:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
print("Downloaded Project Boli source:", RAW_AUDIO, RAW_AUDIO.stat().st_size, "bytes")

# 2) Convert and crop to a 20-second, mono, 16 kHz PCM WAV.
sh(["ffmpeg","-hide_banner","-loglevel","error","-y",
    "-i",str(RAW_AUDIO),"-ss","0","-t","19.5",
    "-ac","1","-ar","16000","-c:a","pcm_s16le",str(AUDIO_PATH)])

probe = subprocess.check_output(
    ["ffprobe","-v","error","-show_entries","format=duration:stream=codec_name,sample_rate,channels",
     "-of","json",str(AUDIO_PATH)], text=True
)
(OUT/"ffprobe.json").write_text(probe, encoding="utf-8")

y, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
y = librosa.util.normalize(y)
sf.write(AUDIO_PATH, y, sr, subtype="PCM_16")
duration = len(y)/sr
print("Prepared source WAV:", sr, "Hz,", duration, "s")

# 3) Signal figures: waveform, STFT, log-Mel.
t = np.arange(len(y))/sr
plt.figure(figsize=(12,3.3))
plt.plot(t,y,linewidth=.65)
for ev in EVENTS:
    plt.axvspan(ev["start"], ev["end"], alpha=.18, label=f'{ev["type"]}: {ev["surface"]}')
plt.xlabel("Time (s)"); plt.ylabel("Normalized amplitude")
plt.title("Original disfluent speech waveform")
plt.legend(loc="upper right", fontsize=8)
plt.grid(alpha=.2)
save_plot(FIG/"waveform_annotated.png")

D = librosa.stft(y, n_fft=400, win_length=400, hop_length=160, window="hann")
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure(figsize=(12,4))
librosa.display.specshow(S_db, sr=sr, hop_length=160, x_axis="time", y_axis="hz")
for ev in EVENTS: plt.axvspan(ev["start"],ev["end"],alpha=.12)
plt.colorbar(format="%+2.0f dB")
plt.title("STFT magnitude spectrum (25 ms window, 10 ms hop, n_fft=400)")
save_plot(FIG/"stft.png")

mel = librosa.feature.melspectrogram(
    y=y,sr=sr,n_fft=400,win_length=400,hop_length=160,
    window="hann",n_mels=80,power=2.0
)
mel_db = librosa.power_to_db(mel, ref=np.max)
plt.figure(figsize=(12,4))
librosa.display.specshow(mel_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel")
for ev in EVENTS: plt.axvspan(ev["start"],ev["end"],alpha=.12)
plt.colorbar(format="%+2.0f dB")
plt.title("80-band log-Mel spectrogram")
save_plot(FIG/"logmel_original.png")

# 4) REAL Wav2Vec2 CTC inference.
W2V_ID = "facebook/wav2vec2-base-960h"
print("Loading", W2V_ID)
w2v_proc = Wav2Vec2Processor.from_pretrained(W2V_ID)
w2v_model = Wav2Vec2ForCTC.from_pretrained(W2V_ID)
w2v_model.eval()
inputs = w2v_proc(y, sampling_rate=sr, return_tensors="pt")
t0=time.perf_counter()
with torch.inference_mode():
    logits = w2v_model(**inputs).logits
w2v_seconds=time.perf_counter()-t0
pred_ids = torch.argmax(logits, dim=-1)[0]
w2v_text = norm(w2v_proc.decode(pred_ids))
print("Wav2Vec2:", w2v_text)
print("Inference seconds:", w2v_seconds)

blank_id = w2v_proc.tokenizer.pad_token_id
ids = pred_ids.cpu().tolist()
tokens = [w2v_proc.tokenizer.convert_ids_to_tokens(i) for i in ids]
display_tokens = ["_" if i == blank_id else tok for i,tok in zip(ids,tokens)]

collapsed_ids=[]
prev=None
for i in ids:
    if i != prev:
        collapsed_ids.append(i)
    prev=i
no_blank_ids=[i for i in collapsed_ids if i != blank_id]
collapsed_tokens=[w2v_proc.tokenizer.convert_ids_to_tokens(i) for i in collapsed_ids]
collapsed_display=["_" if i==blank_id else tok for i,tok in zip(collapsed_ids,collapsed_tokens)]
no_blank_tokens=[w2v_proc.tokenizer.convert_ids_to_tokens(i) for i in no_blank_ids]

(OUT/"ctc_frame_tokens.txt").write_text(" ".join(display_tokens), encoding="utf-8")
(OUT/"ctc_after_merge.txt").write_text(" ".join(collapsed_display), encoding="utf-8")
(OUT/"ctc_after_blank_removal.txt").write_text(" ".join(no_blank_tokens), encoding="utf-8")
pd.DataFrame({
    "frame":np.arange(len(ids)),
    "time_s":np.arange(len(ids))*duration/len(ids),
    "token_id":ids,
    "token":display_tokens,
}).to_csv(OUT/"ctc_frame_tokens.csv", index=False)

# 5) REAL Whisper-small encoder-decoder inference.
del w2v_model, logits, inputs
gc.collect()
WHISPER_ID="openai/whisper-small"
print("Loading", WHISPER_ID)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_ID, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_proc = AutoProcessor.from_pretrained(WHISPER_ID)
whisper_asr = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_proc.tokenizer,
    feature_extractor=whisper_proc.feature_extractor,
    device=-1,
)
t0=time.perf_counter()
whisper_raw=whisper_asr(
    {"array":y,"sampling_rate":sr},
    generate_kwargs={"language":"english","task":"transcribe"},
)
whisper_seconds=time.perf_counter()-t0
whisper_text=norm(whisper_raw["text"])
print("Whisper:", whisper_text)
print("Inference seconds:", whisper_seconds)

# 6) WER against verbatim + clean. Manual alignment + jiwer cross-check.
metrics={}
for name,hyp in [("wav2vec2",w2v_text),("whisper",whisper_text)]:
    metrics[name]={}
    for refname,ref in [("verbatim",VERBATIM),("clean",CLEAN)]:
        rows, counts=alignment(ref,hyp)
        score=float(wer(norm(ref),norm(hyp)))
        metrics[name][refname]={
            "wer":score,
            "manual":counts,
            "alignment":rows,
            "jiwer_matches_manual":abs(score-counts["WER_manual"])<1e-12,
        }
        pd.DataFrame(rows).to_csv(OUT/f"alignment_{name}_{refname}.csv", index=False)

best_name=min(["wav2vec2","whisper"], key=lambda n:metrics[n]["clean"]["wer"])
best_text=w2v_text if best_name=="wav2vec2" else whisper_text
print("Best clean-reference ASR:", best_name, metrics[best_name]["clean"]["wer"])

comparison=pd.DataFrame([
    {
      "Model":"Wav2Vec2",
      "Architecture":"CNN feature encoder + Transformer encoder + CTC head",
      "Decoding":"CTC greedy argmax",
      "WER_verbatim":metrics["wav2vec2"]["verbatim"]["wer"],
      "WER_clean":metrics["wav2vec2"]["clean"]["wer"],
      "Inference_s":w2v_seconds,
    },
    {
      "Model":"Whisper-small",
      "Architecture":"Log-Mel + encoder-decoder Transformer",
      "Decoding":"Autoregressive attention",
      "WER_verbatim":metrics["whisper"]["verbatim"]["wer"],
      "WER_clean":metrics["whisper"]["clean"]["wer"],
      "Inference_s":whisper_seconds,
    },
])
comparison.to_csv(OUT/"asr_comparison.csv",index=False)

plt.figure(figsize=(8,3.8))
x=np.arange(2); width=.36
plt.bar(x-width/2,comparison["WER_verbatim"]*100,width,label="Verbatim")
plt.bar(x+width/2,comparison["WER_clean"]*100,width,label="Clean")
plt.xticks(x,comparison["Model"])
plt.ylabel("WER (%)"); plt.title("ASR WER comparison")
plt.legend(); plt.grid(axis="y",alpha=.2)
save_plot(FIG/"wer_comparison.png")

# 7) REAL local TTS from the best real ASR transcript.
TTS_ID="facebook/mms-tts-eng"
print("Loading",TTS_ID)
tts_tokenizer=AutoTokenizer.from_pretrained(TTS_ID)
tts_model=VitsModel.from_pretrained(TTS_ID)
tts_model.eval()
tts_inputs=tts_tokenizer(best_text,return_tensors="pt")
t0=time.perf_counter()
with torch.inference_mode():
    tts_waveform=tts_model(**tts_inputs).waveform[0].cpu().numpy()
tts_seconds=time.perf_counter()-t0
native_tts_sr=int(tts_model.config.sampling_rate)
if native_tts_sr != 16000:
    tts_waveform=librosa.resample(tts_waveform,orig_sr=native_tts_sr,target_sr=16000)
tts_y=librosa.util.normalize(tts_waveform.astype(np.float32))
tts_sr=16000
sf.write(TTS_PATH,tts_y,tts_sr,subtype="PCM_16")
tts_duration=len(tts_y)/tts_sr
print("TTS input:",best_text)
print("TTS seconds:",tts_seconds,"duration:",tts_duration)

# 8) REAL round-trip ASR. Use the same best system chosen by clean-reference WER.
if best_name=="whisper":
    t0=time.perf_counter()
    rt_raw=whisper_asr(
        {"array":tts_y,"sampling_rate":tts_sr},
        generate_kwargs={"language":"english","task":"transcribe"},
    )
    rt_seconds=time.perf_counter()-t0
    roundtrip_text=norm(rt_raw["text"])
else:
    del whisper_asr, whisper_model
    gc.collect()
    w2v_proc_rt=Wav2Vec2Processor.from_pretrained(W2V_ID)
    w2v_model_rt=Wav2Vec2ForCTC.from_pretrained(W2V_ID)
    inp=w2v_proc_rt(tts_y,sampling_rate=tts_sr,return_tensors="pt")
    t0=time.perf_counter()
    with torch.inference_mode():
        out=w2v_model_rt(**inp).logits
    rt_seconds=time.perf_counter()-t0
    roundtrip_text=norm(w2v_proc_rt.decode(torch.argmax(out,dim=-1)[0]))

roundtrip_wer=float(wer(norm(best_text),norm(roundtrip_text)))
print("Round-trip transcript:",roundtrip_text)
print("Round-trip WER:",roundtrip_wer)

# 9) TTS comparison figures.
t2=np.arange(len(tts_y))/tts_sr
plt.figure(figsize=(12,3))
plt.plot(t,y,linewidth=.55,label="Original")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.title("Original disfluent speech waveform")
plt.grid(alpha=.2)
save_plot(FIG/"waveform_original.png")

plt.figure(figsize=(12,3))
plt.plot(t2,tts_y,linewidth=.55,label="TTS")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.title("TTS output waveform")
plt.grid(alpha=.2)
save_plot(FIG/"waveform_tts.png")

tts_mel=librosa.feature.melspectrogram(
    y=tts_y,sr=tts_sr,n_fft=400,win_length=400,hop_length=160,
    window="hann",n_mels=80,power=2.0)
tts_mel_db=librosa.power_to_db(tts_mel,ref=np.max)
plt.figure(figsize=(12,4))
librosa.display.specshow(tts_mel_db,sr=tts_sr,hop_length=160,x_axis="time",y_axis="mel")
plt.colorbar(format="%+2.0f dB"); plt.title("TTS output 80-band log-Mel spectrogram")
save_plot(FIG/"logmel_tts.png")

plt.figure(figsize=(6,3.5))
plt.bar(["Original","TTS"],[duration,tts_duration])
plt.ylabel("Duration (s)"); plt.title("Original vs TTS duration")
plt.grid(axis="y",alpha=.2)
save_plot(FIG/"duration_comparison.png")

# 10) Reproducibility/provenance bundle.
versions={
  "python":sys.version,
  "platform":platform.platform(),
  "torch":torch.__version__,
  "transformers":__import__("transformers").__version__,
  "librosa":librosa.__version__,
  "jiwer":__import__("jiwer").__version__ if hasattr(__import__("jiwer"),"__version__") else "installed",
}
results={
  "student":{"name":"Adithya Sanjeevi","regno":REGNO},
  "source":{
    "dataset":"Project Boli",
    "citation":"A. Batra, M. Narang, N. K. Sharma and P. K. Das, Boli: A dataset for understanding stuttering experience and analyzing stuttered speech, ICASSP 2025, DOI: 10.1109/ICASSP49660.2025.10888349",
    "repository":"projectboli/Project_Boli_Dataset",
    "commit":SOURCE_COMMIT,
    "audio_repo_path":SOURCE_AUDIO_PATH,
    "transcript_repo_path":SOURCE_TRANSCRIPT_PATH,
    "raw_audio_sha256":sha256(RAW_AUDIO),
    "prepared_audio_sha256":sha256(AUDIO_PATH),
    "segment_start_s":0.0,
    "segment_end_s":19.5,
    "sample_rate_hz":sr,
    "channels":1,
    "duration_s":duration,
  },
  "references":{"verbatim":VERBATIM,"clean":CLEAN},
  "events":EVENTS,
  "wav2vec2":{
    "model_id":W2V_ID,"transcript":w2v_text,"inference_s":w2v_seconds,
    "n_frames":len(ids),"blank_token_id":blank_id,
    "wer_verbatim":metrics["wav2vec2"]["verbatim"]["wer"],
    "wer_clean":metrics["wav2vec2"]["clean"]["wer"],
  },
  "whisper":{
    "model_id":WHISPER_ID,"transcript":whisper_text,"inference_s":whisper_seconds,
    "wer_verbatim":metrics["whisper"]["verbatim"]["wer"],
    "wer_clean":metrics["whisper"]["clean"]["wer"],
  },
  "best_asr":best_name,
  "tts":{"engine":"facebook/mms-tts-eng (VITS)","model_id":TTS_ID,"input_text":best_text,"inference_s":tts_seconds,"duration_s":tts_duration,"wav_sha256":sha256(TTS_PATH)},
  "roundtrip":{"model":best_name,"transcript":roundtrip_text,"wer":roundtrip_wer,"inference_s":rt_seconds},
  "versions":versions,
}
(OUT/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
(OUT/"events.csv").write_text(pd.DataFrame(EVENTS).to_csv(index=False),encoding="utf-8")
(OUT/"references.txt").write_text("VERBATIM:\n"+VERBATIM+"\n\nCLEAN:\n"+CLEAN+"\n",encoding="utf-8")
(OUT/"provenance.txt").write_text(
    f"Dataset: Project Boli\nRepository: projectboli/Project_Boli_Dataset\nCommit: {SOURCE_COMMIT}\n"
    f"Audio: {SOURCE_AUDIO_PATH}\nAnnotations: {SOURCE_TRANSCRIPT_PATH}\n"
    f"Segment: 0.000-19.500 s\nPrepared WAV SHA256: {sha256(AUDIO_PATH)}\n",
    encoding="utf-8"
)
print(json.dumps(results,indent=2))

# workflow trigger
