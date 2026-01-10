import sys
import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from datetime import timedelta

audio_path = sys.argv[1]
output_path = sys.argv[2]

model_size = os.environ.get("WHISPER_MODEL", "medium")

print("▶ Loading Whisper...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")

print("▶ Transcribing...")
segments, info = model.transcribe(audio_path, language="cs")

print("▶ Loading diarization...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.environ.get("HF_TOKEN")
)

diarization = pipeline(audio_path)

def fmt(ts):
    return str(timedelta(seconds=int(ts)))

print("▶ Writing output...")
with open(output_path, "w", encoding="utf-8") as f:
    for segment in segments:
        start = segment.start
        text = segment.text.strip()

        speaker = "UNKNOWN"
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.start <= start <= turn.end:
                speaker = spk
                break

        f.write(f"[{fmt(start)}] {speaker}:\n{text}\n\n")

print("✅ Hotovo")
