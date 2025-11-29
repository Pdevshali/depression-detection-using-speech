from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from inference import ensemble_predict_from_bytes  # import from your module

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    # ---- 1. Read uploaded bytes ----
    file_bytes = await file.read()

    # ---- 2. Check audio duration BEFORE running model ----
    import soundfile as sf
    import io

    try:
        data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
        duration = len(data) / sr
    except Exception as e:
        return {
            "ok": False,
            "reason": "invalid_audio",
            "detail": f"Could not read audio: {e}"
        }

    # ---- 3. Reject if audio too long ----
    if duration > 1000:      # > 16 minutes
        return {
            "ok": False,
            "reason": "audio_too_long",
            "detail": f"Audio is {duration/60:.1f} minutes. Max allowed is 16 minutes.",
        }

    print(f"📥 Received audio {file.filename}, duration = {duration:.2f}s")

    # ---- 4. Run model inference ----
    result = ensemble_predict_from_bytes(file_bytes)

    print("📤 Finished. Result:", result.get("label"), "Prob:", result.get("pos_prob"))

    return result

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
