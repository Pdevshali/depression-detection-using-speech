# app.py
import streamlit as st
import requests
import io
import numpy as np
import matplotlib.pyplot as plt

from audiorecorder import audiorecorder  # pip install streamlit-audiorecorder

API_URL = "http://localhost:8000/predict-audio"  # change if running elsewhere

st.set_page_config(page_title="Voice-based Depression Screening", page_icon="🧠")
st.title("🧠 Voice-based Depression Screening – One Long Response")

st.markdown("""
**Instructions**

Please answer the following questions in **one continuous speech recording** (about 2–5 minutes):

1. Can you briefly introduce yourself?  
2. How are you feeling these days?  
3. What is your daily routine like?  
4. What things make you feel stressed or low?  
5. How do you usually relax or cope with stress?  
6. How has your sleep been recently?  
7. Is there anything you regret or worry about?

You can either:
- **Upload** a `.wav` file, or  
- **Record** your voice directly here.
""")

st.markdown("---")

# ---- Choose input mode ----
mode = st.radio("Choose input method:", ["Upload .wav file", "Record in browser"])

audio_bytes = None
filename = None

if mode == "Upload .wav file":
    st.subheader("📁 Upload your recording (.wav)")
    audio_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if audio_file is not None:
        filename = audio_file.name
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

elif mode == "Record in browser":
    st.subheader("🎙️ Record your voice here")
    st.write("Click to record, speak your full answer, then click again to stop.")
    audio = audiorecorder("Start recording", "Stop recording")

    if len(audio) > 0:
        # audio is a pydub.AudioSegment
        st.audio(audio.tobytes(), format="audio/wav")
        audio_bytes = audio.tobytes()
        filename = "recorded_audio.wav"

st.markdown("---")

if st.button("Analyze my voice"):
    if audio_bytes is None:
        st.warning("Please upload or record audio first.")
    else:
        with st.spinner("Sending audio to backend and analyzing..."):
            files = {"file": (filename, audio_bytes, "audio/wav")}
            try:
                resp = requests.post(API_URL, files=files, timeout=120)
                data = resp.json()
            except Exception as e:
                st.error(f"Error contacting backend: {e}")
                data = None

        if data:
            if not data.get("ok", False):
                st.error(f"Prediction failed: {data.get('reason')}")
                if "detail" in data:
                    st.info(data["detail"])
            else:
                label = data["label"]
                pos_prob = data["pos_prob"]
                thr = data["threshold"]

                st.markdown("## 🧾 Result")
                st.write(f"**Predicted label:** `{label}`")
                st.write(f"**Estimated probability of depression:** `{pos_prob:.3f}` (threshold `{thr:.3f}`)")
                st.write(f"**Chunks analyzed:** {data['n_chunks']}")
                st.write(f"**Short chunks skipped:** {data['skipped_short']}")

                st.markdown("""
> ⚠️ **Important:**  
> This is a research prototype, **not a medical diagnosis**.  
> If you feel distressed or are concerned about your mental health, please contact a mental health professional.
""")

                # Plot chunk-level probabilities
                chunk_probs = np.array(data["chunk_pos_probs"])
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(chunk_probs, marker="o")
                ax.axhline(thr, linestyle="--", label="Threshold")
                ax.set_xlabel("Chunk index")
                ax.set_ylabel("P(Depressed)")
                ax.set_title("Per-chunk ensemble depression probability")
                ax.legend()
                st.pyplot(fig)
