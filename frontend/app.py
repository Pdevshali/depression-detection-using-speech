import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from audiorecorder import audiorecorder  # pip install streamlit-audiorecorder

# 👉 Change this if your backend runs elsewhere
API_URL = "http://localhost:8000/predict-audio"

# --------- Interview Script ----------
SCRIPT = [
    "Can you briefly introduce yourself?",
    "How are you feeling these days?",
    "What is your daily routine like?",
    "What things make you feel stressed or low?",
    "How do you usually relax or cope with stress?",
    "How has your sleep been recently?",
    "Is there anything you regret or worry about these days?"
]

st.set_page_config(page_title="Depression Screening Chatbot", page_icon="🧠")
st.title("🧠 Voice-based Depression Screening Chatbot")

st.markdown("""
This is a **research prototype** that uses your **voice** to estimate signs of depression.
It is **NOT** a clinical diagnosis.

You'll be asked a few questions.  
For each question:
1. You can optionally type a short text answer.  
2. Then **speak your answer** and either upload or record it.  
3. I’ll analyze your voice and show the model’s estimate.

> ⚠️ If you are in crisis or feeling unsafe, please contact a mental health professional or local helpline.
""")

st.markdown("---")

# --------- Session state init ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_idx" not in st.session_state:
    st.session_state.question_idx = 0
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "last_chunk_probs" not in st.session_state:
    st.session_state.last_chunk_probs = []
if "last_threshold" not in st.session_state:
    st.session_state.last_threshold = 0.45

# --------- Show chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------- Start / Restart interview ----------
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start / Restart Interview"):
        st.session_state.messages = []
        st.session_state.question_idx = 0
        st.session_state.interview_started = True
        if SCRIPT:
            first_q = SCRIPT[0]
            first_q1 = SCRIPT[1]
            first_q2= SCRIPT[2]
            first_q3= SCRIPT[3]
            first_q4= SCRIPT[4]
            first_q5= SCRIPT[5]
            first_q6= SCRIPT[6]
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "Hi, I'm your voice-based assistant. Let's begin.\n\n"
                    f"**Question 1:** {first_q}\n\n"
                    f"**Question 2:** {first_q1}\n\n"
                    f"**Question 3:** {first_q2}\n\n"
                    f"**Question 4:** {first_q3}\n\n"
                    f"**Question 5:** {first_q4}\n\n"
                    f"**Question 6:** {first_q5}\n\n"
                    f"**Question 7:** {first_q6}\n\n"
                )
            })
        st.rerun()

with col2:
    st.write("")  # spacing

# If interview hasn't started yet, stop here (after showing existing messages + button)
if not st.session_state.interview_started:
    st.stop()

# --------- Determine current question ----------
q_idx = st.session_state.question_idx
if q_idx >= len(SCRIPT):
    with st.chat_message("assistant"):
        st.markdown(
            "We have finished all questions. Thank you for sharing. "
            "You may restart the interview if you wish using the button above."
        )
    st.stop()

current_question = SCRIPT[q_idx]

# Only show current question explicitly if it hasn't been just asked in last message
if not st.session_state.messages or "Question" not in st.session_state.messages[-1]["content"]:
    with st.chat_message("assistant"):
        st.markdown(f"**Question {q_idx+1}:** {current_question}")

# --------- Optional user text answer ----------
user_text = st.chat_input("You can type a brief summary of your answer (optional):")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

st.markdown("### 🎙️ Answer with your voice")

# --------- Choose input method ----------
mode = st.radio("Choose how you want to answer:", ["Upload .wav file", "Record in browser"])

audio_bytes = None
filename = None

if mode == "Upload .wav file":
    audio_file = st.file_uploader("Upload your answer as a WAV file", type=["wav"])
    if audio_file is not None:
        filename = audio_file.name
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

elif mode == "Record in browser":
    st.write("Click to start, answer the question, then click again to stop.")
    recorded_audio = audiorecorder("🎤 Start recording", "⏹ Stop recording")

    # recorded_audio is a pydub.AudioSegment
    if recorded_audio is not None and len(recorded_audio) > 0:
        # Convert AudioSegment -> WAV bytes in memory
        buffer = BytesIO()
        recorded_audio.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()
        filename = "recorded_answer.wav"

        st.audio(audio_bytes, format="audio/wav")

st.markdown("---")

# --------- Send answer ----------
if st.button("Send answer"):
    if audio_bytes is None:
        st.warning("Please upload or record your voice answer first.")
    else:
        # Log that the user sent a voice answer for this question
        st.session_state.messages.append({
            "role": "user",
            "content": f"(Sent a voice answer for Question {q_idx+1}.)"
        })

        with st.spinner("Analyzing your voice..."):
            files = {"file": (filename, audio_bytes, "audio/wav")}
            try:
                resp = requests.post(API_URL, files=files, timeout=120)
                data = resp.json()
            except Exception as e:
                st.error(f"Error contacting backend: {e}")
                data = None

        if data is None:
            st.stop()

        if not data.get("ok", False):
            # Model couldn't analyze (too short, etc.)
            error_msg = f"⚠️ I couldn't analyze your answer. Reason: **{data.get('reason', 'unknown')}**."
            if "detail" in data:
                error_msg += f"\n\nDetails: {data['detail']}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()
        else:
            label = data.get("label", "Unknown")
            pos_prob = data.get("pos_prob", 0.0)
            thr = data.get("threshold", 0.45)
            n_chunks = data.get("n_chunks", 0)
            skipped = data.get("skipped_short", 0)

            # Build assistant response for this answer
            response = (
                f"Thanks for answering **Question {q_idx+1}**.\n\n"
                f"From your **voice** in this answer, the model estimated:\n\n"
                f"- Probability of **depression**: `{pos_prob:.3f}` (threshold = `{thr:.3f}`)\n"
                f"- Chunks analyzed: `{n_chunks}`, skipped short chunks: `{skipped}`\n\n"
            )

            if label == "Depressed":
                response += (
                    "This response is **above** the threshold, so the model flags it as **Depressed (higher risk)**.\n\n"
                    "> ⚠️ This is **not a diagnosis**, only a screening signal. "
                    "If you resonate with this, please consider consulting a mental health professional.\n"
                )
            else:
                response += (
                    "This response is **below** the threshold, so the model does **not** strongly indicate depression.\n\n"
                    "> Still, if you feel low, stressed, or worried, it can help to talk to someone you trust or a professional.\n"
                )

            # Next question or end
            if q_idx + 1 < len(SCRIPT):
                next_q = SCRIPT[q_idx + 1]
                response += f"\n\n**Next Question {q_idx+2}:** {next_q}"
                st.session_state.question_idx += 1
            else:
                response += "\n\nWe have completed all questions. Thank you for sharing."

            st.session_state.messages.append({"role": "assistant", "content": response})

            # Save last chunk probabilities for plotting
            st.session_state.last_chunk_probs = data.get("chunk_pos_probs", [])
            st.session_state.last_threshold = thr

            st.rerun()

# --------- Plot last answer's chunk probabilities (if available) ----------
if st.session_state.last_chunk_probs:
    st.markdown("### 📊 Last answer – per-chunk depression probability")
    chunk_probs = np.array(st.session_state.last_chunk_probs)
    thr = st.session_state.last_threshold
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(chunk_probs, marker="o")
    ax.axhline(thr, linestyle="--", label="Threshold")
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("P(Depressed)")
    ax.set_title("Per-chunk ensemble depression probability (last answer)")
    ax.legend()
    st.pyplot(fig)
