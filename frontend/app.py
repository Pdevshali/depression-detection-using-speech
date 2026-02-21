import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

from audiorecorder import audiorecorder  # pip install streamlit-audiorecorder

# ── reportlab (PDF generation) ────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ─────────────────────────────────────────────────────────────
# PDF Helper
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(label, pos_prob, thr, n_chunks, skipped, chunk_probs, questions):
    """Build a professional PDF report in memory and return bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        fontSize=20, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#555577"),
        spaceAfter=4, alignment=TA_CENTER
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#1a1a2e"),
        spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, leading=14, textColor=colors.HexColor("#222222")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=7.5, textColor=colors.HexColor("#888888"),
        leading=11, spaceBefore=10
    )
    verdict_color = colors.HexColor("#c0392b") if label == "Depressed" else colors.HexColor("#1e8449")
    verdict_style = ParagraphStyle(
        "Verdict", parent=styles["Normal"],
        fontSize=13, fontName="Helvetica-Bold",
        textColor=verdict_color, alignment=TA_CENTER, spaceAfter=6
    )

    story = []
    now = datetime.now().strftime("%d %B %Y, %H:%M")

    # ── Title block ──────────────────────────────────────────
    story.append(Paragraph("Voice-Based Depression Screening", title_style))
    story.append(Paragraph("Research Prototype — Screening Report", subtitle_style))
    story.append(Paragraph(f"Generated: {now}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2,
                             color=colors.HexColor("#1a1a2e"), spaceAfter=14))

    # ── Disclaimer banner ─────────────────────────────────────
    disclaimer_text = (
        "<b>IMPORTANT DISCLAIMER:</b> This report is produced by a research prototype "
        "designed for <b>academic and research purposes only</b>. It does <b>NOT</b> "
        "constitute a clinical diagnosis of depression or any other mental health condition. "
        "If you are concerned about your mental health, please consult a qualified "
        "mental health professional."
    )
    disclaimer_table = Table(
        [[Paragraph(disclaimer_text, ParagraphStyle(
            "DB", parent=styles["Normal"], fontSize=8,
            textColor=colors.HexColor("#7d6608"), leading=12
        ))]],
        colWidths=["100%"]
    )
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef9e7")),
        ("BOX",        (0, 0), (-1, -1), 0.8, colors.HexColor("#d4ac0d")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 14))

    # ── Overall Result ────────────────────────────────────────
    story.append(Paragraph("Overall Result", section_style))

    verdict_icon = "HIGH RISK" if label == "Depressed" else "LOW RISK"
    story.append(Paragraph(f"Verdict: {label} ({verdict_icon})", verdict_style))

    result_data = [
        ["Metric", "Value"],
        ["Depression Probability", f"{pos_prob:.4f}"],
        ["Decision Threshold",     f"{thr:.4f}"],
        ["Chunks Analyzed",        str(n_chunks)],
        ["Chunks Skipped",         str(skipped)],
        ["Model Architecture",     "Wav2Vec2 + RF/XGBoost Ensemble"],
    ]
    result_table = Table(result_data, colWidths=[9*cm, 8*cm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 14))

    # ── Interview Questions ───────────────────────────────────
    story.append(Paragraph("Interview Questions Presented", section_style))
    story.append(Paragraph(
        "The participant answered all questions below in a single continuous audio recording.",
        body_style
    ))
    story.append(Spacer(1, 6))
    q_data = [["#", "Question"]] + [[str(i), q] for i, q in enumerate(questions, 1)]
    q_table = Table(q_data, colWidths=[1*cm, 16*cm])
    q_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(q_table)
    story.append(Spacer(1, 14))

    # ── Chunk probability chart ───────────────────────────────
    if chunk_probs:
        story.append(Paragraph("Per-chunk Depression Probability Chart", section_style))
        story.append(Paragraph(
            "The recording was split into 15-second chunks. "
            "Each bar represents the model's estimated depression probability for that segment. "
            "The dashed line marks the decision threshold.",
            body_style
        ))
        story.append(Spacer(1, 6))

        chunk_arr = np.array(chunk_probs)
        fig, ax = plt.subplots(figsize=(7, 2.6))
        fig.patch.set_facecolor("white")
        bar_colors = ["#e74c3c" if p >= thr else "#2ecc71" for p in chunk_arr]
        ax.bar(range(len(chunk_arr)), chunk_arr, color=bar_colors, alpha=0.88, edgecolor="white")
        ax.axhline(thr, linestyle="--", color="#e67e22", linewidth=1.4,
                   label=f"Threshold ({thr:.2f})")
        ax.set_xlabel("Chunk index", fontsize=8)
        ax.set_ylabel("P(Depressed)", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Per-chunk ensemble probability", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        plt.tight_layout()

        chart_buf = BytesIO()
        fig.savefig(chart_buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_buf.seek(0)

        story.append(RLImage(chart_buf, width=16*cm, height=6*cm))
        story.append(Spacer(1, 10))

    # ── Footer disclaimer ────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.6,
                             color=colors.HexColor("#aaaaaa"), spaceBefore=10))
    story.append(Paragraph(
        "This report was automatically generated by a research prototype system using the "
        "facebook/wav2vec2-base-960h model for feature extraction and a Random Forest / "
        "XGBoost ensemble classifier. The system is intended solely for academic research "
        "and must not be used for clinical decision-making. "
        f"Report generated on {now}.",
        disclaimer_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Risk Gauge Helper
# ─────────────────────────────────────────────────────────────
def render_risk_gauge(pos_prob: float, threshold: float):
    """
    Renders an animated SVG semicircular risk gauge via st.components.v1.html.
    The needle sweeps from the leftmost position (0.0) to pos_prob on load.
    Zones: green (0–0.4), amber (0.4–0.6), red (0.6–1.0).
    The threshold is shown as a small orange tick on the arc.
    """
    # Map probability 0–1 → angle -90° to +90° (half circle)
    def prob_to_angle(p):
        return -90 + p * 180

    needle_angle = prob_to_angle(pos_prob)
    thr_angle    = prob_to_angle(threshold)

    # SVG arc path helper: returns a path string for a filled arc segment
    # cx,cy = centre; r = radius; a1,a2 = start/end angles in degrees
    def arc_path(cx, cy, r_outer, r_inner, a1_deg, a2_deg):
        import math
        a1 = math.radians(a1_deg)
        a2 = math.radians(a2_deg)
        x1o = cx + r_outer * math.cos(a1)
        y1o = cy + r_outer * math.sin(a1)
        x2o = cx + r_outer * math.cos(a2)
        y2o = cy + r_outer * math.sin(a2)
        x1i = cx + r_inner * math.cos(a2)
        y1i = cy + r_inner * math.sin(a2)
        x2i = cx + r_inner * math.cos(a1)
        y2i = cy + r_inner * math.sin(a1)
        large = 1 if abs(a2_deg - a1_deg) > 180 else 0
        return (
            f"M {x1o:.2f} {y1o:.2f} "
            f"A {r_outer} {r_outer} 0 {large} 1 {x2o:.2f} {y2o:.2f} "
            f"L {x1i:.2f} {y1i:.2f} "
            f"A {r_inner} {r_inner} 0 {large} 0 {x2i:.2f} {y2i:.2f} Z"
        )

    import math
    cx, cy = 150, 145
    R_OUT, R_IN = 120, 72

    # Three arc zones (angles in SVG degrees, 0° = right, so add 180 offset)
    # Our gauge: left = -90° (prob 0), right = +90° (prob 1)
    # In SVG coords: left = 180°, right = 0°, bottom = 90° → we flip
    def gauge_angle(p):          # returns SVG angle for probability p
        return 180 - (p * 180)   # p=0 → 180° (left), p=1 → 0° (right)

    g0  = gauge_angle(0.0)
    g04 = gauge_angle(0.4)
    g06 = gauge_angle(0.6)
    g1  = gauge_angle(1.0)

    path_green = arc_path(cx, cy, R_OUT, R_IN, g04, g0)    # right→left green
    path_amber = arc_path(cx, cy, R_OUT, R_IN, g06, g04)
    path_red   = arc_path(cx, cy, R_OUT, R_IN, g1,  g06)

    # Threshold tick
    thr_svg_angle = gauge_angle(threshold)
    thr_rad = math.radians(thr_svg_angle)
    tx1 = cx + (R_IN  - 6) * math.cos(thr_rad)
    ty1 = cy + (R_IN  - 6) * math.sin(thr_rad)
    tx2 = cx + (R_OUT + 6) * math.cos(thr_rad)
    ty2 = cy + (R_OUT + 6) * math.sin(thr_rad)

    # Needle end point (at R_OUT - 8)
    needle_svg_angle = gauge_angle(pos_prob)
    nr = math.radians(needle_svg_angle)
    nx = cx + (R_OUT - 10) * math.cos(nr)
    ny = cy + (R_OUT - 10) * math.sin(nr)

    # Risk label & colour
    if pos_prob < 0.4:
        risk_label, risk_color = "Low Risk", "#27ae60"
    elif pos_prob < 0.6:
        risk_label, risk_color = "Borderline", "#e67e22"
    else:
        risk_label, risk_color = "High Risk", "#c0392b"

    html = f"""
    <div style="display:flex; justify-content:center; padding:10px 0;">
    <svg viewBox="0 0 300 175" width="360" xmlns="http://www.w3.org/2000/svg">

      <!-- Zone arcs -->
      <path d="{path_green}" fill="#2ecc71" opacity="0.88"/>
      <path d="{path_amber}" fill="#f39c12" opacity="0.88"/>
      <path d="{path_red}"   fill="#e74c3c" opacity="0.88"/>

      <!-- Inner circle (hub background) -->
      <circle cx="{cx}" cy="{cy}" r="68" fill="#1e1e2e"/>

      <!-- Threshold tick -->
      <line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}"
            stroke="#ffffff" stroke-width="2.5" stroke-dasharray="4,2" opacity="0.9"/>

      <!-- Needle (animated) -->
      <line id="needle" x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="#cdd6f4" stroke-width="3" stroke-linecap="round"
            style="transform-origin:{cx}px {cy}px;"
      />
      <!-- Needle pivot -->
      <circle cx="{cx}" cy="{cy}" r="7" fill="#cdd6f4"/>

      <!-- Zone labels -->
      <text x="32"  y="148" font-size="9" fill="#2ecc71" font-family="sans-serif" font-weight="bold">LOW</text>
      <text x="133" y="42"  font-size="9" fill="#f39c12" font-family="sans-serif" font-weight="bold">MED</text>
      <text x="245" y="148" font-size="9" fill="#e74c3c" font-family="sans-serif" font-weight="bold">HIGH</text>

      <!-- Centre text -->
      <text x="{cx}" y="{cy - 12}" text-anchor="middle" font-size="22"
            font-weight="bold" fill="{risk_color}" font-family="monospace">{pos_prob:.3f}</text>
      <text x="{cx}" y="{cy + 10}" text-anchor="middle" font-size="10"
            fill="{risk_color}" font-family="sans-serif">{risk_label}</text>
      <text x="{cx}" y="{cy + 26}" text-anchor="middle" font-size="8"
            fill="#6c7086" font-family="sans-serif">thr = {threshold:.2f}</text>
    </svg>
    </div>

    <style>
      #needle {{
        animation: swing 1.4s cubic-bezier(0.22, 1, 0.36, 1) forwards;
      }}
      @keyframes swing {{
        0%   {{ transform: rotate({gauge_angle(0.0) - gauge_angle(pos_prob):.1f}deg); }}
        100% {{ transform: rotate(0deg); }}
      }}
    </style>
    """
    components.html(html, height=200)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict-audio"

SCRIPT = [
    "Can you briefly introduce yourself?",
    "How are you feeling these days?",
    "What is your daily routine like?",
    "What things make you feel stressed or low?",
    "How do you usually relax or cope with stress?",
    "How has your sleep been recently?",
    "Is there anything you regret or worry about these days?"
]

# ─────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Depression Screening Chatbot", page_icon="🧠")
st.title("🧠 Voice-based Depression Screening Chatbot")

st.markdown("""
This is a **research prototype** that uses your **voice** to estimate signs of depression.
It is **NOT** a clinical diagnosis.

Read all the questions, then record **one continuous audio** answering them all in order and submit.

> ⚠️ If you are in crisis or feeling unsafe, please contact a mental health professional or local helpline.
""")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "result" not in st.session_state:
    st.session_state.result = None
if "chunk_probs" not in st.session_state:
    st.session_state.chunk_probs = []
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.45

# ─────────────────────────────────────────────────────────────
# Progress bar  (3 steps)
# ─────────────────────────────────────────────────────────────
_STEPS = ["▶️ Start Interview", "🎙️ Record & Submit", "📊 Results"]

if not st.session_state.interview_started:
    _step_idx = 0
elif not st.session_state.analyzed:
    _step_idx = 1
else:
    _step_idx = 2

_progress = _step_idx / (len(_STEPS) - 1) if _step_idx > 0 else 0.0

_col_bar, _col_label = st.columns([4, 1])
with _col_bar:
    st.progress(_progress)
with _col_label:
    if _step_idx == 2:
        st.markdown("✅ **Done!**")
    else:
        st.markdown(f"**Step {_step_idx + 1} / {len(_STEPS)}**")

st.caption(f"Current step: *{_STEPS[_step_idx]}*")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Start / Restart button
# ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    if st.button("▶️ Start / Restart Interview", type="primary"):
        st.session_state.interview_started = True
        st.session_state.analyzed = False
        st.session_state.result = None
        st.session_state.chunk_probs = []
        st.rerun()

if not st.session_state.interview_started:
    st.info("👆 Click **▶️ Start / Restart Interview** above to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# STEP 2 — Show all questions + single audio input
# ─────────────────────────────────────────────────────────────
st.markdown("### 📋 Interview Questions")
st.markdown(
    "Answer **all** the questions below in **one single recording** — "
    "speak each answer in order without stopping."
)

for i, q in enumerate(SCRIPT, start=1):
    st.markdown(f"**Q{i}.** {q}")

st.markdown("---")

if not st.session_state.analyzed:
    st.markdown("### 🎙️ Record or Upload Your Full Answer")

    mode = st.radio(
        "Choose your input method:",
        ["Upload .wav file", "Record in browser"],
        horizontal=True,
    )

    audio_bytes = None
    filename = None

    if mode == "Upload .wav file":
        audio_file = st.file_uploader("Upload a WAV file with all your answers", type=["wav"])
        if audio_file is not None:
            filename = audio_file.name
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")

    elif mode == "Record in browser":
        st.write(
            "Click **🎤 Start recording**, answer all questions in order, "
            "then click **⏹ Stop recording**."
        )
        recorded_audio = audiorecorder("🎤 Start recording", "⏹ Stop recording")

        if recorded_audio is not None and len(recorded_audio) > 0:
            buffer = BytesIO()
            recorded_audio.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()
            filename = "full_interview_recording.wav"
            st.audio(audio_bytes, format="audio/wav")

    st.markdown("---")

    # ── Analyze button ────────────────────────────────────
    if st.button("🔍 Analyze All Answers", type="primary", use_container_width=True):
        if audio_bytes is None:
            st.warning("⚠️ Please upload or record your audio first.")
        else:
            with st.spinner("Analyzing your voice — this may take a moment..."):
                files = {"file": (filename, audio_bytes, "audio/wav")}
                try:
                    resp = requests.post(API_URL, files=files, timeout=1000)
                    data = resp.json()
                except requests.exceptions.ReadTimeout as e:
                    st.error(f"Request timed out: {e}")
                    data = None
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")
                    data = None

            if data is None:
                st.error("No response received. Is the backend server running?")
            elif not data.get("ok", False):
                reason = data.get("reason", "unknown")
                detail = data.get("detail", "")
                st.error(
                    f"⚠️ Could not analyze the audio.\n\n"
                    f"**Reason:** {reason}\n\n{detail}"
                )
            else:
                st.session_state.result = data
                st.session_state.chunk_probs = data.get("chunk_pos_probs", [])
                st.session_state.threshold = data.get("threshold", 0.45)
                st.session_state.analyzed = True
                st.rerun()

# ─────────────────────────────────────────────────────────────
# STEP 3 — Results
# ─────────────────────────────────────────────────────────────
if st.session_state.analyzed and st.session_state.result:
    data     = st.session_state.result
    label    = data.get("label", "Unknown")
    pos_prob = data.get("pos_prob", 0.0)
    thr      = data.get("threshold", 0.45)
    n_chunks = data.get("n_chunks", 0)
    skipped  = data.get("skipped_short", 0)

    st.markdown("### 📊 Results")

    # ── Metric cards ─────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Probability", f"{pos_prob:.3f}")
    c2.metric("Decision Threshold", f"{thr:.3f}")
    c3.metric("Chunks Analyzed", n_chunks)

    # ── Risk gauge ────────────────────────────────────────
    render_risk_gauge(pos_prob, thr)

    # ── Verdict box ───────────────────────────────────────
    if label == "Depressed":
        st.error(
            f"🔴 **Verdict: {label}** — Probability `{pos_prob:.3f}` is **above** "
            f"the threshold `{thr:.3f}`.\n\n"
            "This is a screening signal only and **NOT a clinical diagnosis**. "
            "If you resonate with this result, please consider speaking with a qualified "
            "mental health professional."
        )
    else:
        st.success(
            f"🟢 **Verdict: {label}** — Probability `{pos_prob:.3f}` is **below** "
            f"the threshold `{thr:.3f}`.\n\n"
            "Even if the model does not flag high risk, reaching out to someone you trust "
            "is always a good idea if you are feeling low."
        )

    if skipped > 0:
        st.info(f"ℹ️ {skipped} short chunk(s) were skipped (too brief to process reliably).")

    # ── Per-chunk bar chart ───────────────────────────────
    if st.session_state.chunk_probs:
        st.markdown("#### 📈 Per-chunk Depression Probability")
        st.caption(
            "The audio was split into 15-second chunks. Each bar shows the model's "
            "depression probability estimate for that segment. "
            "🔴 Red = above threshold  |  🟢 Green = below threshold."
        )
        chunk_arr = np.array(st.session_state.chunk_probs)
        fig, ax = plt.subplots(figsize=(9, 3))
        bar_colors = ["#e74c3c" if p >= thr else "#2ecc71" for p in chunk_arr]
        ax.bar(range(len(chunk_arr)), chunk_arr, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.axhline(
            thr, linestyle="--", color="#f39c12", linewidth=1.5,
            label=f"Threshold ({thr:.2f})"
        )
        ax.set_xlabel("Chunk index")
        ax.set_ylabel("P(Depressed)")
        ax.set_ylim(0, 1)
        ax.set_title("Per-chunk depression probability across the full recording")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # ── PDF Download ──────────────────────────────────────────────
    st.markdown("#### 📄 Download Report")
    st.caption("Export a professional PDF summary of this screening session.")

    if st.button("📄 Generate & Download PDF Report", use_container_width=True):
        with st.spinner("Building PDF..."):
            pdf_bytes = generate_pdf_report(
                label=label,
                pos_prob=pos_prob,
                thr=thr,
                n_chunks=n_chunks,
                skipped=skipped,
                chunk_probs=st.session_state.chunk_probs,
                questions=SCRIPT
            )
        st.download_button(
            label="⬇️ Click here to download the PDF",
            data=pdf_bytes,
            file_name=f"depression_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer**: This tool is a research prototype for academic purposes only. "
        "It must not be used as a substitute for professional medical or psychiatric evaluation."
    )
