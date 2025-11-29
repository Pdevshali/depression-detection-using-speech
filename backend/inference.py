# inference.py
# ============================================
# FULL INFERENCE + ENSEMBLE WRAPPER (RF + XGB)
# Uses: Wav2Vec2 + RF + XGB + alpha + threshold
# Exposes:
#   - ensemble_predict_file(path, ...)
#   - ensemble_predict_from_bytes(file_bytes, ...)
# ============================================

import os, io
import numpy as np
import torch
import librosa
import soundfile as sf
from collections import Counter
from scipy.special import expit
from joblib import load
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ---------------- CONFIG ----------------
TARGET_SR = 16000
SEG_S = 15.0
OVERLAP_S = 0.0
TRIM_SILENCE = True
TOP_DB = 20
MIN_CHUNK_SEC = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---------------- Load Wav2Vec2 ----------------
W2V = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(W2V)
w2v_model = Wav2Vec2Model.from_pretrained(W2V).to(DEVICE)
w2v_model.eval()
print("Loaded Wav2Vec2")

# ---------------- Helpers ----------------
def load_and_preprocess(path, target_sr=TARGET_SR, trim_silence=TRIM_SILENCE, top_db=TOP_DB):
    """Load audio from file path, convert to mono, resample, normalize, trim silence."""
    y, sr = sf.read(path, dtype='float32')
    if y.ndim > 1:
        y = librosa.to_mono(y.T)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=top_db)
    return y.astype(np.float32), sr

def load_and_preprocess_from_bytes(file_bytes, target_sr=TARGET_SR, trim_silence=TRIM_SILENCE, top_db=TOP_DB):
    """Same as above, but from in-memory bytes (for FastAPI uploads)."""
    y, sr = sf.read(io.BytesIO(file_bytes), dtype='float32')
    if y.ndim > 1:
        y = librosa.to_mono(y.T)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=top_db)
    return y.astype(np.float32), sr

def chunk_audio(y, sr, seg_s=SEG_S, overlap_s=OVERLAP_S):
    if seg_s <= 0:
        return [y]
    seg_len = int(seg_s * sr)
    step = seg_len - int(overlap_s * sr)
    step = max(step, 1)
    chunks = []
    for start in range(0, max(1, len(y) - seg_len + 1), step):
        end = start + seg_len
        chunks.append(y[start:end])
    remainder = y[len(chunks)*step:]
    if len(remainder) > int(1 * sr):
        chunks.append(remainder)
    if len(chunks) == 0:
        chunks.append(y)
    return chunks

def extract_wav2vec_embedding_from_wave(y, sr=TARGET_SR):
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = w2v_model(**inputs)
        hidden = outputs.last_hidden_state
        emb = torch.mean(hidden, dim=1).squeeze().cpu().numpy()
    return emb

# ---------------- Load models & ensemble meta ----------------
# Get the project root directory (parent of 'backend' folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, "models")

rf_path = os.path.join(models_dir, "rf_wav2vec_model222.joblib")
xgb_path = os.path.join(models_dir, "xgb_wav2vec_model222.joblib")
alpha_path = os.path.join(models_dir, "ensemble_alpha222.npy")
ens_thresh_path = os.path.join(models_dir, "ensemble_threshold222.npy")

rf_model = None
xgb_model = None
alpha = None
ens_thresh = None

if os.path.exists(rf_path) and os.path.exists(xgb_path):
    rf_model = load(rf_path)
    xgb_model = load(xgb_path)
    print("Loaded RF and XGB models.")
else:
    print("Ensemble model files not found - will try single-model mode if available.")

if os.path.exists(alpha_path):
    a_arr = np.load(alpha_path, allow_pickle=True)
    try:
        alpha = float(a_arr.item())
    except Exception:
        alpha = float(a_arr[0])
    print("Loaded ensemble alpha:", alpha)
else:
    print("ensemble_alpha.npy not found. Default alpha=0.5 will be used if ensemble invoked.")
    alpha = 0.5

if os.path.exists(ens_thresh_path):
    t_arr = np.load(ens_thresh_path, allow_pickle=True)
    try:
        ens_thresh = float(t_arr.item())
    except Exception:
        ens_thresh = float(t_arr[0])
    print("Loaded ensemble threshold:", ens_thresh)
else:
    print("ensemble_threshold.npy not found. Will compute/require threshold externally or use 0.5 fallback.")
    ens_thresh = None

# Optional single-model fallback
single_model_path = "xgb_wav2vec_model.joblib"
single_model = None
if os.path.exists(single_model_path):
    single_model = load(single_model_path)
    print("Loaded single-model fallback:", single_model_path)

# For deployment, you liked 0.45 as threshold:
DEFAULT_DEPLOY_THRESH = 0.438

# ---------------- Prediction helpers ----------------
def chunk_level_probs_for_wave(y, sr, model, scaler=None, seg_s=SEG_S, overlap_s=OVERLAP_S, verbose=False):
    chunks = chunk_audio(y, sr, seg_s=seg_s, overlap_s=overlap_s)
    probs = []
    preds = []
    diagnostics = {"n_chunks": 0, "skipped_short": 0}
    for ch in chunks:
        if len(ch) < MIN_CHUNK_SEC * sr:
            diagnostics["skipped_short"] += 1
            continue
        diagnostics["n_chunks"] += 1
        emb = extract_wav2vec_embedding_from_wave(ch, sr=sr)
        x = emb.reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)
        if hasattr(model, "predict_proba"):
            p_all = model.predict_proba(x)[0]
            pred = int(model.predict(x)[0])
        else:
            dec = model.decision_function(x)
            if dec.ndim == 1:
                prob_pos = float(expit(dec[0]))
                p_all = np.array([1 - prob_pos, prob_pos])
            else:
                exps = np.exp(dec - dec.max())
                p_all = (exps / exps.sum(axis=1, keepdims=True))[0]
            pred = int(np.argmax(p_all))
        probs.append(p_all)
        preds.append(int(pred))
    if len(probs) == 0:
        return None, None, diagnostics
    probs = np.vstack(probs)
    return probs, preds, diagnostics

def chunk_level_probs_for_file(path, model, scaler=None, seg_s=SEG_S, overlap_s=OVERLAP_S, verbose=False):
    y, sr = load_and_preprocess(path)
    return chunk_level_probs_for_wave(y, sr, model, scaler=scaler, seg_s=seg_s, overlap_s=overlap_s, verbose=verbose)

def pos_index_from_model(m):
    cls = getattr(m, "classes_", None)
    if cls is None:
        return -1
    try:
        return int(list(cls).index(1))
    except ValueError:
        return int(len(cls) - 1)

def ensemble_predict_file(path,
                          rf_model=rf_model,
                          xgb_model=xgb_model,
                          alpha=alpha,
                          scaler=None,
                          seg_s=SEG_S,
                          overlap_s=OVERLAP_S,
                          agg_method="mean_proba",
                          ensemble_threshold=None,
                          verbose=False):
    """
    Returns: final_label_str, pos_prob(float), diagnostics dict
    """
    if ensemble_threshold is None:
        ensemble_threshold = DEFAULT_DEPLOY_THRESH

    probs_xgb, preds_xgb, diag_x = chunk_level_probs_for_file(path, xgb_model, scaler=scaler, seg_s=seg_s, overlap_s=overlap_s, verbose=verbose)
    probs_rf, preds_rf, diag_rf = chunk_level_probs_for_file(path, rf_model, scaler=scaler, seg_s=seg_s, overlap_s=overlap_s, verbose=verbose)

    if probs_xgb is None or probs_rf is None:
        diagnostics = {"n_chunks": 0, "skipped_short": max(diag_x.get("skipped_short",0), diag_rf.get("skipped_short",0))}
        return None, None, diagnostics

    n_chunks = min(probs_xgb.shape[0], probs_rf.shape[0])
    if probs_xgb.shape[0] != probs_rf.shape[0]:
        probs_xgb = probs_xgb[:n_chunks]
        probs_rf  = probs_rf[:n_chunks]

    pos_idx_xgb = pos_index_from_model(xgb_model)
    pos_idx_rf  = pos_index_from_model(rf_model)

    p_xgb_pos = probs_xgb[:, pos_idx_xgb]
    p_rf_pos  = probs_rf[:, pos_idx_rf]
    p_ens_pos = alpha * p_xgb_pos + (1.0 - alpha) * p_rf_pos

    probs_ens = np.vstack([1.0 - p_ens_pos, p_ens_pos]).T  # (n_chunks, 2)

    if agg_method == "mean_proba":
        mean_probs = probs_ens.mean(axis=0)
        pos_prob = float(mean_probs[1])
    elif agg_method == "max":
        mean_probs = np.array([1.0 - p_ens_pos.max(), p_ens_pos.max()])
        pos_prob = float(mean_probs[1])
    elif agg_method == "topk_mean":
        k = min(2, len(p_ens_pos))
        pos_prob = float(np.sort(p_ens_pos)[-k:].mean())
        mean_probs = np.array([1.0 - pos_prob, pos_prob])
    else:
        chunk_preds = (p_ens_pos >= 0.5).astype(int)
        most = Counter(chunk_preds).most_common(1)[0][0]
        pos_prob = float((p_ens_pos >= 0.5).mean())
        mean_probs = np.array([1-pos_prob, pos_prob])

    thresh = ensemble_threshold
    final_label = "Depressed" if pos_prob >= thresh else "Not Depressed"

    diagnostics = {
        "mean_probs": mean_probs.tolist(),
        "chunk_pos_probs": p_ens_pos.tolist(),
        "n_chunks": int(n_chunks),
        "skipped_short": int(max(diag_x.get("skipped_short",0), diag_rf.get("skipped_short",0))),
    }
    return final_label, pos_prob, diagnostics

def ensemble_predict_from_bytes(file_bytes,
                                rf_model=rf_model,
                                xgb_model=xgb_model,
                                alpha=alpha,
                                threshold=None,
                                agg_method="mean_proba"):
    """
    API-friendly wrapper:
    Input: raw WAV bytes
    Output: JSON-serializable dict
    """
    if threshold is None:
        threshold = DEFAULT_DEPLOY_THRESH

    y, sr = load_and_preprocess_from_bytes(file_bytes)
    probs_xgb, preds_xgb, diag_x = chunk_level_probs_for_wave(y, sr, xgb_model, scaler=None, seg_s=SEG_S, overlap_s=OVERLAP_S)
    probs_rf, preds_rf, diag_rf = chunk_level_probs_for_wave(y, sr, rf_model, scaler=None, seg_s=SEG_S, overlap_s=OVERLAP_S)

    if probs_xgb is None or probs_rf is None:
        return {
            "ok": False,
            "reason": "no_valid_chunks",
            "detail": "All chunks too short or invalid.",
            "n_chunks": 0,
            "skipped_short": max(diag_x.get("skipped_short",0), diag_rf.get("skipped_short",0))
        }

    n_chunks = min(probs_xgb.shape[0], probs_rf.shape[0])
    if probs_xgb.shape[0] != probs_rf.shape[0]:
        probs_xgb = probs_xgb[:n_chunks]
        probs_rf  = probs_rf[:n_chunks]

    pos_idx_xgb = pos_index_from_model(xgb_model)
    pos_idx_rf  = pos_index_from_model(rf_model)

    p_xgb_pos = probs_xgb[:, pos_idx_xgb]
    p_rf_pos  = probs_rf[:, pos_idx_rf]
    p_ens_pos = alpha * p_xgb_pos + (1.0 - alpha) * p_rf_pos

    probs_ens = np.vstack([1.0 - p_ens_pos, p_ens_pos]).T
    mean_probs = probs_ens.mean(axis=0)
    pos_prob = float(mean_probs[1])

    label = "Depressed" if pos_prob >= threshold else "Not Depressed"

    return {
        "ok": True,
        "label": label,
        "pos_prob": pos_prob,
        "threshold": threshold,
        "n_chunks": int(n_chunks),
        "skipped_short": int(max(diag_x.get("skipped_short",0), diag_rf.get("skipped_short",0))),
        "chunk_pos_probs": p_ens_pos.tolist()
    }
