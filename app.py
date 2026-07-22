"""
Emotion Classifier — Streamlit App
-----------------------------------
Loads the BiLSTM + Attention model trained in
Emotion_Classification_with_Attention.ipynb and serves an interactive
web app: type free text, get the dominant emotion + a confidence chart
across all 6 classes + an attention heatmap explaining the prediction.

Run with:  streamlit run app.py
Expects a sibling "models/" folder containing:
  bilstm_attention_explainer.keras, tokenizer.pkl, label_map.json,
  config.json, (optional) metrics.json
"""

import os
import re
import json
import pickle

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#F6F8FC",
    "surface": "#FFFFFF",
    "surface_alt": "#EEF1FB",
    "primary": "#4C5FD5",
    "primary_dark": "#3A49B0",
    "text": "#262A3E",
    "text_soft": "#666B85",
    "border": "#E4E8F5",
}

EMOTION_COLORS = {
    "joy":      "#F5A623",
    "sadness":  "#5B8DEF",
    "anger":    "#E85D53",
    "fear":     "#8B7FD1",
    "surprise": "#2FBFA5",
    "disgust":  "#6B9E5A",
}
EMOTION_EMOJI = {
    "joy": "😊", "sadness": "😢", "anger": "😠",
    "fear": "😨", "surprise": "😮", "disgust": "🤢",
}
EMOTION_BLURB = {
    "joy": "Warm, upbeat, content",
    "sadness": "Down, hurt, low energy",
    "anger": "Frustrated, offended, hostile",
    "fear": "Anxious, threatened, worried",
    "surprise": "Startled, unexpected, curious",
    "disgust": "Repulsed, distasteful, put off",
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# --------------------------------------------------------------------------
# Global CSS — light, calm, professional palette (not dark, not cliché cream)
# --------------------------------------------------------------------------
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3, h4 {{
    font-family: 'Poppins', sans-serif !important;
    color: {COLORS['text']} !important;
}}

.stApp {{
    background: linear-gradient(180deg, {COLORS['bg']} 0%, #F1F3FC 100%);
}}

section[data-testid="stSidebar"] {{
    background-color: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
}}

/* Hero header */
.hero {{
    padding: 1.75rem 2rem;
    background: linear-gradient(120deg, {COLORS['primary']} 0%, #6C7AE0 100%);
    border-radius: 20px;
    color: white;
    margin-bottom: 1.75rem;
    box-shadow: 0 10px 30px rgba(76, 95, 213, 0.25);
}}
.hero h1 {{
    color: white !important;
    margin: 0 0 0.35rem 0;
    font-size: 2rem;
}}
.hero p {{
    color: rgba(255,255,255,0.9);
    margin: 0;
    font-size: 1.02rem;
}}

/* Cards */
.card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 18px rgba(76, 95, 213, 0.06);
    margin-bottom: 1.2rem;
}}

/* Result hero card */
.result-card {{
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    color: white;
    box-shadow: 0 10px 26px rgba(0,0,0,0.12);
}}
.result-emoji {{ font-size: 3rem; line-height: 1; }}
.result-label {{ font-family:'Poppins',sans-serif; font-size: 1.7rem; font-weight: 700; margin: 0.2rem 0 0.1rem 0; }}
.result-sub {{ opacity: 0.92; font-size: 0.95rem; }}
.result-conf {{ font-family:'Poppins',sans-serif; font-size: 1.05rem; font-weight: 600; margin-top: 0.6rem; }}

/* Token chips for attention */
.chip {{
    display: inline-block;
    padding: 4px 9px;
    margin: 3px;
    border-radius: 8px;
    font-size: 0.95rem;
    color: {COLORS['text']};
}}

/* Buttons */
.stButton > button {{
    background: {COLORS['primary']};
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(76, 95, 213, 0.3);
}}
.stButton > button:hover {{
    background: {COLORS['primary_dark']};
    color: white;
}}

textarea {{
    border-radius: 12px !important;
}}

footer {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Text cleaning — MUST mirror the notebook's clean_text() exactly
# --------------------------------------------------------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
SUBREDDIT_USER_RE = re.compile(r"/?u/\w+|/?r/\w+")
MASK_TOKEN_RE = re.compile(r"\[name\]|\[religion\]", flags=re.IGNORECASE)
NON_ALPHA_RE = re.compile(r"[^a-zA-Z0-9'!?.,\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = SUBREDDIT_USER_RE.sub(" ", text)
    text = MASK_TOKEN_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="custom")
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.W = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)
            self.supports_masking = True

        def call(self, hidden_states, mask=None):
            score = self.V(tf.nn.tanh(self.W(hidden_states)))
            if mask is not None:
                mask = tf.cast(mask, score.dtype)
                mask = tf.expand_dims(mask, axis=-1)
                score += (1.0 - mask) * -1e9
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
            return context_vector, attention_weights

        def compute_mask(self, inputs, mask=None):
            return None, None

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config

    model_path = os.path.join(MODELS_DIR, "bilstm_attention_explainer.keras")
    tokenizer_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
    label_map_path = os.path.join(MODELS_DIR, "label_map.json")
    config_path = os.path.join(MODELS_DIR, "config.json")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    missing = [p for p in [model_path, tokenizer_path, label_map_path, config_path] if not os.path.exists(p)]
    if missing:
        return None

    model = tf.keras.models.load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_map_path) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    with open(config_path) as f:
        config = json.load(f)
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {"model": model, "tokenizer": tokenizer, "label_map": label_map,
            "config": config, "metrics": metrics}


def predict(text, artifacts):
    import tensorflow as tf

    tokenizer = artifacts["tokenizer"]
    max_len = artifacts["config"]["max_len"]
    model = artifacts["model"]
    label_map = artifacts["label_map"]

    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs, attn = model.predict(padded, verbose=0)
    probs = probs[0]

    tokens = clean.split()[:max_len]
    weights = attn[0, :len(tokens), 0]
    weights = weights / (weights.sum() + 1e-9) if weights.sum() > 0 else weights

    pred_idx = int(np.argmax(probs))
    pred_label = label_map[pred_idx]
    return pred_label, probs, tokens, weights


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🎭 Emotion Classifier</h1>
  <p>Type a sentence and see which of six core emotions it carries — joy, sadness, anger, fear, surprise, or disgust —
  powered by a BiLSTM + Attention model trained on Google's GoEmotions dataset.</p>
</div>
""", unsafe_allow_html=True)

artifacts = load_artifacts()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About this app")
    st.markdown(
        "This demo classifies free text into **6 core emotions** using a "
        "Bidirectional LSTM with an attention layer, trained on the "
        "[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset."
    )
    st.markdown("**Emotion palette**")
    for name, color in EMOTION_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:4px 0;">'
            f'<div style="width:14px;height:14px;border-radius:4px;background:{color};margin-right:8px;"></div>'
            f'<span>{EMOTION_EMOJI[name]} {name.capitalize()} — {EMOTION_BLURB[name]}</span></div>',
            unsafe_allow_html=True,
        )

    if artifacts and artifacts.get("metrics"):
        st.markdown("---")
        st.markdown("### Model performance (test set)")
        for name, m in artifacts["metrics"].items():
            st.metric(name, f"{m['accuracy']*100:.1f}% acc", f"F1 {m['macro_f1']:.3f}")

    st.markdown("---")
    st.caption("Built with Streamlit · TensorFlow/Keras · GoEmotions (Google Research)")

# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------
if artifacts is None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.warning(
        "**Model artifacts not found.**\n\n"
        "This app expects a `models/` folder (next to `app.py`) containing "
        "`bilstm_attention_explainer.keras`, `tokenizer.pkl`, `label_map.json`, and `config.json`.\n\n"
        "Run the last cell of **Emotion_Classification_with_Attention.ipynb** "
        "(*Section 14 — Save Artifacts for Deployment*) and copy the resulting "
        "`models/` folder next to this app, then restart Streamlit."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

col_input, col_examples = st.columns([2.3, 1])

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Enter your text")
    default_text = st.session_state.get("text_input", "")
    text = st.text_area(
        "Text",
        value=default_text,
        height=130,
        placeholder="e.g. I can't believe I got the job, I'm over the moon!",
        label_visibility="collapsed",
    )
    analyze = st.button("✨ Analyze emotion", width="content")
    st.markdown('</div>', unsafe_allow_html=True)

with col_examples:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Try an example")
    examples = [
        ("😊", "I am so happy right now, this is the best day of my life!"),
        ("😢", "I feel so lonely and heartbroken since she left."),
        ("😠", "How dare you talk to me like that, I am furious!"),
        ("😨", "I am terrified of what might happen tomorrow."),
        ("😮", "Wait, what?! I did not expect that at all."),
        ("🤢", "That is absolutely disgusting, I can't even look at it."),
    ]
    for emoji, ex in examples:
        if st.button(f"{emoji} {ex[:28]}...", key=ex, width="stretch"):
            st.session_state["text_input"] = ex
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if analyze and text.strip():
    pred_label, probs, tokens, weights = predict(text, artifacts)
    color = EMOTION_COLORS.get(pred_label, COLORS["primary"])
    emoji = EMOTION_EMOJI.get(pred_label, "")
    confidence = float(np.max(probs)) * 100

    res_col, chart_col = st.columns([1, 1.5])

    with res_col:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {color}, {color}CC);">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{pred_label.capitalize()}</div>
            <div class="result-sub">{EMOTION_BLURB.get(pred_label, "")}</div>
            <div class="result-conf">{confidence:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with chart_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        label_map = artifacts["label_map"]
        names = [label_map[i] for i in range(len(label_map))]
        order = np.argsort(probs)
        fig = go.Figure(go.Bar(
            x=[probs[i] * 100 for i in order],
            y=[names[i].capitalize() for i in order],
            orientation="h",
            marker_color=[EMOTION_COLORS.get(names[i], COLORS["primary"]) for i in order],
            text=[f"{probs[i]*100:.1f}%" for i in order],
            textposition="outside",
        ))
        fig.update_layout(
            title="Confidence across all 6 classes",
            xaxis_title="Confidence (%)",
            xaxis_range=[0, 100],
            height=290,
            margin=dict(l=10, r=30, t=40, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Which words drove this prediction?")
    st.caption("Darker highlight = higher attention weight from the BiLSTM + Attention model.")
    if len(tokens) == 0:
        st.info("No recognizable tokens to visualize.")
    else:
        max_w = max(weights.max(), 1e-9)
        spans = []
        for tok, w in zip(tokens, weights):
            alpha = 0.12 + 0.75 * (w / max_w)
            spans.append(
                f'<span class="chip" style="background:{color}{int(alpha*255):02x};">{tok}</span>'
            )
        st.markdown(" ".join(spans), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif analyze:
    st.info("Please enter some text first.")
