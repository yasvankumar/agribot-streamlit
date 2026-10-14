"""
AgriBot — Unified Web Interface (Algorithm 5: Frontend Orchestration)

Session State: uploaded image, prediction result, selected language (Tamil/Eng)
Event Handling:
  • On Upload  → trigger Module 1 (Visual Intelligence) automatically
  • On Success → trigger Module 3 (Real-Time Context) in background
Dynamic Rendering:
  • Image + Disease Name
  • Advisory Cards (treatment info)
  • Live Widgets (weather icons & price tags)
Translation Layer:
  • If Language == 'Tamil', map ALL static text through translation dictionary
"""

import streamlit as st
import os, json
from PIL import Image
import pandas as pd
from datetime import datetime

from utils.lm_client         import LMStudioClient
from utils.rag_pipeline      import RAGPipeline
from utils.disease_detection import DiseaseDetector
from utils.weather_service   import WeatherService
from utils.market_service    import MarketService
from utils.language_utils    import LanguageUtils

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 5 — TRANSLATION LAYER
# If Language == 'Tamil', map ALL static UI text through this dictionary.
# ═══════════════════════════════════════════════════════════════════════════════
UI_STRINGS = {
    "en": {
        "app_title":          "AgriBot – Agricultural AI Assistant",
        "subtitle":           "Bilingual AI for Tamil Nadu Farmers",
        "tab_chat":           "💬 Chat",
        "tab_disease":        "🔬 Disease Detection",
        "tab_hub":            "📊 Advisory Hub",
        "settings":           "⚙️ Settings",
        "language_label":     "Language",
        "location_label":     "Your district / city",
        "location_ph":        "Chennai, Tamil Nadu",
        "ow_key_label":       "OpenWeatherMap API Key",
        "reload_models":      "🔄 Reload Models",
        "yolo_label":         "YOLO model path (best.pt)",
        "iso_label":          "Isolation Forest path (.pkl)",
        "weather_btn":        "🌤️ Weather Forecast",
        "market_btn":         "💰 Market Prices",
        "tip_btn":            "💡 Random Farming Tip",
        "clear_btn":          "🗑️ Clear Chat",
        "chat_title":         "Chat with AgriBot",
        "chat_caption":       "Ask in English, Tamil, or Tanglish – RAG knowledge base + live weather/market context.",
        "chat_placeholder":   "Type your farming question…",
        "thinking":           "Thinking…",
        "disease_title":      "Crop Disease Detection",
        "disease_caption":    "Upload a clear, close-up photo of a single crop leaf.",
        "upload_label":       "Choose image",
        "analyse_btn":        "🔍 Analyse",
        "step1_label":        "Step 1 – Gatekeeper: Isolation Forest",
        "step1_ok":           "✅ ACCEPT — Valid leaf image",
        "step1_reject":       "❌ REJECT — Invalid / out-of-distribution image",
        "step2_yolo":         "Step 2 – YOLOv11 Disease Classification",
        "step2_anomaly":      "Step 2 – Qwen3-VL Underclass Visual Assessment",
        "step3_label":        "Step 3 – Qwen3-VL Deep Vision Analysis",
        "gatekeeper_result":  "Gatekeeper Result",
        "yolo_result":        "YOLOv11 Classification Result",
        "ai_analysis":        "AI Visual Analysis (Qwen3-VL + RAG)",
        "underclass_notice":  "⚠️ UNDERCLASS — Image outside 38 trained classes. Probabilistic assessment only.",
        "condition":          "Condition",
        "confidence":         "Confidence",
        "severity":           "Severity",
        "treatment":          "Treatment",
        "prevention":         "Prevention",
        "market_price":       "Market Price",
        "rag_extracts":       "📚 RAG Knowledge Extracts (top-3 chunks)",
        "known_classes":      "38 PlantVillage Classes supported by YOLOv11",
        "hub_title":          "Advisory Hub",
        "weather_expander":   "🌤️ Live Weather + Agricultural Alerts",
        "market_expander":    "💰 Live Market Prices",
        "knowledge_expander": "📚 Agricultural Knowledge Base (RAG)",
        "search_expander":    "🔍 Search Knowledge Base",
        "disease_lookup":     "🦠 Disease Database Lookup",
        "no_location":        "Enter your location in the sidebar to see live weather.",
        "get_knowledge":      "Get Knowledge",
        "search_ph":          "e.g., tomato blight, rice blast",
        "search_btn":         "Search",
        "topic_label":        "Topic",
        "lm_offline_warn":    "LM Studio is not running. Open LM Studio → load Qwen3-VL → Start Server (port 1234).",
        "crop_label":         "Crop",
        "anomaly_score":      "Anomaly Score (raw IF decision_function)",
        "threshold":          "Threshold",
        "decision":           "Decision",
        "full_analysis":      "Full Bilingual Analysis",
    },
    "ta": {
        "app_title":          "AgriBot – விவசாய AI உதவியாளர்",
        "subtitle":           "தமிழ்நாடு விவசாயிகளுக்கான இரு மொழி AI",
        "tab_chat":           "💬 அரட்டை",
        "tab_disease":        "🔬 நோய் கண்டறிதல்",
        "tab_hub":            "📊 ஆலோசனை மையம்",
        "settings":           "⚙️ அமைப்புகள்",
        "language_label":     "மொழி",
        "location_label":     "உங்கள் மாவட்டம் / நகரம்",
        "location_ph":        "சென்னை, தமிழ்நாடு",
        "ow_key_label":       "OpenWeatherMap API திறவுகோல்",
        "reload_models":      "🔄 மாடல்களை மீட்டேற்று",
        "yolo_label":         "YOLO மாடல் பாதை (best.pt)",
        "iso_label":          "Isolation Forest பாதை (.pkl)",
        "weather_btn":        "🌤️ வானிலை முன்னறிவிப்பு",
        "market_btn":         "💰 சந்தை விலைகள்",
        "tip_btn":            "💡 விவசாய குறிப்பு",
        "clear_btn":          "🗑️ அரட்டை அழி",
        "chat_title":         "AgriBot உடன் பேசுங்கள்",
        "chat_caption":       "ஆங்கிலம், தமிழ் அல்லது Tanglish-ல் கேளுங்கள் – RAG அறிவுத்தளம் + நேரடி வானிலை/சந்தை தகவல்.",
        "chat_placeholder":   "உங்கள் விவசாய கேள்வியை தட்டச்சு செய்யுங்கள்…",
        "thinking":           "யோசித்துக்கொண்டிருக்கிறேன்…",
        "disease_title":      "பயிர் நோய் கண்டறிதல்",
        "disease_caption":    "ஒரு பயிர் இலையின் தெளிவான, நெருக்கமான புகைப்படத்தை பதிவேற்றவும்.",
        "upload_label":       "படத்தை தேர்வு செய்யுங்கள்",
        "analyse_btn":        "🔍 பகுப்பாய்வு செய்",
        "step1_label":        "படி 1 – வாயில்காப்பவர்: Isolation Forest",
        "step1_ok":           "✅ ஏற்கப்பட்டது — சரியான இலை படம்",
        "step1_reject":       "❌ நிராகரிக்கப்பட்டது — பயிற்சி வகுப்புகளுக்கு வெளியே",
        "step2_yolo":         "படி 2 – YOLOv11 நோய் வகைப்பாடு",
        "step2_anomaly":      "படி 2 – Qwen3-VL காட்சி மதிப்பீடு (வகுப்பு இல்லாத நிலை)",
        "step3_label":        "படி 3 – Qwen3-VL ஆழமான காட்சி பகுப்பாய்வு",
        "gatekeeper_result":  "வாயில்காப்பவர் முடிவு",
        "yolo_result":        "YOLOv11 வகைப்பாடு முடிவு",
        "ai_analysis":        "AI காட்சி பகுப்பாய்வு (Qwen3-VL + RAG)",
        "underclass_notice":  "⚠️ பயிற்சி வகுப்புகளுக்கு வெளியே — சாத்தியமான மதிப்பீடு மட்டுமே.",
        "condition":          "நிலை",
        "confidence":         "நம்பிக்கை",
        "severity":           "தீவிரம்",
        "treatment":          "சிகிச்சை",
        "prevention":         "தடுப்பு",
        "market_price":       "சந்தை விலை",
        "rag_extracts":       "📚 RAG அறிவு பகுதிகள் (தலைமை 3)",
        "known_classes":      "YOLOv11 ஆதரிக்கும் 38 PlantVillage வகுப்புகள்",
        "hub_title":          "ஆலோசனை மையம்",
        "weather_expander":   "🌤️ நேரடி வானிலை + விவசாய எச்சரிக்கைகள்",
        "market_expander":    "💰 நேரடி சந்தை விலைகள்",
        "knowledge_expander": "📚 விவசாய அறிவுத்தளம் (RAG)",
        "search_expander":    "🔍 அறிவுத்தளத்தில் தேடுங்கள்",
        "disease_lookup":     "🦠 நோய் தரவுத்தள தேடல்",
        "no_location":        "நேரடி வானிலையைக் காண பக்கப்பட்டியில் உங்கள் இடத்தை உள்ளிடவும்.",
        "get_knowledge":      "தகவல் பெறுங்கள்",
        "search_ph":          "எ.கா., தக்காளி blight, நெல் blast",
        "search_btn":         "தேடு",
        "topic_label":        "தலைப்பு",
        "lm_offline_warn":    "LM Studio இயங்கவில்லை. LM Studio திறந்து Qwen3-VL load செய்து Server தொடங்கவும்.",
        "crop_label":         "பயிர்",
        "anomaly_score":      "அசாதாரண மதிப்பெண் (IF decision_function)",
        "threshold":          "வாசல் மதிப்பு",
        "decision":           "முடிவு",
        "full_analysis":      "முழுமையான இரு மொழி பகுப்பாய்வு",
    },
    "tanglish": {  # Tanglish: English keys, mixed labels
        "app_title":          "AgriBot – Agricultural AI Assistant",
        "subtitle":           "Tamil Nadu Farmers-க்கான Bilingual AI",
        "tab_chat":           "💬 Chat | அரட்டை",
        "tab_disease":        "🔬 Disease Detection | நோய் கண்டறிதல்",
        "tab_hub":            "📊 Advisory Hub | ஆலோசனை மையம்",
        "settings":           "⚙️ Settings | அமைப்புகள்",
        "language_label":     "Language | மொழி",
        "location_label":     "Location | இடம்",
        "location_ph":        "Chennai, Tamil Nadu",
        "ow_key_label":       "OpenWeatherMap API Key",
        "reload_models":      "🔄 Reload Models",
        "yolo_label":         "YOLO model path",
        "iso_label":          "Isolation Forest path",
        "weather_btn":        "🌤️ Weather | வானிலை",
        "market_btn":         "💰 Market | சந்தை விலை",
        "tip_btn":            "💡 Farming Tip | குறிப்பு",
        "clear_btn":          "🗑️ Clear Chat",
        "chat_title":         "AgriBot-உடன் Chat பண்ணுங்கள்",
        "chat_caption":       "English, Tamil, or Tanglish-ல் கேளுங்கள் – RAG + Live weather/market context.",
        "chat_placeholder":   "Farming question கேளுங்கள்…",
        "thinking":           "யோசிக்கிறேன்… Thinking…",
        "disease_title":      "Crop Disease Detection | பயிர் நோய் கண்டறிதல்",
        "disease_caption":    "ஒரு crop leaf-ஐ upload பண்ணுங்கள்.",
        "upload_label":       "Image choose பண்ணுங்கள்",
        "analyse_btn":        "🔍 Analyse பண்ணு",
        "step1_label":        "Step 1 – Gatekeeper: Isolation Forest",
        "step1_ok":           "✅ ACCEPT – Valid leaf image",
        "step1_reject":       "❌ REJECT – Training classes-க்கு வெளியே",
        "step2_yolo":         "Step 2 – YOLOv11 Disease Classification",
        "step2_anomaly":      "Step 2 – Qwen3-VL Underclass Visual Assessment",
        "step3_label":        "Step 3 – Qwen3-VL Deep Vision + RAG Analysis",
        "gatekeeper_result":  "Gatekeeper Result",
        "yolo_result":        "YOLOv11 Result",
        "ai_analysis":        "AI Vision Analysis (Qwen3-VL + RAG)",
        "underclass_notice":  "⚠️ UNDERCLASS – 38 classes-க்கு வெளியே. Probabilistic assessment மட்டுமே.",
        "condition":          "Condition | நிலை",
        "confidence":         "Confidence | நம்பிக்கை",
        "severity":           "Severity | தீவிரம்",
        "treatment":          "Treatment | சிகிச்சை",
        "prevention":         "Prevention | தடுப்பு",
        "market_price":       "Market Price | சந்தை விலை",
        "rag_extracts":       "📚 RAG Knowledge Extracts (top-3)",
        "known_classes":      "38 PlantVillage Classes",
        "hub_title":          "Advisory Hub | ஆலோசனை மையம்",
        "weather_expander":   "🌤️ Live Weather + Agricultural Alerts",
        "market_expander":    "💰 Live Market Prices | சந்தை விலைகள்",
        "knowledge_expander": "📚 Agricultural Knowledge Base (RAG)",
        "search_expander":    "🔍 Search Knowledge Base",
        "disease_lookup":     "🦠 Disease Database Lookup",
        "no_location":        "Sidebar-ல் location enter பண்ணுங்கள்.",
        "get_knowledge":      "Knowledge பெறுங்கள்",
        "search_ph":          "e.g., tomato blight, நெல் blast",
        "search_btn":         "Search பண்ணு",
        "topic_label":        "Topic | தலைப்பு",
        "lm_offline_warn":    "LM Studio இயங்கவில்லை. Open பண்ணி model load செய்யுங்கள்.",
        "crop_label":         "Crop | பயிர்",
        "anomaly_score":      "Anomaly Score (IF decision_function)",
        "threshold":          "Threshold | வாசல்",
        "decision":           "Decision | முடிவு",
        "full_analysis":      "Full Bilingual Analysis | முழு பகுப்பாய்வு",
    },
}


def T(key: str) -> str:
    """Translation Layer — Algorithm 5: maps static text to active language."""
    lang = st.session_state.get("language", "en")
    d    = UI_STRINGS.get(lang, UI_STRINGS["en"])
    return d.get(key, UI_STRINGS["en"].get(key, key))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriBot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;500;600&display=swap');

html,body,[class*="css"] { font-family:'Source Sans 3',sans-serif; }
h1,h2,h3                 { font-family:'Playfair Display',serif; }

[data-testid="stSidebar"] {
    background:linear-gradient(170deg,#0d2016 0%,#163322 60%,#0a1a0f 100%);
    border-right:2px solid #2a5c3a;
}
[data-testid="stSidebar"] *   { color:#b4dbc4 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3  { color:#e8f5e9 !important; }
[data-testid="stSidebar"] .stButton button {
    background:linear-gradient(135deg,#2d6a4f,#1b4332);
    border:none; color:#fff !important; border-radius:10px;
    width:100%; font-weight:600; padding:10px 14px;
    transition:all .18s; margin:3px 0;
    box-shadow:0 2px 8px rgba(0,0,0,.28);
}
[data-testid="stSidebar"] .stButton button:hover {
    background:linear-gradient(135deg,#40916c,#2d6a4f);
    transform:translateY(-1px);
}

.hero {
    background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 45%,#33691e 100%);
    border-radius:18px; padding:26px 36px; margin-bottom:22px;
    box-shadow:0 10px 36px rgba(27,94,32,.38); border:1px solid #4caf50;
}
.hero h1 { color:#f1f8e9 !important; font-size:2.5rem; margin:0;
            text-shadow:0 3px 12px rgba(0,0,0,.32); }
.hero p  { color:#a5d6a7; margin:4px 0 0; font-size:1.05rem; }
.pill {
    display:inline-flex; align-items:center; gap:5px;
    padding:4px 13px; border-radius:20px;
    font-size:.77rem; font-weight:700;
}
.pill-ok   { background:#e8f5e9; color:#1b5e20; border:1.5px solid #81c784; }
.pill-warn { background:#fff8e1; color:#e65100; border:1.5px solid #ffcc80; }
.pill-err  { background:#fce4ec; color:#b71c1c; border:1.5px solid #f48fb1; }

/* ── Advisory Cards (Algorithm 5 spec) ── */
.advisory-card {
    background:#f9fdf3; border-radius:14px; padding:20px 24px;
    border-left:5px solid #2e7d32; box-shadow:0 3px 14px rgba(46,125,50,.12);
    margin:10px 0;
}
.advisory-card h4 { color:#1b5e20; margin-top:0; font-family:'Playfair Display',serif; }
.advisory-card-warn {
    background:#fff8e1; border-left:5px solid #f57f17;
    border-radius:14px; padding:20px 24px; margin:10px 0;
    box-shadow:0 3px 14px rgba(245,127,23,.1);
}
.advisory-card-warn h4 { color:#bf360c; margin-top:0; }
.advisory-card-reject {
    background:#fce4ec; border-left:5px solid #c62828;
    border-radius:14px; padding:20px 24px; margin:10px 0;
    box-shadow:0 3px 14px rgba(198,40,40,.1);
}
.advisory-card-reject h4 { color:#b71c1c; margin-top:0; }

/* ── Live Widgets (Algorithm 5 spec) ── */
.live-widget {
    background:linear-gradient(135deg,#e8f5e9,#f1f8e9);
    border-radius:12px; padding:14px 18px; border:1px solid #a5d6a7;
    display:inline-block; margin:5px; text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,.07);
}
.live-widget .value { font-size:1.5rem; font-weight:700; color:#1b5e20; }
.live-widget .label { font-size:.75rem; color:#4caf50; }
.price-widget {
    background:linear-gradient(135deg,#fff8e1,#fffde7);
    border-radius:12px; padding:14px 18px; border:1px solid #ffe082;
    margin:8px 0; box-shadow:0 2px 8px rgba(0,0,0,.07);
}
.price-widget .crop-name { font-weight:700; color:#e65100; font-size:1.05rem; }
.price-widget .price     { font-size:1.3rem; font-weight:700; color:#1b5e20; }

/* ── Alert banners ── */
.alert-danger  { background:#fff3e0; border-left:5px solid #e53935;
                 border-radius:10px; padding:14px 18px; margin:6px 0; }
.alert-warning { background:#fffde7; border-left:5px solid #f9a825;
                 border-radius:10px; padding:14px 18px; margin:6px 0; }
.alert-info    { background:#e3f2fd; border-left:5px solid #1565c0;
                 border-radius:10px; padding:14px 18px; margin:6px 0; }
.alert-success { background:#f1f8e9; border-left:5px solid #2e7d32;
                 border-radius:10px; padding:14px 18px; margin:6px 0; }

[data-testid="stMetric"] {
    background:#f1f8e9; border-radius:12px; padding:14px;
    border:1px solid #a5d6a7; box-shadow:0 2px 6px rgba(0,0,0,.06);
}
[data-baseweb="tab"] { font-weight:600; font-size:.95rem; }
[aria-selected="true"] { color:#2e7d32 !important; border-bottom:3px solid #2e7d32 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE  (Algorithm 5: store image, prediction, language)
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in [("messages",[]), ("language","en"), ("location",""),
              ("openweather_key",""), ("last_analysis",None),
              ("uploaded_image",None), ("detection_done", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE INIT
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_services():
    return dict(
        lm      = LMStudioClient(),
        rag     = RAGPipeline(),
        disease = DiseaseDetector(),
        weather = WeatherService(),
        market  = MarketService(),
        lang    = LanguageUtils(),
    )

svc = init_services()

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
lm_ok   = svc["lm"].is_server_running()
yolo_ok = svc["disease"].models_loaded["yolo"]
iso_ok  = svc["disease"].models_loaded["isolation_forest"]

st.markdown(f"""
<div class="hero">
  <h1>🌿 {T('app_title')}</h1>
  <p>{T('subtitle')} &nbsp;·&nbsp; YOLOv11 + Isolation Forest + RAG + Qwen3-VL</p>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;">
    <span class="pill {'pill-ok' if lm_ok   else 'pill-err' }">{'✅' if lm_ok   else '❌'} LM Studio {'Online' if lm_ok else 'Offline'}</span>
    <span class="pill {'pill-ok' if yolo_ok else 'pill-warn'}">{'✅' if yolo_ok else '⚠️'} YOLOv11 {'Ready' if yolo_ok else 'Not Loaded'}</span>
    <span class="pill {'pill-ok' if iso_ok  else 'pill-warn'}">{'✅' if iso_ok  else '⚠️'} Isolation Forest {'Ready' if iso_ok else 'Not Loaded'}</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not lm_ok:
    st.warning(T("lm_offline_warn"), icon="🤖")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## {T('settings')}")

    lang_options = {"English 🇬🇧":"en","தமிழ் 🇮🇳":"ta","Tanglish (Mixed)":"tanglish"}
    prev_lang = st.session_state.language
    sel = st.selectbox(T("language_label"), list(lang_options))
    st.session_state.language = lang_options[sel]
    if lang_options[sel] != prev_lang:
        st.rerun()  # re-render entire UI in new language

    st.markdown("---")
    location = st.text_input(T("location_label"), placeholder=T("location_ph"),
                             value=st.session_state.location)
    st.session_state.location = location

    st.markdown("---")
    ow_key = st.text_input(T("ow_key_label"), type="password",
                           value=st.session_state.openweather_key or os.getenv("OPENWEATHER_API_KEY",""))
    if ow_key:
        os.environ["OPENWEATHER_API_KEY"] = ow_key
        svc["weather"].api_key = ow_key
        st.session_state.openweather_key = ow_key

    st.markdown("---")
    st.markdown(f"### {'🤖 Model Paths' if st.session_state.language=='en' else '🤖 மாடல் பாதைகள்'}")
    yolo_path = st.text_input(T("yolo_label"), value=svc["disease"].yolo_path)
    iso_path  = st.text_input(T("iso_label"),  value=svc["disease"].iso_path)
    if st.button(T("reload_models")):
        svc["disease"].yolo_path  = yolo_path
        svc["disease"].iso_path   = iso_path
        svc["disease"].yolo_model = None
        svc["disease"].iso_model  = None
        svc["disease"].load_errors= []
        svc["disease"]._load_yolo()
        svc["disease"]._load_iso()
        st.rerun()
    for e in svc["disease"].load_errors:
        st.error(e)

    st.markdown("---")
    if st.button(T("weather_btn")):
        if location:
            raw = svc["weather"].get_weather_forecast(location)
            st.session_state.messages.append({"role":"assistant","content":raw,"type":"weather"})
            st.rerun()
        else:
            st.warning(T("no_location"))

    if st.button(T("market_btn")):
        mp = svc["market"].get_market_prices()
        st.session_state.messages.append({"role":"assistant","content":mp,"type":"market"})
        st.rerun()

    if st.button(T("tip_btn")):
        tip = svc["rag"].get_random_agricultural_tip()
        st.session_state.messages.append({"role":"assistant","content":tip})
        st.rerun()

    if st.button(T("clear_btn")):
        st.session_state.messages = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
t_chat, t_disease, t_hub = st.tabs([T("tab_chat"), T("tab_disease"), T("tab_hub")])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – CHAT  (Algorithm 3: RAG Chatbot)
# ─────────────────────────────────────────────────────────────────────────────
with t_chat:
    st.markdown(f"### {T('chat_title')}")
    st.caption(T("chat_caption"))

    def _render_weather_widget(raw: str):
        """Live Widget for weather (Algorithm 5)."""
        try:
            wd = json.loads(raw)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🌡️",f"{wd.get('temperature','?')}°C", f"Feels {wd.get('feels_like','?')}°C")
            c2.metric("💧",f"{wd.get('humidity','?')}%")
            c3.metric("💨",f"{wd.get('wind_speed','?')} km/h")
            c4.metric("🌧️",f"{wd.get('rainfall','?')} mm")
            st.caption(f"📍 {wd.get('location','?')} · {wd.get('weather_description','?')}"
                       + (" · ⚠️ Mock (add API key)" if wd.get("mock") else ""))
            lang = st.session_state.language
            for al in wd.get("farming_alerts",[]):
                css = {"danger":"alert-danger","warning":"alert-warning","info":"alert-info"}.get(al.get("level","info"),"alert-info")
                txt = al.get("ta" if lang=="ta" else "en", al.get("en",""))
                st.markdown(f'<div class="{css}">{al.get("icon","ℹ️")} {txt}</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown(raw)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            mtype = msg.get("type","text")
            if mtype == "weather":
                _render_weather_widget(msg["content"])
            elif mtype == "market":
                try:
                    df = pd.read_json(msg["content"])
                    if "trend" in df.columns:
                        df["trend"] = df["trend"].map(lambda t: ("📈" if t=="up" else "📉" if t=="down" else "➡️")+" "+t.title())
                    st.dataframe(df, use_container_width=True, height=300)
                except Exception:
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input(T("chat_placeholder")):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(T("thinking")):
                detected_lang = svc["lang"].detect_language(prompt)
                rag_ctx       = svc["rag"].search_knowledge_base(prompt, top_k=3)
                weather_txt   = svc["weather"].get_weather_summary_text(location) if location else ""
                market_txt    = ""
                for crop in svc["market"].data:
                    if crop.lower() in prompt.lower():
                        market_txt = svc["market"].get_crop_price_text(crop)
                        break
                response = svc["lm"].generate_bilingual_response(
                    prompt, rag_ctx, detected_lang, st.session_state.language,
                    weather_summary=weather_txt, market_summary=market_txt,
                )
            st.markdown(response)
            st.session_state.messages.append({"role":"assistant","content":response})

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – DISEASE DETECTION  (Algorithms 1 + 2 + vision LLM)
# ─────────────────────────────────────────────────────────────────────────────
with t_disease:
    st.markdown(f"### {T('disease_title')}")
    st.markdown(T("disease_caption"))
    st.caption(
        f"**Algorithm 1 (Gatekeeper):** Isolation Forest — score < {DiseaseDetector.ANOMALY_THRESHOLD} → REJECT  \n"
        "**Algorithm 2 (Expert):** YOLOv11-Nano — CSPNet + C3k2 + PANet + Softmax → 38 classes  \n"
        "**Algorithm 3 (RAG):** Top-3 chunks → strict 'Using ONLY context' prompt → Qwen3-VL"
    )

    # Algorithm 5: on_upload → trigger Module 1 automatically
    uploaded = st.file_uploader(T("upload_label"), type=["png","jpg","jpeg","webp"],
                                key="disease_uploader")

    if uploaded:
        image = Image.open(uploaded)
        lang  = st.session_state.language

        col_img, col_main = st.columns([1, 1.5])
        with col_img:
            st.image(image, caption=T("upload_label"), use_column_width=True)

        with col_main:
            # ── ALGORITHM 5: On Upload → auto-trigger Module 1 ──────────────
            analyse = st.button(T("analyse_btn"), type="primary", use_container_width=True)

        if analyse:
            # ── STEP 1: Isolation Forest (Algorithm 1) ─────────────────────
            with st.status(f"🛡️ {T('step1_label')}…", expanded=True) as s1:
                anomaly   = svc["disease"].check_anomaly(image)
                is_reject = anomaly["is_anomaly"]
                raw_score = anomaly["raw_score"]
                s1.update(
                    label=T("step1_reject") if is_reject else T("step1_ok"),
                    state="error" if is_reject else "complete"
                )

            # Advisory Card — Gatekeeper result
            card_css  = "advisory-card-reject" if is_reject else "advisory-card"
            gate_icon = "❌" if is_reject else "✅"
            st.markdown(f"""
            <div class="{card_css}">
              <h4>{gate_icon} {T('gatekeeper_result')}</h4>
              <table style="width:100%;border-collapse:collapse;">
                <tr><td><b>{T('anomaly_score')}</b></td>
                    <td><code>{raw_score:.4f}</code></td></tr>
                <tr><td><b>{T('threshold')}</b></td>
                    <td><code>{anomaly['threshold']}</code></td></tr>
                <tr><td><b>{T('decision')}</b></td>
                    <td style="font-weight:700;color:{'#c62828' if is_reject else '#2e7d32'};">
                      {anomaly['status']}</td></tr>
              </table>
              <small>{anomaly['description']}</small>
            </div>
            """, unsafe_allow_html=True)

            # ── BRANCH: REJECT (underclass) ───────────────────────────────
            if is_reject:
                st.markdown(f"""
                <div class="advisory-card-warn">
                  <h4>{T('underclass_notice')}</h4>
                  <p>Image outside our 38 PlantVillage training classes.<br>
                  YOLOv11 skipped — passing to <b>Qwen3-VL</b> for visual assessment with honest uncertainty disclosure.</p>
                </div>
                """, unsafe_allow_html=True)

                with st.status(f"🤖 {T('step2_anomaly')}…", expanded=True) as s2:
                    if lm_ok:
                        vision_txt = svc["lm"].analyze_anomalous_image(
                            image, raw_score,
                            threshold=anomaly["threshold"], language=lang)
                        s2.update(label=f"✅ {T('step2_anomaly')} complete", state="complete")
                    else:
                        vision_txt = T("lm_offline_warn")
                        s2.update(label="❌ LM Studio offline", state="error")

                st.markdown(f"#### {T('ai_analysis')}")
                st.markdown(f"""
                <div class="advisory-card-warn">
                  <h4>⚠️ {T('underclass_notice')}</h4>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(vision_txt)

                st.session_state.messages.append({
                    "role":"assistant",
                    "content":(f"**🔬 {T('disease_title')} — Underclass/Anomaly Mode**\n\n"
                               f"IF score: {raw_score:.4f} (threshold {anomaly['threshold']}) — REJECTED\n\n"
                               f"**{T('ai_analysis')}:**\n{vision_txt}")
                })

            # ── BRANCH: ACCEPT (valid leaf) ───────────────────────────────
            else:
                # STEP 2: YOLO (Algorithm 2)
                with st.status(f"🌿 {T('step2_yolo')}…", expanded=True) as s2:
                    det = svc["disease"].detect_disease(image)
                    s2.update(label=f"✅ {T('step2_yolo')}: {det['crop']} – {det['disease']}", state="complete")

                # Advisory Card — YOLO result
                sev_col = {"high":"#c62828","medium":"#e65100","low":"#2e7d32","none":"#2e7d32"}.get(
                    det.get("severity","low"),"#555")
                sim_note = " *(SIMULATED)*" if det.get("simulated") else ""

                st.markdown(f"#### {T('yolo_result')}")
                st.markdown(f"""
                <div class="advisory-card">
                  <h4>🌱 {det['crop']}{sim_note}</h4>
                  <table style="width:100%;border-collapse:collapse;">
                    <tr><td style="width:40%"><b>{T('condition')}</b></td><td>{det['disease']}</td></tr>
                    <tr><td><b>{T('confidence')}</b></td><td>{det['confidence']:.1%}</td></tr>
                    <tr><td><b>{T('severity')}</b></td>
                        <td style="color:{sev_col};font-weight:700;">{det.get('severity','?').upper()}</td></tr>
                    <tr><td><b>{T('treatment')}</b></td><td>{det['treatment']}</td></tr>
                    <tr><td><b>{T('prevention')}</b></td><td>{det.get('prevention','')}</td></tr>
                  </table>
                </div>
                """, unsafe_allow_html=True)

                # Algorithm 5: On Success → trigger Module 3 (Real-Time Context)
                market_info = svc["market"].get_crop_price(det["crop"])
                market_txt  = svc["market"].get_crop_price_text(det["crop"])

                # Live Widget — price tag
                if market_info:
                    trend_icon = "📈" if market_info["trend"]=="up" else "📉" if market_info["trend"]=="down" else "➡️"
                    st.markdown(f"""
                    <div class="price-widget">
                      <span class="crop-name">💰 {T('market_price')} — {market_info['commodity']} ({market_info['tamil_name']})</span><br>
                      <span class="price">₹{market_info['current_price']} {market_info['unit']}</span>
                      &nbsp;{trend_icon} {market_info['change_pct']:+.1f}%
                      &nbsp;·&nbsp; 📍 {market_info['market_location']}<br>
                      <small>🇬🇧 {market_info['selling_tip_en']}</small><br>
                      <small>🇮🇳 {market_info['selling_tip_ta']}</small>
                    </div>
                    """, unsafe_allow_html=True)

                # Live Widget — weather alerts for detected crop
                weather_txt = ""
                if location:
                    weather_txt = svc["weather"].get_weather_summary_text(location)
                    raw_w = svc["weather"].get_weather_forecast(location)
                    try:
                        wd = json.loads(raw_w)
                        alerts = wd.get("farming_alerts",[])
                        if alerts:
                            for al in alerts:
                                css = {"danger":"alert-danger","warning":"alert-warning","info":"alert-info"}.get(
                                    al.get("level","info"),"alert-info")
                                txt = al.get("ta" if lang=="ta" else "en", al.get("en",""))
                                st.markdown(f'<div class="{css}">{al.get("icon","ℹ️")} {txt}</div>',
                                            unsafe_allow_html=True)
                    except Exception:
                        pass

                # RAG context for this disease
                rag_query = f"{det['crop']} {det['disease']} treatment prevention"
                rag_ctx   = svc["rag"].search_knowledge_base(rag_query, top_k=3)

                # STEP 3: Qwen3-VL deep vision + RAG
                with st.status(f"🤖 {T('step3_label')}…", expanded=True) as s3:
                    if lm_ok:
                        ai_analysis = svc["lm"].analyze_valid_leaf(
                            image, det, rag_ctx,
                            weather_summary=weather_txt,
                            market_info=market_txt,
                            language=lang,
                        )
                        s3.update(label=f"✅ {T('step3_label')} complete", state="complete")
                    else:
                        ai_analysis = T("lm_offline_warn")
                        s3.update(label="❌ LM Studio offline", state="error")

                st.markdown(f"#### {T('ai_analysis')}")
                with st.expander(T("full_analysis"), expanded=True):
                    st.markdown(ai_analysis)

                # RAG extracts reference
                with st.expander(T("rag_extracts"), expanded=False):
                    st.markdown(rag_ctx)

                st.session_state.messages.append({
                    "role":"assistant",
                    "content":(f"**🔬 {T('disease_title')}**\n\n"
                               f"• {T('crop_label')}: {det['crop']}\n"
                               f"• {T('condition')}: {det['disease']}\n"
                               f"• {T('confidence')}: {det['confidence']:.1%} | {T('severity')}: {det.get('severity','?')}\n"
                               f"• {T('treatment')}: {det['treatment']}\n\n"
                               f"**{T('ai_analysis')}:**\n{ai_analysis}")
                })

    # Known classes
    with st.expander(T("known_classes"), expanded=False):
        from utils.disease_detection import PLANT_VILLAGE_CLASSES
        cols = st.columns(3)
        for i,(_, (friendly,_)) in enumerate(PLANT_VILLAGE_CLASSES.items()):
            cols[i%3].write(f"• {friendly}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – ADVISORY HUB  (Algorithm 5: Live Widgets + Advisory Cards)
# ─────────────────────────────────────────────────────────────────────────────
with t_hub:
    st.markdown(f"### {T('hub_title')}")

    # Weather Live Widgets
    with st.expander(T("weather_expander"), expanded=bool(location)):
        if location:
            raw_w = svc["weather"].get_weather_forecast(location)
            try:
                wd   = json.loads(raw_w)
                lang = st.session_state.language
                # Live Widget row
                cols = st.columns(4)
                for col, (icon, key, unit) in zip(cols, [
                    ("🌡️","temperature","°C"),("💧","humidity","%"),
                    ("💨","wind_speed"," km/h"),("🌧️","rainfall"," mm")
                ]):
                    col.markdown(f"""
                    <div class="live-widget">
                      <div class="value">{icon} {wd.get(key,'?')}{unit}</div>
                      <div class="label">{key.replace('_',' ').title()}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(f"📍 {wd.get('location','?')} · {wd.get('weather_description','?')}"
                           + (" · ⚠️ Mock – add API key" if wd.get("mock") else " · ✅ Live"))

                alerts = wd.get("farming_alerts",[])
                if alerts:
                    st.markdown(f"**🔔 {'விவசாய எச்சரிக்கைகள்' if lang=='ta' else 'Agricultural Alerts'}:**")
                    for al in alerts:
                        css = {"danger":"alert-danger","warning":"alert-warning","info":"alert-info"}.get(al.get("level","info"),"alert-info")
                        txt = al.get("ta" if lang=="ta" else "en", al.get("en",""))
                        st.markdown(f'<div class="{css}">{al.get("icon","ℹ️")} {txt}</div>', unsafe_allow_html=True)

                for adv in wd.get("farming_advice",[]):
                    st.info(adv)
            except Exception as e:
                st.error(f"Weather error: {e}")
        else:
            st.info(T("no_location"))

    # Market Live Widgets (price tags)
    with st.expander(T("market_expander"), expanded=True):
        mp_raw = svc["market"].get_market_prices()
        try:
            df = pd.read_json(mp_raw)
            if "trend" in df.columns:
                df["trend"] = df["trend"].map(
                    lambda t: ("📈" if t=="up" else "📉" if t=="down" else "➡️")+" "+t.title())
            st.dataframe(df, use_container_width=True, height=380)
            for al in svc["market"].get_market_alerts():
                lang = st.session_state.language
                txt  = al.get("ta" if lang=="ta" else "en", al["en"])
                sev  = al.get("severity","low")
                css  = {"high":"alert-danger","medium":"alert-warning","low":"alert-info"}.get(sev,"alert-info")
                st.markdown(f'<div class="{css}">📊 {txt}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Market error: {e}")

    # RAG Knowledge
    with st.expander(T("knowledge_expander"), expanded=False):
        topics = ["Crop Cultivation","Fertilizer Management","Pest Control","Irrigation Techniques"]
        topic  = st.selectbox(T("topic_label"), topics)
        if st.button(T("get_knowledge")):
            st.markdown(svc["rag"].get_knowledge_by_category(topic))

    # Search
    with st.expander(T("search_expander"), expanded=False):
        q = st.text_input("", placeholder=T("search_ph"), label_visibility="collapsed")
        if q and st.button(T("search_btn")):
            with st.spinner(T("thinking")):
                st.markdown(svc["rag"].search_knowledge_base(q, top_k=3))

    # Disease lookup
    with st.expander(T("disease_lookup"), expanded=False):
        from utils.disease_detection import PLANT_VILLAGE_CLASSES
        kw = st.text_input("", placeholder="e.g., tomato, blight, rust", label_visibility="collapsed")
        if kw:
            matches = [(cr,(n,t)) for cr,(n,t) in PLANT_VILLAGE_CLASSES.items()
                       if kw.lower() in n.lower() or kw.lower() in cr.lower()]
            if matches:
                for _, (name, treatment) in matches[:5]:
                    st.markdown(f"""
                    <div class="advisory-card">
                      <h4>🌱 {name}</h4>
                      <p>{treatment}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No matches found. Try a broader term.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "🌿 **AgriBot** &nbsp;·&nbsp; "
    "IF (97.4% rejection) + YOLOv11-Nano (99.8% acc / 38 classes) + "
    "RAG + Qwen3-VL + OpenWeatherMap &nbsp;·&nbsp; "
    "AI தொழில்நுட்பத்துடன் விவசாயிகளுக்கு அதிகாரம்"
)
