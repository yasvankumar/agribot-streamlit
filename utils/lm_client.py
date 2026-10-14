"""
LM Studio Client — Qwen3-VL local inference.
Implements the strict RAG prompt template from paper Algorithm 3.
"""

import os, io, base64
from PIL import Image
from openai import OpenAI

LM_STUDIO_URL   = os.getenv("LM_STUDIO_URL",   "http://127.0.0.1:1234/v1")   # ✅ /v1 added
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-4b")            # ✅ correct identifier


def _pil_to_b64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class LMStudioClient:
    def __init__(self):
        self.client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        self.model  = LM_STUDIO_MODEL

    def _chat(self, messages: list, max_tokens: int = 1024) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, temperature=0.65,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return (
                f"⚠️ **LM Studio Error:**\n\n`{e}`\n\n"
                "LM Studio-ஐ திறந்து Qwen3-VL மாடலை load செய்து server start பண்ணவும்.\n"
                "Open LM Studio → load Qwen3-VL model → click **▶ Start Server** (port 1234)."
            )

    def is_server_running(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    # ── Algorithm 3: Strict RAG chat ─────────────────────────────────────────
    def generate_bilingual_response(
        self,
        user_query: str,
        rag_context: str,           # pre-fetched top-3 chunks
        detected_language: str,
        preferred_language: str,
        weather_summary: str = "",
        market_summary:  str = "",
    ) -> str:
        """
        Implements Algorithm 3 strict prompt:
        "Using ONLY the context provided below, answer the user's question."
        + bilingual layer (Algorithm 5 Translation Layer)
        """
        lang_rule = {
            "ta":       "முக்கியமாக தமிழில் பதில் சொல்லவும். தொழில்நுட்ப வார்த்தைகளை ஆங்கிலத்திலும் கொடுக்கவும்.",
            "tanglish": "Tanglish-ல் பதில் சொல்லவும் – Tamil script மற்றும் English words இயற்கையாக கலந்து.",
            "en":       "Respond in clear English. Add Tamil translations in brackets for key farming terms.",
        }.get(preferred_language, "Respond in English.")

        # Extra live context beyond RAG (weather + market)
        live_ctx = ""
        if weather_summary:
            live_ctx += f"\n\nLive Weather Data:\n{weather_summary}"
        if market_summary:
            live_ctx += f"\n\nLive Market Price:\n{market_summary}"

        # Algorithm 3 strict prompt
        system = (
            f"You are AgriBot, a bilingual agricultural assistant for Tamil Nadu farmers.\n"
            f"Language rule: {lang_rule}\n\n"
            f"Using ONLY the context provided below, answer the user's question. "
            f"Do not use knowledge outside this context. "
            f"If the context is insufficient, say so and suggest the local Krishi Vigyan Kendra.\n\n"
            f"Agricultural Knowledge Context:\n{rag_context}"
            f"{live_ctx}\n\n"
            f"Guidelines:\n"
            f"- Practical advice for small-scale South-Indian farmers.\n"
            f"- If weather shows humidity >80% + temp >25°C → warn about HIGH FUNGAL RISK.\n"
            f"- Warm, encouraging tone like an experienced farmer-friend.\n"
            f"- Short paragraphs, no excessive bullet points."
        )

        return self._chat(
            [{"role":"system","content":system},
             {"role":"user",  "content":user_query}],
            max_tokens=900
        )

    # ── Vision: VALID leaf (ACCEPT path) ─────────────────────────────────────
    def analyze_valid_leaf(
        self,
        pil_image:       Image.Image,
        yolo_result:     dict,
        rag_context:     str,
        weather_summary: str = "",
        market_info:     str = "",
        language:        str = "en",
    ) -> str:
        """
        Algorithm 2 → success path.
        Qwen3-VL enriches the YOLO classification with visual symptom reasoning
        + RAG treatment advice + live weather/market context.
        """
        lang_rule = {
            "ta":       "முக்கியமாக தமிழில் பதில் சொல்லவும்.",
            "tanglish": "Tanglish-ல் பதில் சொல்லவும்.",
            "en":       "Respond in English; add Tamil translations for key terms in brackets.",
        }.get(language, "Respond in English.")

        sim_note = "\n⚠️ (YOLO model not loaded – result is simulated)" if yolo_result.get("simulated") else ""
        disease_block = (
            f"- Crop: {yolo_result.get('crop','Unknown')}\n"
            f"- Condition: {yolo_result.get('disease','Unknown')}\n"
            f"- Confidence: {yolo_result.get('confidence',0):.1%}\n"
            f"- Severity: {yolo_result.get('severity','?')}{sim_note}"
        )

        extra = ""
        if weather_summary: extra += f"\n\nCurrent Weather:\n{weather_summary}"
        if market_info:     extra += f"\n\nMarket Price for this crop:\n{market_info}"

        prompt = (
            f"Language rule: {lang_rule}\n\n"
            f"YOLOv11 Classification Result:\n{disease_block}\n\n"
            f"Using ONLY the RAG knowledge context below, provide the treatment advice:\n"
            f"{rag_context}"
            f"{extra}\n\n"
            f"Analyse this leaf image visually and provide:\n"
            f"1. **Visual Confirmation** – do you see symptoms matching the YOLO result?\n"
            f"2. **Symptom Description** – colour, spots, texture, margins you observe\n"
            f"3. **Severity Assessment** – early / moderate / severe\n"
            f"4. **Treatment Steps** – organic and chemical options with dosage\n"
            f"5. **Prevention for next season**\n"
            f"6. **Weather Risk Note** – if humidity/temp suggests fungal risk, warn farmer\n"
            f"7. **Market Note** – if price data available, optimal selling window\n\n"
            f"Be precise and practical for a Tamil Nadu farmer."
        )

        messages = [
            {"role":"system","content":"You are a bilingual (Tamil-English) agricultural plant pathologist."},
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{_pil_to_b64(pil_image)}"}},
                {"type":"text","text":prompt},
            ]},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=1100, temperature=0.55)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Vision analysis error: {e}"

    # ── Vision: ANOMALOUS image (REJECT path → Qwen visual assessment) ────────
    def analyze_anomalous_image(
        self,
        pil_image:     Image.Image,
        raw_score:     float,
        threshold:     float = -0.15,
        language:      str   = "en",
    ) -> str:
        """
        Paper Algorithm 1 REJECT path — but instead of hard-stopping,
        Qwen3-VL gives a PROBABILISTIC visual assessment with disclaimer:
        'Since this is an underclass image outside our 38 trained classes,
         we cannot give definitive advice, but the plant disease may be…'
        """
        lang_rule = {
            "ta":       "முக்கியமாக தமிழில் பதில் சொல்லவும்.",
            "tanglish": "Tanglish-ல் பதில் சொல்லவும்.",
            "en":       "Respond in English with Tamil translations for key terms.",
        }.get(language, "Respond in English.")

        prompt = (
            f"Language rule: {lang_rule}\n\n"
            f"**CONTEXT – Isolation Forest Gatekeeper Result:**\n"
            f"• Raw decision_function score: {raw_score:.4f}\n"
            f"• Threshold: {threshold} (paper Algorithm 1)\n"
            f"• Decision: REJECT — this image is OUTSIDE our 38 PlantVillage training classes.\n\n"
            f"This means: our YOLOv11 model was NOT trained on this type of image. "
            f"It could be a rare/regional disease, an unusual crop variety, environmental stress "
            f"(drought, nutrient deficiency, chemical burn), or a non-standard image angle.\n\n"
            f"**Your task — UNDERCLASS ASSESSMENT:**\n"
            f"Since this is an underclass image, you CANNOT give a definitive YOLO-backed diagnosis. "
            f"Instead, visually inspect the image and:\n"
            f"1. Describe what you SEE (colours, spots, wilting, texture, margins)\n"
            f"2. State clearly: 'This image is OUTSIDE our 38 trained disease classes'\n"
            f"3. Give your best probabilistic guess: 'The plant disease MAY BE…' / "
            f"'இது … ஆக இருக்கலாம்'\n"
            f"4. List 2-3 possible causes (disease, nutrient deficiency, stress)\n"
            f"5. Suggest general first-aid steps the farmer can take immediately\n"
            f"6. STRONGLY recommend: local Krishi Vigyan Kendra / agricultural extension officer\n\n"
            f"Be HONEST about uncertainty. Never fabricate a confident diagnosis.\n"
            f"Use hedging language: 'may be', 'possibly', 'இருக்கலாம்', 'சாத்தியம்'."
        )

        messages = [
            {"role":"system","content":"You are an honest, careful bilingual agricultural plant pathologist."},
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{_pil_to_b64(pil_image)}"}},
                {"type":"text","text":prompt},
            ]},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=900, temperature=0.60)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Vision analysis error: {e}"

    def get_farming_advice(self, crop_type, issue, location="Tamil Nadu", language="en"):
        lang_rule = {
            "ta": "தமிழில் பதில் சொல்லவும்.",
            "tanglish": "Tanglish-ல் பதில் சொல்லவும்.",
            "en": "Respond in English with Tamil term translations.",
        }.get(language, "Respond in English.")

        return self._chat([
            {"role":"system","content":f"You are a bilingual agricultural advisor for {location}. {lang_rule}"},
            {"role":"user","content":
             f"Crop: {crop_type}\nIssue: {issue}\n"
             "Give: immediate actions, long-term management, prevention, cost-effective solutions for small farmers."},
        ], max_tokens=800)