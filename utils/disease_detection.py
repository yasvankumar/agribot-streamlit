"""
Disease Detection — exact algorithm from paper:

Algorithm 1: Anomaly Detection (Gatekeeper) — Isolation Forest
  • Feature extraction: colour histograms + texture + edges
  • decision_function score < -0.15 → REJECT ("Invalid Input")
  • decision_function score >= -0.15 → ACCEPT (pass to YOLO)

Algorithm 2: Disease Classification — YOLOv11-Nano
  • Input: resize to 224×224, normalize 0-1, center crop
  • CSPNet backbone + C3k2 blocks + PANet neck
  • GAP → Softmax → 38 PlantVillage classes
  • Output: Class_Name + confidence %
"""

import os, pickle
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import sklearn  # noqa
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

DEFAULT_YOLO_PATH = os.getenv("YOLO_MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "models" / "best.pt"))
DEFAULT_ISO_PATH  = os.getenv("ISO_FOREST_PATH",
    str(Path(__file__).resolve().parents[1] / "models" / "isolation_forest.pkl"))

# ── PlantVillage 38-class label map ──────────────────────────────────────────
PLANT_VILLAGE_CLASSES = {
    "Apple___Apple_scab":
        ("Apple – Apple Scab", "Apply captan or mancozeb fungicide. Remove & destroy infected leaves."),
    "Apple___Black_rot":
        ("Apple – Black Rot", "Prune infected branches. Apply copper-based fungicide during dormancy."),
    "Apple___Cedar_apple_rust":
        ("Apple – Cedar Apple Rust", "Apply myclobutanil fungicide. Remove nearby cedar/juniper trees."),
    "Apple___healthy":
        ("Apple – Healthy", "No disease. Continue good cultural practices."),
    "Blueberry___healthy":
        ("Blueberry – Healthy", "Plant looks healthy. Maintain irrigation and nutrition."),
    "Cherry_(including_sour)___Powdery_mildew":
        ("Cherry – Powdery Mildew", "Apply sulphur-based fungicide. Improve air circulation."),
    "Cherry_(including_sour)___healthy":
        ("Cherry – Healthy", "Plant is healthy. Monitor regularly."),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        ("Maize – Gray Leaf Spot", "Apply strobilurin fungicides. Use resistant hybrids next season."),
    "Corn_(maize)___Common_rust_":
        ("Maize – Common Rust", "Apply triazole or strobilurin fungicide early."),
    "Corn_(maize)___Northern_Leaf_Blight":
        ("Maize – Northern Leaf Blight", "Apply fungicide at first sign. Rotate crops."),
    "Corn_(maize)___healthy":
        ("Maize – Healthy", "Crop is healthy. Ensure adequate nitrogen."),
    "Grape___Black_rot":
        ("Grape – Black Rot", "Apply myclobutanil or mancozeb. Remove mummified fruits."),
    "Grape___Esca_(Black_Measles)":
        ("Grape – Black Measles (Esca)", "No cure; remove infected wood. Apply preventive fungicide on wounds."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        ("Grape – Leaf Blight", "Apply copper fungicide. Improve vineyard air circulation."),
    "Grape___healthy":
        ("Grape – Healthy", "Vine is healthy. Continue standard management."),
    "Orange___Haunglongbing_(Citrus_greening)":
        ("Orange – Citrus Greening (HLB)", "No cure. Remove infected trees. Control Asian citrus psyllid."),
    "Peach___Bacterial_spot":
        ("Peach – Bacterial Spot", "Apply copper bactericide. Avoid overhead irrigation."),
    "Peach___healthy":
        ("Peach – Healthy", "Tree is healthy. Maintain proper pruning and fertilisation."),
    "Pepper,_bell___Bacterial_spot":
        ("Bell Pepper – Bacterial Spot", "Apply copper-based bactericide. Use certified disease-free seed."),
    "Pepper,_bell___healthy":
        ("Bell Pepper – Healthy", "Plant is healthy. Monitor for aphids and thrips."),
    "Potato___Early_blight":
        ("Potato – Early Blight", "Apply chlorothalonil or mancozeb. Remove lower infected leaves."),
    "Potato___Late_blight":
        ("Potato – Late Blight", "Apply metalaxyl or cymoxanil URGENTLY. Destroy infected haulm."),
    "Potato___healthy":
        ("Potato – Healthy", "Crop is healthy. Monitor soil moisture."),
    "Raspberry___healthy":
        ("Raspberry – Healthy", "Plant is healthy. Prune canes annually."),
    "Soybean___healthy":
        ("Soybean – Healthy", "Crop is healthy. Watch for Asian soybean rust."),
    "Squash___Powdery_mildew":
        ("Squash – Powdery Mildew", "Apply potassium bicarbonate or sulphur spray. Improve airflow."),
    "Strawberry___Leaf_scorch":
        ("Strawberry – Leaf Scorch", "Apply captan fungicide. Avoid overhead watering."),
    "Strawberry___healthy":
        ("Strawberry – Healthy", "Plant is healthy. Watch for grey mould."),
    "Tomato___Bacterial_spot":
        ("Tomato – Bacterial Spot", "Apply copper bactericide. Use drip irrigation. Rotate crops."),
    "Tomato___Early_blight":
        ("Tomato – Early Blight", "Apply mancozeb or chlorothalonil. Remove lower leaves. Mulch soil."),
    "Tomato___Late_blight":
        ("Tomato – Late Blight", "Apply metalaxyl URGENTLY. Destroy infected plants. Avoid night watering."),
    "Tomato___Leaf_Mold":
        ("Tomato – Leaf Mould", "Improve ventilation. Apply copper fungicide. Reduce humidity."),
    "Tomato___Septoria_leaf_spot":
        ("Tomato – Septoria Leaf Spot", "Apply chlorothalonil or mancozeb. Remove infected leaves."),
    "Tomato___Spider_mites Two-spotted_spider_mite":
        ("Tomato – Spider Mites", "Apply abamectin miticide or neem oil. Increase humidity."),
    "Tomato___Target_Spot":
        ("Tomato – Target Spot", "Apply azoxystrobin or chlorothalonil. Rotate crops."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        ("Tomato – Yellow Leaf Curl Virus", "No cure. Remove infected plants. Control whitefly with imidacloprid."),
    "Tomato___Tomato_mosaic_virus":
        ("Tomato – Mosaic Virus", "No cure. Remove infected plants. Disinfect tools. Control aphids."),
    "Tomato___healthy":
        ("Tomato – Healthy", "Plant is healthy. Monitor for late blight during rainy periods."),
}

# ── Feature extraction (Algorithm 1 spec: colour histograms + texture + edges) ─
_ISO_SIZE   = (64, 64)
_N_BINS     = 32


def _extract_features(pil_image: Image.Image) -> np.ndarray:
    """
    Extracts the same feature vector the Isolation Forest was trained on:
      • Colour histograms (R, G, B) – 32 bins each, normalised
      • Mean + std per channel (0-1 scaled)
      • Texture: variance across entire image
      • Edge strength: Sobel-like gradient magnitude mean
    """
    img = pil_image.convert("RGB").resize(_ISO_SIZE)
    arr = np.array(img, dtype=np.float32)

    features = []
    for ch in range(3):
        channel = arr[:, :, ch].ravel()
        hist, _ = np.histogram(channel, bins=_N_BINS, range=(0, 255))
        features.extend(hist / (hist.sum() + 1e-8))   # normalised histogram
        features.append(channel.mean() / 255.0)
        features.append(channel.std()  / 255.0)

    # texture: global variance
    gray = arr.mean(axis=2)
    features.append(float(gray.var()) / (255.0 ** 2))

    # edge strength: horizontal + vertical gradient (Sobel-like)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    features.append(float(gx) / 255.0)
    features.append(float(gy) / 255.0)

    return np.array(features, dtype=np.float32).reshape(1, -1)


def _preprocess_for_yolo(pil_image: Image.Image) -> Image.Image:
    """
    Algorithm 2 preprocessing (exact spec):
      1. Resize to 224×224
      2. Center crop (no-op if already square)
      3. Normalize pixel values 0-1 (ultralytics handles internally)
    """
    img = pil_image.convert("RGB")
    # Center crop to square
    w, h = img.size
    min_side = min(w, h)
    left   = (w - min_side) // 2
    top    = (h - min_side) // 2
    img    = img.crop((left, top, left + min_side, top + min_side))
    # Resize to 224×224
    img = img.resize((224, 224), Image.LANCZOS)
    return img


class DiseaseDetector:
    ANOMALY_THRESHOLD = -0.15   # Paper Algorithm 1, Step 5: score < -0.15 → REJECT

    def __init__(self, yolo_path=DEFAULT_YOLO_PATH, iso_path=DEFAULT_ISO_PATH):
        self.yolo_path   = yolo_path
        self.iso_path    = iso_path
        self.yolo_model  = None
        self.iso_model   = None
        self.load_errors = []
        self._load_yolo()
        self._load_iso()

    def _load_yolo(self):
        if not YOLO_AVAILABLE:
            self.load_errors.append("ultralytics not installed – pip install ultralytics")
            return
        if not os.path.exists(self.yolo_path):
            self.load_errors.append(f"YOLO model not found: {self.yolo_path}")
            return
        try:
            self.yolo_model = _YOLO(self.yolo_path)
        except Exception as e:
            self.load_errors.append(f"YOLO load error: {e}")

    def _load_iso(self):
        if not SKLEARN_AVAILABLE:
            self.load_errors.append("scikit-learn not installed – pip install scikit-learn")
            return
        if not os.path.exists(self.iso_path):
            self.load_errors.append(f"Isolation Forest not found: {self.iso_path}")
            return

        # ── Try joblib first (standard way scikit-learn models are saved) ──
        if JOBLIB_AVAILABLE:
            try:
                self.iso_model = joblib.load(self.iso_path)
                return   # success — stop here
            except Exception:
                pass     # joblib failed, fall through to pickle

        # ── Fallback: plain pickle ─────────────────────────────────────────
        try:
            with open(self.iso_path, "rb") as f:
                self.iso_model = pickle.load(f)
        except Exception as e:
            self.load_errors.append(
                f"Isolation Forest load error: {e}  "
                f"(tried joblib={'yes' if JOBLIB_AVAILABLE else 'not installed'} + pickle)"
            )

    @property
    def models_loaded(self):
        return {"yolo": self.yolo_model is not None,
                "isolation_forest": self.iso_model is not None}

    # ── Algorithm 1: Gatekeeper ────────────────────────────────────────────
    def check_anomaly(self, pil_image: Image.Image) -> dict:
        """
        Returns:
          is_anomaly: True if decision_function score < -0.15
          raw_score:  raw isolation forest decision_function score
          threshold:  -0.15 (from paper)
          status:     'REJECT' | 'ACCEPT'
        """
        if self.iso_model is None:
            return self._simulate_anomaly()

        try:
            feats = _extract_features(pil_image)
            raw_score = float(self.iso_model.decision_function(feats)[0])
            is_anomaly = raw_score < self.ANOMALY_THRESHOLD

            return {
                "is_anomaly":  is_anomaly,
                "raw_score":   round(raw_score, 4),
                "threshold":   self.ANOMALY_THRESHOLD,
                "status":      "REJECT" if is_anomaly else "ACCEPT",
                "description": (
                    "Image does not match known crop-leaf patterns (shoe, rock, random object, etc.)."
                    if is_anomaly else
                    "Valid crop leaf image — passing to YOLOv11 classifier."
                ),
            }
        except Exception as e:
            return {"is_anomaly": False, "raw_score": 0.0,
                    "threshold": self.ANOMALY_THRESHOLD, "status": "ACCEPT",
                    "description": f"Anomaly check error: {e}"}

    # ── Algorithm 2: YOLOv11 Classifier ───────────────────────────────────
    def detect_disease(self, pil_image: Image.Image) -> dict:
        """
        Preprocesses to 224×224 (center crop + normalize) then runs YOLO inference.
        """
        if self.yolo_model is None:
            return self._simulate_detection()

        try:
            preprocessed = _preprocess_for_yolo(pil_image)
            results = self.yolo_model.predict(preprocessed, verbose=False, imgsz=224)
            return self._parse(results)
        except Exception as e:
            return {"disease": f"Inference error: {e}", "crop": "Unknown",
                    "confidence": 0.0, "treatment": "Could not run YOLO.",
                    "prevention": "", "severity": "unknown", "class_raw": "", "error": True}

    def _parse(self, results) -> dict:
        r = results[0]
        # Classification head
        if hasattr(r, "probs") and r.probs is not None:
            top_idx    = int(r.probs.top1)
            confidence = float(r.probs.top1conf)
            class_raw  = r.names.get(top_idx, str(top_idx))
        # Detection head
        elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes):
            best       = int(r.boxes.conf.argmax())
            confidence = float(r.boxes.conf[best])
            cls_id     = int(r.boxes.cls[best])
            class_raw  = r.names.get(cls_id, str(cls_id))
        else:
            return {"disease": "No disease detected", "crop": "Unknown",
                    "confidence": 0.0, "treatment": "Try a clearer leaf photo.",
                    "prevention": "", "severity": "none", "class_raw": ""}

        if class_raw in PLANT_VILLAGE_CLASSES:
            friendly, treatment = PLANT_VILLAGE_CLASSES[class_raw]
        else:
            friendly  = class_raw.replace("___", " – ").replace("_", " ")
            treatment = "Consult local Krishi Vigyan Kendra."

        parts = friendly.split(" – ", 1)
        crop  = parts[0]
        cond  = parts[1] if len(parts) > 1 else friendly

        return {
            "disease":    cond,
            "crop":       crop,
            "confidence": round(confidence, 4),
            "treatment":  treatment,
            "prevention": "Regular scouting, resistant varieties, crop rotation.",
            "severity":   "high" if confidence > 0.80 else "medium" if confidence > 0.50 else "low",
            "class_raw":  class_raw,
        }

    # ── Simulation fallbacks ───────────────────────────────────────────────
    def _simulate_detection(self) -> dict:
        import random
        class_raw, (friendly, treatment) = random.choice(list(PLANT_VILLAGE_CLASSES.items()))
        parts = friendly.split(" – ", 1)
        return {"disease":    (parts[1] if len(parts) > 1 else friendly) + " (SIMULATED)",
                "crop":       parts[0],
                "confidence": round(random.uniform(0.55, 0.90), 2),
                "treatment":  treatment,
                "prevention": "Load YOLO model for real predictions.",
                "severity":   "medium", "class_raw": class_raw, "simulated": True}

    def _simulate_anomaly(self) -> dict:
        import random
        raw = round(random.uniform(-0.35, 0.20), 4)
        is_a = raw < self.ANOMALY_THRESHOLD
        return {"is_anomaly":  is_a, "raw_score": raw,
                "threshold":   self.ANOMALY_THRESHOLD,
                "status":      "REJECT" if is_a else "ACCEPT",
                "description": "Simulation – Isolation Forest not loaded."}

    # ── Utilities ──────────────────────────────────────────────────────────
    def get_all_known_diseases(self):
        return [v[0] for v in PLANT_VILLAGE_CLASSES.values()]

    def search_diseases(self, keyword: str):
        kw = keyword.lower()
        out = []
        for cr, (friendly, treatment) in PLANT_VILLAGE_CLASSES.items():
            if kw in friendly.lower() or kw in cr.lower():
                parts = friendly.split(" – ", 1)
                out.append({"crop": parts[0],
                             "disease": parts[1] if len(parts) > 1 else friendly,
                             "treatment": treatment})
        return out