# AgriBot - AI Agricultural Assistant

Bilingual (Tamil / English) farming assistant for crop disease detection, agricultural Q&A, weather, and market information.

## Architecture

```text
Farmer -> Streamlit App
             |-- Chat and vision -> LM Studio-compatible API
             |-- Disease detection -> Isolation Forest + YOLOv11-Nano
             |-- Weather -> OpenWeatherMap API
             `-- Market -> Simulated / Agmarknet
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The disease-detection weights are included in `models/` and load automatically. To use alternate weights, set `YOLO_MODEL_PATH` or `ISO_FOREST_PATH`.

## Deploy to Streamlit Community Cloud

1. Create an app from this GitHub repository.
2. Set **Main file path** to `app.py`.

Disease detection runs with the included models. Chat and vision analysis require an LM Studio-compatible API endpoint reachable from the cloud app; the default `127.0.0.1` endpoint refers to the cloud container, not your computer. Configure `LM_STUDIO_URL` and `LM_STUDIO_MODEL` as environment variables when a remotely reachable inference service is available. Weather forecasts use simulated data until you enter an OpenWeatherMap API key in the app settings.

This deployment repository contains the runnable app and required inference models. Training datasets, experiments, and generated outputs from the original `AGRIBOT` folder remain outside this repository to avoid oversized cloud clones.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | OpenAI-compatible model server URL |
| `LM_STUDIO_MODEL` | `qwen/qwen3-vl-4b` | Model identifier |
| `YOLO_MODEL_PATH` | `models/best.pt` | Path to YOLO weights |
| `ISO_FOREST_PATH` | `models/isolation_forest.pkl` | Path to Isolation Forest |
| `OPENWEATHER_API_KEY` | *(none)* | OpenWeatherMap API key |

## Supported classes

38 PlantVillage classes across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
