# Dammage Backend — Road Damage & Waste Detection API

FastAPI backend serving two YOLO ensemble endpoints:

- `POST /detect/road` — 4-model RDD2022 ensemble (Potholes, Linear / Transverse / Alligator Cracks, Patch)
- `POST /detect/waste` — TACO + PhucHuwu + palakaeron-pile ensemble (plastic, glass, metal, paper, cardboard, medical, e-waste, banana/orange peel, overflow/garbage pile)

Both endpoints accept a multipart file upload (`file=<image>`) and return JSON with normalised English labels, confidence, and bounding boxes.

## Setup

```bash
# 1. Install deps (requires uv — https://github.com/astral-sh/uv)
uv sync

# 2. Download all model weights (~400 MB) into models/
bash scripts/download_models.sh

# 3. Run
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

## Models

Weights are not committed (GitHub's 100MB-per-file cap). `scripts/download_models.sh` fetches them from the original GitHub / HuggingFace sources.

| File | Source | Size | Purpose |
|------|--------|------|---------|
| `road.pt` | oracl4/RoadDamageDetection | 90 MB | YOLOv8s RDD2022, 4 classes |
| `road_yolo11.pt` | dayeeen/road-damage-detection-yolov11 | 19 MB | YOLO11 RDD (Indonesian labels, auto-translated) |
| `road_yolo11x.pt` | Nothingger/RDD_YOLO_pretrained | 114 MB | YOLO11x RDD, biggest backbone |
| `road_yolo12s.pt` | rezzzq/yolo12s-road-damage-rdd2022 | 19 MB | YOLOv12s RDD, adds Patch class |
| `waste_yolo11.pt` | PhucHuwu/YOLOv8_Detecting_and_Classifying_Waste | 6 MB | 7 categorical waste classes |
| `waste_taco.pt` | jeremy-rico/litter-detection | 52 MB | YOLOv8m TACO, 60 street-litter classes |
| `waste_yolo11l.pt` | Oguri02/trash-detection-yolo11l | 51 MB | YOLO11l, 4 material classes |
| `waste_material.pt` | HrutikAdsare/waste-detection-yolov8 | 52 MB | YOLOv8m, 8 material classes |
| `waste_pile.pt` | palakaeron/Garbage-detection-ngr | 6 MB | Pile-level detection (garbage / overflow / bin) |

## Inference tuning

Knobs in `main.py`:

- `INFER_CONF`, `INFER_IOU`, `INFER_IMGSZ`, `INFER_AUGMENT`, `INFER_MAX_DET` — defaults optimised for accuracy (1536px + TTA + low conf).
- `TILE_SIZE`, `TILE_OVERLAP`, `TILE_MIN_IMG` — SAHI-style tiling on large images for small-object recall.
- Per-model confidence overrides inside `run_waste_ensemble` to suppress noisy models.
- `_filter_waste` — drops oversized boxes (shop signs, buildings) and requires pile labels to cover ≥ 8% of the image.

Pile detector runs at its native 640 px with augment disabled — at 1536 px it hallucinates "overflow" on shop signs.

## API example

```bash
curl -X POST -F "file=@pothole.jpg" http://127.0.0.1:8000/detect/road
# {"kind":"road","width":1280,"height":768,"detections":[{"label":"Pothole","confidence":0.924,"box":{"x1":..,"y1":..,"x2":..,"y2":..}}, ...]}
```

## License

Model weights retain their original upstream licenses — see each source repo linked above.
