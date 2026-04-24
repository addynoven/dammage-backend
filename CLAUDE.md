# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Download model weights (~400MB, fetched from GitHub release)
bash scripts/download_models.sh

# Run the server
uv run uvicorn main:app --host 127.0.0.1 --port 8000

# Manual endpoint test
curl -X POST -F "file=@image.jpg" http://127.0.0.1:8000/detect/road
curl -X POST -F "file=@image.jpg" http://127.0.0.1:8000/detect/waste
```

No linter, formatter, or test runner is configured. Python 3.12+ required.

## Architecture

Single-file FastAPI app (`main.py`) — all routes, ML inference logic, model loading, and helpers are co-located.

**Endpoints:**
- `GET /` — health check
- `POST /detect/road` — road damage detection (multipart image upload)
- `POST /detect/waste` — waste/litter detection (multipart image upload)

**Response shape** (both endpoints):
```json
{ "kind": "road|waste", "width": int, "height": int, "detections": [{ "label": str, "confidence": float, "box": {"x1","y1","x2","y2"} }] }
```

## ML Inference Pipeline

Both endpoints use a **multi-model ensemble** strategy. Models are lazy-loaded into a global `_models` dict on first request and cached in memory.

### Road (`run_road_ensemble`)
4 YOLO models (YOLOv8s, YOLO11, YOLO11x, YOLOv12s) run in parallel, results merged.

### Waste (`run_waste_ensemble`)
3 YOLO models (TACO 60-class, YOLO11 7-class, pile detector) run independently; their detections are combined.

### Shared pipeline steps
1. **Label normalization** — `LABEL_TRANSLATE` dict maps each model's class names to canonical English labels.
2. **SAHI tiling** — Images ≥ `TILE_MIN_IMG` (1100px) are split into 1024×1024 tiles with 25% overlap; boxes are re-projected back to original coordinates before deduplication.
3. **Deduplication** (`_merge_dedupe`) — Removes overlapping boxes by IoU > 0.5; highest-confidence box wins per label.
4. **Size filtering** (`_filter_waste`) — Waste only: pile labels require ≥8% image coverage; regular trash requires ≤25%.

### Key inference constants (tune here)
```python
INFER_CONF = 0.10       # Confidence threshold
INFER_IMGSZ = 1536      # Input resolution
INFER_AUGMENT = True    # Test-time augmentation
INFER_IOU = 0.6         # NMS IoU
TILE_SIZE = 1024
TILE_OVERLAP = 0.25
TILE_MIN_IMG = 1100
```

## Model Weights

Stored in `models/` (gitignored). 9 `.pt` files total — 4 road models, 5 waste models. The pile detector (`waste_pile.pt`) uses fixed `imgsz=640, augment=False` to match its training resolution and avoid false positives on signs.

`WASTE_BASE_MODEL = "yolov8l.pt"` is auto-downloaded by ultralytics on first use (COCO-pretrained backbone).

## CORS

Fully open (`allow_origins=["*"]`). Intentional — designed for hackathon frontend integration.
