from __future__ import annotations

import gc
import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

# Road models — best backbone first
ROAD_WEIGHTS = [
    BASE_DIR / "models" / "road_yolo11x.pt",   # Nothingger YOLO11x — biggest backbone
    BASE_DIR / "models" / "road.pt",            # oracl4 YOLOv8s — different training set
]

# Waste models — ordered by quality
WASTE_WEIGHTS = [
    BASE_DIR / "models" / "waste_taco.pt",      # TACO 60-class — fine-grained street litter
    BASE_DIR / "models" / "waste_yolo11.pt",    # PhucHuwu 7-class — catches what TACO misses
]

# Road label normalization → canonical English
LABEL_TRANSLATE = {
    "Potholes": "Pothole",
    "Longitudinal Crack": "Linear Crack",
    "D00": "Linear Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "Repair": "Patch",
    "lubang": "Pothole",
    "retak-buaya": "Alligator Crack",
    "retak-garis": "Linear Crack",
    "tambalan": "Patch",
}

INFER_CONF = 0.10
INFER_IMGSZ = 640       # smaller = less RAM, still works well
INFER_AUGMENT = False
INFER_IOU = 0.6
INFER_MAX_DET = 300

TILE_SIZE = 640
TILE_OVERLAP = 0.25
TILE_MIN_IMG = 800

app = FastAPI(title="Dammage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
)


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _predict(model: YOLO, img: Image.Image, offset: tuple[float, float] = (0.0, 0.0)) -> list:
    results = model.predict(
        img,
        conf=INFER_CONF,
        iou=INFER_IOU,
        imgsz=INFER_IMGSZ,
        augment=INFER_AUGMENT,
        max_det=INFER_MAX_DET,
        verbose=False,
    )
    out = []
    if not results or results[0].boxes is None:
        return out
    ox, oy = offset
    names = model.names
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        out.append({
            "label": label,
            "confidence": float(box.conf[0]),
            "box": {"x1": x1 + ox, "y1": y1 + oy, "x2": x2 + ox, "y2": y2 + oy},
        })
    return out


def _tile_predict(model: YOLO, img: Image.Image) -> list:
    dets = _predict(model, img)
    w, h = img.size
    if max(w, h) >= TILE_MIN_IMG:
        step = int(TILE_SIZE * (1 - TILE_OVERLAP))
        seen: set[tuple[int, int]] = set()
        for y in list(range(0, max(1, h - TILE_SIZE), step)) + [max(0, h - TILE_SIZE)]:
            for x in list(range(0, max(1, w - TILE_SIZE), step)) + [max(0, w - TILE_SIZE)]:
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                tile = img.crop((x, y, min(x + TILE_SIZE, w), min(y + TILE_SIZE, h)))
                dets.extend(_predict(model, tile, offset=(float(x), float(y))))
    return dets


def _stage_predict(weights: Path, img: Image.Image) -> list:
    """Load one model, predict, then immediately free it from memory."""
    if not weights.exists():
        return []
    model = YOLO(str(weights))
    try:
        return _tile_predict(model, img)
    finally:
        del model
        _free_memory()


def _iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    ba = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return inter / (aa + ba - inter + 1e-9)


def _dedupe(dets: list, iou_thresh: float = 0.5) -> list:
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in dets:
        if not any(k["label"] == d["label"] and _iou(k["box"], d["box"]) > iou_thresh for k in kept):
            kept.append(d)
    return kept


async def read_image(file: UploadFile) -> tuple[Image.Image, tuple[int, int]]:
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    return img, img.size


def run_road(img: Image.Image) -> list:
    all_dets: list = []
    for weights in ROAD_WEIGHTS:
        # Stage-wise: one model loaded at a time
        dets = _stage_predict(weights, img)
        for d in dets:
            d["label"] = LABEL_TRANSLATE.get(d["label"], d["label"])
        all_dets.extend(dets)
    return _dedupe(all_dets)


def run_waste(img: Image.Image) -> list:
    img_area = img.size[0] * img.size[1]
    all_dets: list = []
    for weights in WASTE_WEIGHTS:
        # Stage-wise: one model loaded at a time
        dets = _stage_predict(weights, img)
        for d in dets:
            d["label"] = d["label"].replace("-", " ").replace("_", " ").strip().title()
        all_dets.extend(dets)
    # Drop boxes that cover >25% of frame (false positives on backgrounds)
    all_dets = [
        d for d in all_dets
        if (max(0.0, d["box"]["x2"] - d["box"]["x1"]) * max(0.0, d["box"]["y2"] - d["box"]["y1"]) / img_area) <= 0.25
    ]
    return _dedupe(all_dets, iou_thresh=0.4)


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/detect/road", "/detect/waste"]}


@app.post("/detect/road")
async def road_endpoint(file: UploadFile = File(...)):
    img, (w, h) = await read_image(file)
    return {"kind": "road", "width": w, "height": h, "detections": run_road(img)}


@app.post("/detect/waste")
async def waste_endpoint(file: UploadFile = File(...)):
    img, (w, h) = await read_image(file)
    return {"kind": "waste", "width": w, "height": h, "detections": run_waste(img)}
