from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
ROAD_WEIGHTS = BASE_DIR / "models" / "road.pt"                     # oracl4 YOLOv8s RDD
ROAD_WEIGHTS_YOLO11 = BASE_DIR / "models" / "road_yolo11.pt"       # dayeeen YOLO11 RDD
ROAD_WEIGHTS_YOLO11X = BASE_DIR / "models" / "road_yolo11x.pt"     # Nothingger YOLO11x RDD (top)
ROAD_WEIGHTS_YOLO12 = BASE_DIR / "models" / "road_yolo12s.pt"      # rezzzq YOLOv12s RDD
WASTE_WEIGHTS_YOLO11 = BASE_DIR / "models" / "waste_yolo11.pt"     # PhucHuwu 7-class
WASTE_WEIGHTS_TACO = BASE_DIR / "models" / "waste_taco.pt"         # jeremy TACO 60-class
WASTE_WEIGHTS_YOLO11L = BASE_DIR / "models" / "waste_yolo11l.pt"   # Oguri02 YOLO11l 4-class
WASTE_WEIGHTS_MATERIAL = BASE_DIR / "models" / "waste_material.pt" # HrutikAdsare 8 material-class
WASTE_WEIGHTS_PILE = BASE_DIR / "models" / "waste_pile.pt"         # palakaeron: garbage/garbage_bin/overflow (pile as whole)

# Normalize class labels across all models to a single English taxonomy
LABEL_TRANSLATE = {
    # dayeeen (Indonesian)
    "lubang": "Pothole",
    "retak-buaya": "Alligator Crack",
    "retak-garis": "Linear Crack",
    "tambalan": "Patch",
    # oracl4
    "Potholes": "Pothole",
    "Longitudinal Crack": "Linear Crack",
    "Transverse Crack": "Transverse Crack",
    "Alligator Crack": "Alligator Crack",
    # rezzzq (RDD2022 codes)
    "D00": "Linear Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "Repair": "Patch",
}

# Waste: COCO-pretrained YOLOv8l (largest non-x model).
# ultralytics auto-downloads on first use.
WASTE_BASE_MODEL = "yolov8l.pt"
WASTE_CLASSES = {
    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "backpack", "handbag",
    "suitcase", "book", "vase", "scissors", "umbrella", "tie",
    "frisbee", "skateboard", "remote", "cell phone", "toothbrush",
}

# Inference quality knobs — max accuracy mode
INFER_CONF = 0.10
INFER_IMGSZ = 1536
INFER_AUGMENT = True
INFER_IOU = 0.6
INFER_MAX_DET = 500

# SAHI-style tile inference: split large images into overlapping tiles,
# run inference on each, merge boxes. Helps catch small potholes/cracks.
TILE_SIZE = 1024
TILE_OVERLAP = 0.25
TILE_MIN_IMG = 1100  # lowered so typical ~1200px web images also get tiled

app = FastAPI(title="Dammage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_models: dict[str, YOLO] = {}


def get_model(kind: str) -> YOLO:
    if kind in _models:
        return _models[kind]
    if kind == "road":
        if not ROAD_WEIGHTS.exists():
            raise HTTPException(500, f"Road model weights missing: {ROAD_WEIGHTS}")
        _models[kind] = YOLO(str(ROAD_WEIGHTS))
    elif kind == "road2":
        _models[kind] = YOLO(str(ROAD_WEIGHTS_YOLO11))
    elif kind == "road3":
        _models[kind] = YOLO(str(ROAD_WEIGHTS_YOLO11X))
    elif kind == "road4":
        _models[kind] = YOLO(str(ROAD_WEIGHTS_YOLO12))
    elif kind == "waste":
        _models[kind] = YOLO(WASTE_BASE_MODEL)
    elif kind == "waste_yolo11":
        _models[kind] = YOLO(str(WASTE_WEIGHTS_YOLO11))
    elif kind == "waste_taco":
        _models[kind] = YOLO(str(WASTE_WEIGHTS_TACO))
    elif kind == "waste_yolo11l":
        _models[kind] = YOLO(str(WASTE_WEIGHTS_YOLO11L))
    elif kind == "waste_material":
        _models[kind] = YOLO(str(WASTE_WEIGHTS_MATERIAL))
    elif kind == "waste_pile":
        _models[kind] = YOLO(str(WASTE_WEIGHTS_PILE))
    else:
        raise HTTPException(400, f"Unknown model kind: {kind}")
    return _models[kind]


async def read_image(file: UploadFile) -> tuple[Image.Image, tuple[int, int]]:
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    return img, img.size


def _predict(
    model: YOLO,
    img: Image.Image,
    offset=(0.0, 0.0),
    conf: float | None = None,
    imgsz: int | None = None,
    augment: bool | None = None,
):
    results = model.predict(
        img,
        conf=conf if conf is not None else INFER_CONF,
        iou=INFER_IOU,
        imgsz=imgsz if imgsz is not None else INFER_IMGSZ,
        augment=INFER_AUGMENT if augment is None else augment,
        max_det=INFER_MAX_DET,
        verbose=False,
    )
    names = model.names
    out = []
    if not results or results[0].boxes is None:
        return out, names
    ox, oy = offset
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        out.append({
            "label": label,
            "confidence": float(box.conf[0]),
            "box": {"x1": x1 + ox, "y1": y1 + oy, "x2": x2 + ox, "y2": y2 + oy},
        })
    return out, names


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aarea = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    barea = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aarea + barea - inter + 1e-9)


def _merge_dedupe(dets, iou_thresh=0.5):
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in dets:
        if any(
            k["label"] == d["label"] and _iou(k["box"], d["box"]) > iou_thresh
            for k in kept
        ):
            continue
        kept.append(d)
    return kept


def _tile_predict(model: YOLO, img: Image.Image, conf: float | None = None):
    dets, _ = _predict(model, img, conf=conf)
    w, h = img.size
    if max(w, h) >= TILE_MIN_IMG:
        step = int(TILE_SIZE * (1 - TILE_OVERLAP))
        ys = list(range(0, max(1, h - TILE_SIZE), step)) + [max(0, h - TILE_SIZE)]
        xs = list(range(0, max(1, w - TILE_SIZE), step)) + [max(0, w - TILE_SIZE)]
        seen = set()
        for y in ys:
            for x in xs:
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                x2, y2 = min(x + TILE_SIZE, w), min(y + TILE_SIZE, h)
                tile = img.crop((x, y, x2, y2))
                d, _ = _predict(model, tile, offset=(x, y), conf=conf)
                dets.extend(d)
    return dets


PILE_LABELS = {"Overflow", "Garbage", "Garbage Bin"}


def _area(b) -> float:
    return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])


def _bbox_iou(a, b) -> float:
    return _iou(a, b)


def _bbox_overlap_frac(inner, outer) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    ix1, iy1 = max(inner["x1"], outer["x1"]), max(inner["y1"], outer["y1"])
    ix2, iy2 = min(inner["x2"], outer["x2"]), min(inner["y2"], outer["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ia = _area(inner)
    return inter / ia if ia > 0 else 0.0


def _filter_waste(dets, img_w: int, img_h: int):
    """Size-based filtering:
    - Pile labels must cover >=8% of image (0.5% "Overflow" is a false positive on a sign).
    - Non-pile labels must cover <25% of image (real trash items are not half the frame).
    """
    img_area = img_w * img_h
    kept = []
    for d in dets:
        area_frac = _area(d["box"]) / img_area
        if d["label"] in PILE_LABELS:
            if area_frac >= 0.08:
                kept.append(d)
        elif area_frac <= 0.25:
            kept.append(d)
    return kept


def run_detection(model: YOLO, img: Image.Image, class_filter: set[str] | None = None):
    dets = _tile_predict(model, img)
    for d in dets:
        d["label"] = LABEL_TRANSLATE.get(d["label"], d["label"])
    merged = _merge_dedupe(dets)
    if class_filter is not None:
        merged = [d for d in merged if d["label"] in class_filter]
    return merged


def run_road_ensemble(img: Image.Image):
    """4-model ensemble:
    - oracl4 YOLOv8s RDD (English, 4 class)
    - dayeeen YOLO11 RDD (Indonesian, 4 class)
    - Nothingger YOLO11x RDD (English, 4 class, biggest backbone)
    - rezzzq YOLOv12s RDD (5 class incl. Repair/Patch)
    """
    all_dets = []
    for key in ("road", "road2", "road3", "road4"):
        all_dets.extend(_tile_predict(get_model(key), img))
    for d in all_dets:
        d["label"] = LABEL_TRANSLATE.get(d["label"], d["label"])
    return _merge_dedupe(all_dets)


def _normalize_waste_label(s: str) -> str:
    # Lowercase, replace dashes/underscores, title-case.
    s = s.replace("-", " ").replace("_", " ").strip()
    return s.title()


def run_waste_ensemble(img: Image.Image):
    """Trash-specific ensemble — 3 trustworthy models:
    - jeremy TACO (60 fine-grained street litter classes, precise boxes)
    - PhucHuwu YOLO11 (7 categorical classes)
    - palakaeron pile detector (pile-level detection)

    Previously included HrutikAdsare + Oguri02 — dropped; they fire huge boxes on
    buildings/signs. Size filter (_filter_waste) catches remaining oversized boxes.
    """
    all_dets = []
    all_dets.extend(_tile_predict(get_model("waste_taco"), img, conf=0.20))
    all_dets.extend(_tile_predict(get_model("waste_yolo11"), img, conf=0.35))

    # Pile detector: full image at native 640px (model's training size).
    # At 1536px this model hallucinates "overflow" on shop signs at high confidence.
    # augment=False — TTA also increases false positives on this model.
    pile_dets, _ = _predict(
        get_model("waste_pile"), img, conf=0.50, imgsz=640, augment=False,
    )
    all_dets.extend(pile_dets)

    for d in all_dets:
        d["label"] = _normalize_waste_label(d["label"])

    w, h = img.size
    all_dets = _filter_waste(all_dets, w, h)
    return _merge_dedupe(all_dets, iou_thresh=0.4)


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/detect/road", "/detect/waste"]}


@app.post("/detect/road")
async def detect_road(file: UploadFile = File(...)):
    img, (w, h) = await read_image(file)
    detections = run_road_ensemble(img)
    return {"kind": "road", "width": w, "height": h, "detections": detections}


@app.post("/detect/waste")
async def detect_waste(file: UploadFile = File(...)):
    img, (w, h) = await read_image(file)
    detections = run_waste_ensemble(img)
    return {"kind": "waste", "width": w, "height": h, "detections": detections}
