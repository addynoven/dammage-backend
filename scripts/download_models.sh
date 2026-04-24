#!/usr/bin/env bash
# Downloads all YOLO weights used by the road damage + waste detection ensembles.
# Run from the backend/ directory:  bash scripts/download_models.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

dl() {
  local dst="$1" url="$2"
  if [ -s "models/$dst" ]; then
    echo "[skip] models/$dst already exists"
    return
  fi
  echo "[dl] models/$dst"
  curl -fsSL -o "models/$dst" "$url"
}

# Road damage detection models
dl road.pt          "https://raw.githubusercontent.com/oracl4/RoadDamageDetection/main/models/YOLOv8_Small_RDD.pt"
dl road_yolo11.pt   "https://media.githubusercontent.com/media/dayeeen/road-damage-detection-yolov11/main/models/best/best.pt"
dl road_yolo11x.pt  "https://huggingface.co/Nothingger/RDD_YOLO_pretrained/resolve/main/YOLOv11x_RDD_Trained.pt"
dl road_yolo12s.pt  "https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022/resolve/main/yolo12s_RDD2022_best.pt"

# Waste detection models
dl waste_yolo11.pt   "https://raw.githubusercontent.com/PhucHuwu/YOLOv8_Detecting_and_Classifying_Waste/main/new_improve/best.pt"
dl waste_taco.pt     "https://raw.githubusercontent.com/jeremy-rico/litter-detection/master/runs/detect/train/yolov8m_100epochs/weights/best.pt"
dl waste_yolo11l.pt  "https://huggingface.co/Oguri02/trash-detection-yolo11l/resolve/main/best.pt"
dl waste_material.pt "https://huggingface.co/HrutikAdsare/waste-detection-yolov8/resolve/main/best.pt"
dl waste_pile.pt     "https://raw.githubusercontent.com/palakaeron/Garbage-detection-ngr/main/models/best.pt"

echo "All models downloaded to models/"
ls -lh models/
