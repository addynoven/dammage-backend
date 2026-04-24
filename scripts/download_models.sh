#!/usr/bin/env bash
# Downloads all YOLO weights used by the road damage + waste detection ensembles.
# Primary source: this repo's GitHub release (stable, guaranteed availability).
# Run from the backend/ directory:  bash scripts/download_models.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

BASE="https://github.com/ananddub/dammage-backend/releases/download/v1.0"

dl() {
  local name="$1"
  if [ -s "models/$name" ]; then
    echo "[skip] models/$name already exists"
    return
  fi
  echo "[dl]   models/$name"
  curl -fsSL -o "models/$name" "$BASE/$name"
}

dl road.pt
dl road_yolo11.pt
dl road_yolo11x.pt
dl road_yolo12s.pt
dl waste_yolo11.pt
dl waste_taco.pt
dl waste_yolo11l.pt
dl waste_material.pt
dl waste_pile.pt

echo
echo "All models in models/ — ready to run: uv run uvicorn main:app --port 8000"
ls -lh models/
