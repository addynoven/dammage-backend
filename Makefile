.PHONY: install models run dev setup check

# First-time setup: install deps + download models
setup: install models

install:
	uv sync

models:
	bash scripts/download_models.sh

# Run server (production-ish)
run:
	uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Run server with hot reload for development
dev:
	uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Smoke test both endpoints (requires a test image)
check:
	@echo "--- Health ---"
	curl -s http://127.0.0.1:8000/ | python3 -m json.tool
	@echo "\n--- Road (test.jpg) ---"
	curl -s -X POST -F "file=@test.jpg" http://127.0.0.1:8000/detect/road | python3 -m json.tool
	@echo "\n--- Waste (test.jpg) ---"
	curl -s -X POST -F "file=@test.jpg" http://127.0.0.1:8000/detect/waste | python3 -m json.tool
