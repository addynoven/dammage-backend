# Dammage — Road Damage & Waste Detection Platform

## What This Project Does

**Dammage** is an AI-powered urban infrastructure inspection tool. Users upload a photo taken from a road or public space, and the system automatically detects and locates:

- **Road damage** — potholes, cracks, patched areas
- **Waste / litter** — trash items, garbage piles, overflowing bins

The backend is a REST API. The frontend sends an image, receives a list of detected objects with bounding boxes and confidence scores, and renders them visually on top of the photo.

The core use case: a citizen, city inspector, or drone operator takes a photo → uploads it → immediately sees what's wrong and where. No manual labeling. No expertise required.

---

## Base URL

```
http://127.0.0.1:8000
```

CORS is fully open — any origin can call these endpoints directly from the browser.

---

## Endpoints

### `GET /`

Health check. Returns confirmation the server is running.

**Response:**
```json
{
  "status": "ok",
  "endpoints": ["/detect/road", "/detect/waste"]
}
```

---

### `POST /detect/road`

Detects road surface damage in the uploaded image.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | image file | Any common format: JPG, PNG, WEBP. Max recommended size: ~5MB. |

**Example (browser `fetch`):**
```js
const form = new FormData();
form.append("file", imageFile);

const res = await fetch("http://127.0.0.1:8000/detect/road", {
  method: "POST",
  body: form,
});
const data = await res.json();
```

**Response:**
```json
{
  "kind": "road",
  "width": 1280,
  "height": 768,
  "detections": [
    {
      "label": "Pothole",
      "confidence": 0.924,
      "box": { "x1": 312, "y1": 455, "x2": 489, "y2": 601 }
    },
    {
      "label": "Alligator Crack",
      "confidence": 0.761,
      "box": { "x1": 100, "y1": 200, "x2": 400, "y2": 350 }
    }
  ]
}
```

**Possible `label` values:**

| Label | What it means |
|-------|---------------|
| `Pothole` | A hole or depression in the road surface |
| `Linear Crack` | A long straight crack running along or across the road |
| `Transverse Crack` | A crack perpendicular to the direction of traffic |
| `Alligator Crack` | A network of interconnected cracks resembling reptile skin — indicates structural failure |
| `Patch` | A previously repaired section of road (filled pothole, applied material) |

---

### `POST /detect/waste`

Detects waste, litter, and garbage in the uploaded image.

**Request:** Same as `/detect/road` — `multipart/form-data` with a `file` field.

**Response:**
```json
{
  "kind": "waste",
  "width": 1920,
  "height": 1080,
  "detections": [
    {
      "label": "Bottle",
      "confidence": 0.872,
      "box": { "x1": 530, "y1": 710, "x2": 620, "y2": 900 }
    },
    {
      "label": "Garbage",
      "confidence": 0.913,
      "box": { "x1": 0, "y1": 400, "x2": 800, "y2": 1080 }
    }
  ]
}
```

**Possible `label` values (examples — not exhaustive):**

| Category | Example labels |
|----------|---------------|
| Containers | `Bottle`, `Cup`, `Can`, `Carton`, `Box` |
| Packaging | `Plastic Bag`, `Wrapper`, `Styrofoam`, `Blister Pack` |
| Paper | `Cardboard`, `Paper`, `Newspaper` |
| Food waste | `Banana Peel`, `Orange Peel`, `Food Waste` |
| Medical | `Mask`, `Glove`, `Syringe` |
| Electronic | `Battery`, `Cable` |
| Piles / bins | `Garbage` (large pile), `Garbage Bin`, `Overflow` (overflowing bin) |

The waste model uses 60+ fine-grained TACO classes merged with broader category labels. The frontend should expect varied label strings and handle them gracefully (e.g., display whatever label comes back).

---

## Response Schema (both endpoints)

```ts
interface DetectionResponse {
  kind: "road" | "waste";
  width: number;          // original image width in pixels
  height: number;         // original image height in pixels
  detections: Detection[];
}

interface Detection {
  label: string;          // what was detected
  confidence: number;     // 0.0–1.0 — how certain the model is
  box: BoundingBox;
}

interface BoundingBox {
  x1: number;  // left edge (pixels, from left of image)
  y1: number;  // top edge (pixels, from top of image)
  x2: number;  // right edge
  y2: number;  // bottom edge
}
```

All bounding box coordinates are in **pixels relative to the original image dimensions** returned in `width` / `height`.

---

## Error Responses

| Status | When it happens |
|--------|----------------|
| `400` | File is not a valid image |
| `500` | Model weights not found on server |

Error body:
```json
{ "detail": "Invalid image: ..." }
```

---

## How to Render Bounding Boxes

The `box` values are pixel coordinates. To draw them on a displayed image:

```js
// Scale box to the displayed size of the image
const scaleX = displayedWidth / data.width;
const scaleY = displayedHeight / data.height;

const left   = detection.box.x1 * scaleX;
const top    = detection.box.y1 * scaleY;
const width  = (detection.box.x2 - detection.box.x1) * scaleX;
const height = (detection.box.y2 - detection.box.y1) * scaleY;
```

Draw a rectangle overlay at `(left, top)` with `width × height`. Label it with `detection.label` and optionally `detection.confidence` formatted as a percentage.

---

## UX Considerations

### Loading time
- **First request** after the server starts is slow (5–15 seconds) because all YOLO models are loaded into memory on demand. Subsequent requests are fast (1–5 seconds depending on image size).
- Show a loading indicator. Consider adding a message like "Warming up models..." on first use.

### Empty results
- `detections` can be an empty array `[]` — the image had no damage / no waste detected.
- Show a friendly "Nothing detected" state, not an error.

### Confidence display
- Confidence ranges from ~0.10 (minimum threshold) to 1.0.
- Suggested UX: color-code boxes by confidence (red = high confidence, yellow = medium) or show percentage badge on the box label.

### Multiple detections
- A single image can have many detections (up to ~500, but typically 2–20).
- Consider a sidebar list view alongside the image overlay so users can see all detections at a glance.

### Two modes
- The app has two completely separate detection modes: **Road** and **Waste**.
- Design clear mode switching — the user should always know which detection type they're running.

---

## Suggested Frontend Flow

```
1. User selects mode: "Road Inspection" or "Waste Detection"
2. User uploads or drags an image
3. Frontend sends POST to /detect/road or /detect/waste
4. Show loading state
5. Receive response → render image with bounding box overlays
6. Show a list/count of detections: label + confidence
7. Allow user to upload another image
```

---

## Color Palette Suggestions (for labels)

**Road damage:**
| Label | Suggested color |
|-------|----------------|
| Pothole | Red `#EF4444` |
| Linear Crack | Orange `#F97316` |
| Transverse Crack | Amber `#F59E0B` |
| Alligator Crack | Deep red `#DC2626` |
| Patch | Blue `#3B82F6` |

**Waste:**
| Category | Suggested color |
|----------|----------------|
| Individual items (bottles, cups, etc.) | Orange `#F97316` |
| Garbage pile / Overflow | Red `#EF4444` |
| Garbage Bin | Yellow `#EAB308` |

---

## Tech Stack (backend — for context)

| Piece | Detail |
|-------|--------|
| Runtime | Python 3.12 |
| Framework | FastAPI |
| ML models | YOLOv8 / YOLO11 / YOLOv12 via ultralytics |
| Strategy | 4-model ensemble (road) + 3-model ensemble (waste) |
| CORS | Fully open (`*`) — no auth required |
| Auth | None — open API |
