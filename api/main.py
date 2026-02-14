"""
AerialVision FastAPI Application
REST API and WebSocket for traffic monitoring
"""

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pathlib import Path
import base64
import io
from PIL import Image, ImageDraw
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add models to path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from models.detector import TrafficDetector
except Exception:
    TrafficDetector = None


# Initialize FastAPI
app = FastAPI(
    title="AerialVision API",
    description="AI-Powered Urban Traffic Monitoring System",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


class MockTrafficDetector:
    """Lightweight fallback detector for serverless environments."""

    def __init__(self, conf_threshold: float = 0.5, iou_threshold: float = 0.45, img_size: int = 640):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = "cpu-mock"

    def update_thresholds(self, conf: float = None, iou: float = None):
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou

    def detect(self, image, return_image: bool = True):
        start = datetime.now(timezone.utc).timestamp()
        frame = image if isinstance(image, np.ndarray) else np.array(Image.open(str(image)).convert("RGB"))
        if frame is None:
            raise ValueError("Invalid image for mock detector")

        gray = frame.mean(axis=2)
        h, w = gray.shape[:2]
        grid = 8
        cell_h = max(1, h // grid)
        cell_w = max(1, w // grid)

        pil_img = Image.fromarray(frame.astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)

        detections = []
        for gy in range(grid):
            for gx in range(grid):
                y1 = gy * cell_h
                y2 = min(h, (gy + 1) * cell_h)
                x1 = gx * cell_w
                x2 = min(w, (gx + 1) * cell_w)
                patch = gray[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                contrast = float(patch.max() - patch.min())
                if contrast < 40:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area < 500:
                    continue

                box_w = max(24, int((x2 - x1) * 0.8))
                box_h = max(18, int((y2 - y1) * 0.55))
                bx1 = max(0, int(x1 + (x2 - x1 - box_w) / 2))
                by1 = max(0, int(y1 + (y2 - y1 - box_h) / 2))
                bx2 = min(w - 1, bx1 + box_w)
                by2 = min(h - 1, by1 + box_h)
                if bx2 <= bx1 or by2 <= by1:
                    continue

                aspect = box_w / max(box_h, 1)
                if aspect < 0.9 or aspect > 4.0:
                    continue

                confidence = min(0.97, max(self.conf_threshold, 0.45 + (contrast / 255.0)))
                class_name = "truck" if area > ((h * w) / 55) else "car"
                if confidence < self.conf_threshold:
                    continue

                center_x = float((bx1 + bx2) / 2)
                center_y = float((by1 + by2) / 2)
                if any(abs(existing["bbox_center"][0] - center_x) < 14 and abs(existing["bbox_center"][1] - center_y) < 14 for existing in detections):
                    continue

                draw.rectangle([bx1, by1, bx2, by2], outline=(60, 240, 200), width=2)
                draw.text((bx1 + 2, max(0, by1 - 12)), f"{class_name}:{confidence:.2f}", fill=(60, 240, 200))

                detections.append({
                    "class_id": 0 if class_name == "car" else 1,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                    "bbox_center": [center_x, center_y],
                    "bbox_area": float((bx2 - bx1) * (by2 - by1)),
                })

        detections = detections[:35]
        by_class = {
            "car": sum(1 for d in detections if d["class_name"] == "car"),
            "van": 0,
            "truck": sum(1 for d in detections if d["class_name"] == "truck"),
            "bus": 0,
            "motor": 0,
            "bicycle": 0,
            "other": 0,
        }

        inference = max(0.001, datetime.now(timezone.utc).timestamp() - start)
        result = {
            "detections": detections,
            "count": len(detections),
            "total_vehicles": len(detections),
            "by_class": by_class,
            "inference_time": round(inference, 3),
            "image_size": [frame.shape[0], frame.shape[1]],
            "device": self.device,
        }
        if return_image:
            result["annotated_image"] = np.array(pil_img)
        return result

    def detect_video(self, video_path: str, output_path: str = None):
        return {
            "frames_processed": 0,
            "total_detections": 0,
            "avg_vehicles_per_frame": 0,
            "avg_inference_time": 0,
            "avg_fps": 0,
            "output_path": output_path,
            "mode": "mock",
            "note": "Video batch tracking is unavailable in serverless mock mode.",
        }


DETECTOR_MODE = "yolo"
try:
    if TrafficDetector is None:
        raise RuntimeError("TrafficDetector unavailable")
    detector = TrafficDetector(model_path="yolov11n.pt", conf_threshold=0.5)
except Exception:
    detector = MockTrafficDetector(conf_threshold=0.5, iou_threshold=0.45)
    DETECTOR_MODE = "mock"

RECENT_EVENTS: deque = deque(maxlen=200)


def compute_traffic_insights(detections: List[Dict[str, Any]], image_size: List[int]) -> Dict[str, Any]:
    if not image_size or len(image_size) < 2:
        image_size = [1, 1]

    height, width = image_size[0], image_size[1]
    image_area = float(max(width * height, 1))
    total = len(detections)

    class_counts: Dict[str, int] = {}
    confidence_values: List[float] = []
    hotspot = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
    heavy_count = 0

    for det in detections:
        class_name = str(det.get("class_name", "other")).lower()
        conf = float(det.get("confidence", 0.0))
        center = det.get("bbox_center", [0.0, 0.0])
        x = float(center[0]) if len(center) > 0 else 0.0
        y = float(center[1]) if len(center) > 1 else 0.0

        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidence_values.append(conf)

        if class_name in {"truck", "bus"}:
            heavy_count += 1

        if x <= width / 2 and y <= height / 2:
            hotspot["top_left"] += 1
        elif x > width / 2 and y <= height / 2:
            hotspot["top_right"] += 1
        elif x <= width / 2 and y > height / 2:
            hotspot["bottom_left"] += 1
        else:
            hotspot["bottom_right"] += 1

    density_per_mpx = round((total / image_area) * 1_000_000, 2)
    avg_confidence = round((sum(confidence_values) / max(len(confidence_values), 1)) * 100, 2)
    heavy_ratio = round((heavy_count / max(total, 1)) * 100, 2)

    if density_per_mpx >= 26:
        congestion_level = "high"
    elif density_per_mpx >= 12:
        congestion_level = "moderate"
    else:
        congestion_level = "low"

    dominant_class = max(class_counts.items(), key=lambda item: item[1])[0] if class_counts else "none"

    return {
        "density_per_mpx": density_per_mpx,
        "avg_confidence_pct": avg_confidence,
        "heavy_vehicle_ratio_pct": heavy_ratio,
        "congestion_level": congestion_level,
        "dominant_class": dominant_class,
        "hotspot_quadrants": hotspot,
    }


def record_event(event_type: str, total_vehicles: int, inference_time: float, insights: Dict[str, Any]) -> None:
    RECENT_EVENTS.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "total_vehicles": int(total_vehicles),
            "inference_ms": round(float(inference_time) * 1000, 2),
            "congestion_level": insights.get("congestion_level", "unknown"),
            "density_per_mpx": insights.get("density_per_mpx", 0.0),
        }
    )


@app.get("/")
async def root():
    """Serve dashboard"""
    dashboard_path = web_dir / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path))
    return {"message": "AerialVision API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "YOLOv11",
        "device": detector.device,
        "version": "1.0.0",
        "mode": DETECTOR_MODE,
    }


@app.post("/api/v1/detect")
async def detect_vehicles(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.45)
):
    try:
        contents = await file.read()
        image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        detector.update_thresholds(conf=conf_threshold, iou=iou_threshold)
        result = detector.detect(image, return_image=True)

        insights = compute_traffic_insights(result["detections"], result["image_size"])
        result["insights"] = insights
        record_event("detect", result["total_vehicles"], result["inference_time"], insights)

        annotated_image = result["annotated_image"]
        pil_annotated = Image.fromarray(annotated_image.astype(np.uint8)) if isinstance(annotated_image, np.ndarray) else annotated_image
        output = io.BytesIO()
        pil_annotated.save(output, format="JPEG")
        img_base64 = base64.b64encode(output.getvalue()).decode("utf-8")
        result["annotated_image_url"] = f"data:image/jpeg;base64,{img_base64}"

        del result["annotated_image"]
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/track")
async def track_vehicles(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.45)
):
    try:
        if DETECTOR_MODE == "mock":
            raise HTTPException(
                status_code=501,
                detail="Video tracking is unavailable in serverless mock mode. Run locally for full tracking."
            )

        video_path = Path("temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(await file.read())

        detector.update_thresholds(conf=conf_threshold, iou=iou_threshold)
        result = detector.detect_video(str(video_path), output_path="output_video.mp4")
        if video_path.exists():
            video_path.unlink()

        insights = {
            "density_per_mpx": 0.0,
            "avg_confidence_pct": 0.0,
            "heavy_vehicle_ratio_pct": 0.0,
            "congestion_level": "video_batch",
            "dominant_class": "unknown",
            "hotspot_quadrants": {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0},
        }
        record_event("track", result.get("avg_vehicles_per_frame", 0), result.get("avg_inference_time", 0), insights)
        result["insights"] = insights
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            img_data = base64.b64decode(data["frame"])
            frame = np.array(Image.open(io.BytesIO(img_data)).convert("RGB"))

            conf = data.get("conf_threshold", 0.5)
            iou = data.get("iou_threshold", 0.45)
            detector.update_thresholds(conf=conf, iou=iou)
            result = detector.detect(frame, return_image=False)
            insights = compute_traffic_insights(result["detections"], result["image_size"])
            record_event("stream", result["total_vehicles"], result["inference_time"], insights)

            await websocket.send_json(
                {
                    "total_vehicles": result["total_vehicles"],
                    "by_class": result["by_class"],
                    "detections": result["detections"],
                    "inference_time": result["inference_time"],
                    "fps": round(1.0 / result["inference_time"], 1),
                    "insights": insights,
                }
            )
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()


@app.get("/api/v1/stats")
async def get_stats():
    return {
        "model": "YOLOv11n",
        "device": detector.device,
        "conf_threshold": detector.conf_threshold,
        "iou_threshold": detector.iou_threshold,
        "img_size": detector.img_size,
        "recent_events_buffer": RECENT_EVENTS.maxlen,
        "mode": DETECTOR_MODE,
    }


@app.get("/api/v1/history")
async def get_history(limit: int = 20):
    limit = max(1, min(limit, 200))
    return {"events": list(RECENT_EVENTS)[-limit:]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
