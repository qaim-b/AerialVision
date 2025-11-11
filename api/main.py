"""
AerialVision FastAPI Application
REST API and WebSocket for traffic monitoring
"""

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pathlib import Path
import base64
import io
from PIL import Image
import sys
import os

# Add models to path
sys.path.append(str(Path(__file__).parent.parent))
from models.detector import TrafficDetector
from models.tracker import VehicleTracker

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

# Initialize detector
detector = TrafficDetector(
    model_path='yolov11n.pt',
    conf_threshold=0.5
)

# Initialize tracker
tracker = VehicleTracker()

# Routes
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
        "version": "1.0.0"
    }


@app.post("/api/v1/detect")
async def detect_vehicles(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5)
):
    """
    Detect vehicles in uploaded image
    
    Args:
        file: Image file
        conf_threshold: Confidence threshold
        
    Returns:
        Detection results
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Update threshold
        detector.update_thresholds(conf=conf_threshold)
        
        # Detect
        result = detector.detect(image, return_image=True)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', result['annotated_image'])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        result['annotated_image_url'] = f"data:image/jpeg;base64,{img_base64}"
        
        # Remove raw image array
        del result['annotated_image']
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/track")
async def track_vehicles(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5)
):
    """
    Track vehicles in uploaded video
    
    Args:
        file: Video file
        conf_threshold: Confidence threshold
        
    Returns:
        Tracking statistics
    """
    try:
        # Save uploaded video
        video_path = Path("temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(await file.read())
        
        # Update threshold
        detector.update_thresholds(conf=conf_threshold)
        
        # Process video
        result = detector.detect_video(
            str(video_path),
            output_path="output_video.mp4"
        )
        
        # Cleanup
        video_path.unlink()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive frame
            data = await websocket.receive_json()
            
            # Decode base64 image
            img_data = base64.b64decode(data['frame'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame"})
                continue
            
            # Detect
            conf = data.get('conf_threshold', 0.5)
            detector.update_thresholds(conf=conf)
            result = detector.detect(frame, return_image=False)
            
            # Send results
            await websocket.send_json({
                'total_vehicles': result['total_vehicles'],
                'by_class': result['by_class'],
                'detections': result['detections'],
                'inference_time': result['inference_time'],
                'fps': round(1.0 / result['inference_time'], 1)
            })
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "model": "YOLOv11n",
        "device": detector.device,
        "conf_threshold": detector.conf_threshold,
        "iou_threshold": detector.iou_threshold,
        "img_size": detector.img_size
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
