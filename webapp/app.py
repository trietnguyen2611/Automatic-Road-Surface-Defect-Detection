import json
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from webapp.model_service import DefectDetector, CLASSES, CLASS_COLORS

app = FastAPI(
    title="Road Surface Defect Detection",
    description="Ứng dụng nhận diện và phát hiện khuyết tật mặt đường (Ổ gà, Vết nứt, Nắp cống) theo thời gian thực.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Khởi tạo model service singleton
detector = DefectDetector()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Trang chủ ứng dụng Web."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "classes": CLASSES,
            "class_colors": CLASS_COLORS,
            "model_name": detector.model_path.name
        }
    )

@app.get("/api/info")
async def get_model_info():
    """Thông tin về mô hình đang hoạt động."""
    return {
        "status": "ready",
        "model_file": detector.model_path.name,
        "classes": CLASSES,
        "colors": CLASS_COLORS
    }

@app.get("/api/samples")
async def get_sample_images(offset: int = 0, limit: int = 24):
    """Lấy danh sách các ảnh mẫu từ dataset để test nhanh với phân trang."""
    images_dir = BASE_DIR.parent / "Data" / "dts" / "images"
    if not images_dir.exists():
        images_dir = BASE_DIR.parent / "dts" / "images"
    if not images_dir.exists():
        images_dir = BASE_DIR.parent / "Data" / "dataset_split" / "images" / "test"
        
    sample_files = []
    total = 0
    if images_dir.exists():
        all_imgs = sorted(list(images_dir.glob("*.jpg")))
        total = len(all_imgs)
        offset = max(0, offset)
        limit = max(1, min(limit, 100))
        sample_files = [f.name for f in all_imgs[offset : offset + limit]]
        
    return {
        "samples": sample_files,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + len(sample_files)) < total
    }

@app.get("/api/sample/{filename}")
async def get_sample_file(filename: str):
    """Trả về file ảnh mẫu cụ thể."""
    from fastapi.responses import FileResponse
    images_dir = BASE_DIR.parent / "Data" / "dts" / "images"
    if not images_dir.exists():
        images_dir = BASE_DIR.parent / "dts" / "images"
    target = images_dir / filename
    if not target.exists():
        target = BASE_DIR.parent / "Data" / "dataset_split" / "images" / "test" / filename
    if target.exists():
        return FileResponse(str(target))
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.post("/api/predict/sample/{filename}")
async def predict_sample(filename: str, conf: float = 0.25, iou: float = 0.45):
    """Chạy phát hiện trực tiếp trên ảnh mẫu trong dataset."""
    images_dir = BASE_DIR.parent / "Data" / "dts" / "images"
    if not images_dir.exists():
        images_dir = BASE_DIR.parent / "dts" / "images"
    target = images_dir / filename
    if not target.exists():
        target = BASE_DIR.parent / "Data" / "dataset_split" / "images" / "test" / filename
    if target.exists():
        with open(target, "rb") as f:
            image_bytes = f.read()
        res = detector.predict_image(image_bytes, conf=conf, iou=iou)
        res["filename"] = filename
        return JSONResponse(res)
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.post("/api/model/reload")
async def reload_model():
    """Tải lại trọng số mới nhất khi vừa train xong."""
    detector.reload_model()
    return {
        "status": "reloaded",
        "model_file": detector.model_path.name
    }

@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45)
):
    """Xử lý hình ảnh tải lên và trả về kết quả phát hiện khuyết tật."""
    try:
        image_bytes = await file.read()
        result = detector.predict_image(image_bytes, conf=conf, iou=iou)
        result["filename"] = file.filename
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/predict/frame")
async def predict_frame(
    request: Request
):
    """Xử lý frame từ camera qua HTTP POST (dự phòng khi không dùng WebSocket)."""
    try:
        data = await request.json()
        image_data = data.get("image", "")
        conf = float(data.get("conf", 0.25))
        iou = float(data.get("iou", 0.45))
        
        if "," in image_data:
            image_data = image_data.split(",")[1]
            
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse({"error": "Invalid frame"}, status_code=400)
            
        res = detector.predict_frame_json(frame, conf=conf, iou=iou)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket Stream cho Webcam Real-time Detection độ trễ cực thấp.
    Client gửi frame Base64 JSON -> Server trả về tọa độ Bounding Box, FPS, latency.
    """
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                data = json.loads(raw_text)
                image_data = data.get("image", "")
                conf = float(data.get("conf", 0.25))
                iou = float(data.get("iou", 0.45))
                
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                    
                img_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    res = detector.predict_frame_json(frame, conf=conf, iou=iou)
                    await websocket.send_text(json.dumps(res))
                else:
                    await websocket.send_text(json.dumps({"error": "Frame decode failed"}))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.app:app", host="0.0.0.0", port=8000, reload=True)
