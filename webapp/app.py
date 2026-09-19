"""
FastAPI Server cho Hệ thống Nhận diện Khuyết tật Mặt đường (Flexi-YOLO & YOLOv8n Baseline)
Hỗ trợ chuyển đổi mô hình linh hoạt, đối chiếu thực nghiệm khoa học từ Paper PLOS ONE 2025.
"""

import json
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from webapp.model_service import DefectDetector, CLASSES, CLASS_COLORS, PAPER_METRICS

app = FastAPI(
    title="Flexi-YOLO: Road Surface Defect Detection",
    description="Ứng dụng nhận diện khuyết tật mặt đường theo Paper PLOS ONE 2025 (Wise-IoU, DCNv-C2f, AKConv, GAM, G-Head).",
    version="2.0.0"
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

# Khởi tạo model service singleton (mặc định Flexi-YOLO)
detector = DefectDetector(default_mode="flexi_yolo")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Trang chủ ứng dụng Web."""
    active_info = detector.get_active_info()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "classes": CLASSES,
            "class_colors": CLASS_COLORS,
            "active_model": active_info["mode"],
            "model_name": active_info["name"],
            "gflops": active_info["gflops"],
            "f1_score": active_info["f1_score"],
            "paper": PAPER_METRICS
        }
    )

@app.get("/api/info")
async def get_model_info():
    """Thông tin chi tiết về mô hình đang kích hoạt."""
    active_info = detector.get_active_info()
    return {
        "status": "ready",
        "active_model": active_info["mode"],
        "info": active_info,
        "available_models": list(PAPER_METRICS["models"].keys()),
        "classes": CLASSES,
        "colors": CLASS_COLORS
    }

@app.post("/api/model/select")
async def select_model(request: Request):
    """Chuyển đổi giữa Baseline (yolov8n) và Paper Method (flexi_yolo)."""
    try:
        body = await request.json()
        target_mode = body.get("mode", "flexi_yolo")
        new_info = detector.set_active_model(target_mode)
        return JSONResponse({
            "status": "success",
            "message": f"Đã kích hoạt mô hình: {new_info['name']}",
            "active_model": target_mode,
            "info": new_info
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/paper/benchmark")
async def get_paper_benchmark():
    """Trả về dữ liệu so sánh khoa học từ Paper (Bảng 3 Ablation & Bảng 4 SOTA)."""
    return JSONResponse({
        "paper_info": {
            "title": PAPER_METRICS["title"],
            "journal": PAPER_METRICS["journal"],
            "doi": PAPER_METRICS["doi"],
            "url": "https://doi.org/10.1371/journal.pone.0325993"
        },
        "ablation_study": [
            {"model": "YOLOv8n (Baseline)", "precision": "88.0%", "recall": "73.5%", "map50": "80.8%", "map50_95": "61.4%", "gflops": "8.1", "f1": "0.80"},
            {"model": "+ Wise-IoU", "precision": "90.3%", "recall": "73.5%", "map50": "81.4%", "map50_95": "61.5%", "gflops": "8.1", "f1": "0.81"},
            {"model": "+ DCNv-C2f", "precision": "88.1%", "recall": "75.4%", "map50": "82.6%", "map50_95": "62.5%", "gflops": "7.9", "f1": "0.81"},
            {"model": "+ AKConv", "precision": "91.6%", "recall": "74.8%", "map50": "82.4%", "map50_95": "62.3%", "gflops": "8.0", "f1": "0.82"},
            {"model": "+ GAM Attention", "precision": "87.4%", "recall": "76.1%", "map50": "82.5%", "map50_95": "62.4%", "gflops": "8.3", "f1": "0.81"},
            {"model": "Flexi-YOLO (All + G-Head)", "precision": "90.7%", "recall": "78.2%", "map50": "86.1%", "map50_95": "65.3%", "gflops": "7.6", "f1": "0.84", "highlight": True}
        ],
        "sota_comparison": [
            {"model": "YOLOv5n", "precision": "85.2%", "recall": "71.0%", "map50": "78.6%", "gflops": "4.5"},
            {"model": "YOLOv7-tiny", "precision": "86.7%", "recall": "72.4%", "map50": "79.8%", "gflops": "13.2"},
            {"model": "YOLOv8n (Baseline)", "precision": "88.0%", "recall": "73.5%", "map50": "80.8%", "gflops": "8.1"},
            {"model": "YOLOv9t", "precision": "87.1%", "recall": "72.9%", "map50": "79.5%", "gflops": "7.8"},
            {"model": "YOLOv10n", "precision": "88.5%", "recall": "74.1%", "map50": "81.2%", "gflops": "8.2"},
            {"model": "RT-DETR", "precision": "88.9%", "recall": "75.8%", "map50": "82.1%", "gflops": "56.9"},
            {"model": "Flexi-YOLO (Đề xuất)", "precision": "90.7%", "recall": "78.2%", "map50": "86.1%", "gflops": "7.6", "highlight": True}
        ]
    })

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
    info = detector.get_active_info()
    return {
        "status": "reloaded",
        "active_model": info["mode"],
        "model_name": info["name"],
        "model_file": info["model_file"]
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
async def predict_frame(request: Request):
    """Xử lý frame từ camera qua HTTP POST."""
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
    """WebSocket Stream cho Webcam Real-time Detection độ trễ cực thấp."""
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
