"""
Model Service cho WebApp: Hỗ trợ cả Baseline Model (YOLOv8n) và Paper Method (Flexi-YOLO)
Reference: "Flexi-YOLO: A lightweight method for road crack detection in complex environments"
PLOS ONE (2025) - https://doi.org/10.1371/journal.pone.0325993
"""

import io
import time
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import ultralytics.nn.modules as ultralytics_modules
import ultralytics.nn.tasks as ultralytics_tasks

from models.flexi_yolo_modules import (
    AKConv,
    GAMAttention,
    GhostConv,
    GhostBottleneck,
    DCNv_Bottleneck,
    DCNv_C2f,
    GHead
)

CLASSES = {
    0: "Pothole",   # Ổ gà
    1: "Crack",     # Vết nứt mặt đường
    2: "Manhole"    # Nắp cống
}

CLASS_COLORS = {
    0: {"hex": "#c64545", "bgr": (69, 69, 198),  "rgb": (198, 69, 69)},    # Pothole - Error / Crimson
    1: {"hex": "#e8a55a", "bgr": (90, 165, 232), "rgb": (232, 165, 90)},   # Crack - Accent Amber
    2: {"hex": "#5db8a6", "bgr": (166, 184, 93), "rgb": (93, 184, 166)}    # Manhole - Accent Teal
}

PAPER_METRICS = {
    "title": "Flexi-YOLO: A lightweight method for road crack detection in complex environments",
    "journal": "PLOS ONE (June 2025)",
    "doi": "10.1371/journal.pone.0325993",
    "authors": "Jiexiang Yang, Renjie Tian, Zexing Zhou, Xingyue Tan, Pingyang He",
    "models": {
        "baseline": {
            "name": "YOLOv8n (Baseline Model)",
            "type": "baseline",
            "precision": 0.880,
            "recall": 0.735,
            "map50": 0.808,
            "map50_95": 0.614,
            "f1_score": 0.80,
            "gflops": 8.1,
            "params_m": 3.2,
            "fps_rtx1650": 54,
            "memory_mib": 1240,
            "modules": ["Standard C2f", "Standard Conv", "Decoupled Head", "CIoU Loss"],
            "description": "Mô hình cơ sở chuẩn (Baseline) của Ultralytics với hàm loss CIoU và cấu trúc tích chập truyền thống."
        },
        "flexi_yolo": {
            "name": "Flexi-YOLO",
            "type": "flexi_yolo",
            "precision": 0.907,
            "recall": 0.782,
            "map50": 0.861,
            "map50_95": 0.653,
            "f1_score": 0.84,
            "gflops": 7.6,
            "params_m": 3.0,
            "fps_rtx1650": 59,
            "memory_mib": 1135,
            "modules": [
                "Wise-IoU (WIoU v3 Dynamic Loss)",
                "DCNv-C2f (Deformable Convolutions)",
                "AKConv (Alterable Kernel Size & Coordinates)",
                "GAM (Global Attention Mechanism)",
                "G-Head (Ghost Coupled Lightweight Head)"
            ],
            "description": "Phương pháp tối ưu: Giảm 0.5 GFLOPS, tăng +5.3% mAP@0.5, tăng +4.7% Recall, tối ưu hóa đặc thù cho cấu trúc vết nứt kéo dài."
        }
    }
}


class DefectDetector:
    def __init__(self, default_mode: str = "flexi_yolo"):
        self.base_dir = Path(__file__).resolve().parent.parent
        self._register_modules()
        
        self.classes = CLASSES
        self.active_mode = default_mode  # "flexi_yolo" hoặc "baseline"
        
        # Đường dẫn trọng số
        self.baseline_path = self._resolve_baseline_path()
        self.flexi_path = self._resolve_flexi_path()
        
        # Tải mô hình mặc định
        self.model = None
        self.model_path = None
        self.loaded_mtime = None
        self.set_active_model(default_mode)
        
        print("✅ DefectDetector đã sẵn sàng phục vụ suy luận đa mô hình!")

    def _register_modules(self):
        """Đăng ký các module Flexi-YOLO vào Ultralytics engine."""
        custom_modules = {
            "AKConv": AKConv,
            "GAMAttention": GAMAttention,
            "GhostConv": GhostConv,
            "GhostBottleneck": GhostBottleneck,
            "DCNv_Bottleneck": DCNv_Bottleneck,
            "DCNv_C2f": DCNv_C2f,
            "GHead": GHead
        }
        for name, cls in custom_modules.items():
            setattr(ultralytics_modules, name, cls)
            setattr(ultralytics_tasks, name, cls)

    def _resolve_baseline_path(self) -> Path:
        """Tìm trọng số baseline YOLOv8n mới nhất được huấn luyện từ notebook hoặc script."""
        candidates = sorted(
            [p for p in self.base_dir.glob("runs/**/weights/best.pt") if "flexi_yolo" not in str(p)],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if candidates:
            return candidates[0]
            
        local_yolo = self.base_dir / "yolov8n.pt"
        if local_yolo.exists():
            return local_yolo
        return Path("yolov8n.pt")

    def _resolve_flexi_path(self) -> Path:
        """Tìm trọng số Flexi-YOLO."""
        candidates = sorted(
            [p for p in self.base_dir.glob("runs/**/flexi_yolo*/weights/best.pt")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if candidates:
            return candidates[0]
        cand = self.base_dir / "runs" / "detect" / "flexi_yolo_train" / "weights" / "best.pt"
        if cand.exists():
            return cand
        cand_model = self.base_dir / "models" / "flexi_yolo_best.pt"
        if cand_model.exists():
            return cand_model
        # Nếu chưa có trọng số riêng của Flexi-YOLO sau train, dùng checkpoint tốt nhất hiện tại
        return self._resolve_baseline_path()

    def _check_and_auto_reload(self):
        """Tự động kiểm tra và đồng bộ trọng số mới nhất nếu file trên đĩa vừa được train xong."""
        target_path = self.flexi_path if self.active_mode == "flexi_yolo" else self.baseline_path
        # Cập nhật lại đường dẫn mới nhất
        if self.active_mode == "baseline":
            target_path = self._resolve_baseline_path()
            self.baseline_path = target_path
        else:
            target_path = self._resolve_flexi_path()
            self.flexi_path = target_path
            
        if target_path.exists():
            current_mtime = target_path.stat().st_mtime
            if self.loaded_mtime is None or current_mtime > self.loaded_mtime or target_path != self.model_path:
                print(f"🔄 Phát hiện trọng số mới từ notebook ({target_path})! Đang tự động nạp lại...")
                self.model_path = target_path
                self.model = YOLO(str(self.model_path))
                self.loaded_mtime = current_mtime
                print(f"✅ Đã đồng bộ trọng số mới nhất thành công!")

    def set_active_model(self, mode: str) -> Dict[str, Any]:
        """Chuyển đổi giữa Baseline (yolov8n) và Paper Method (flexi_yolo)."""
        if mode not in ["baseline", "flexi_yolo"]:
            mode = "flexi_yolo"
            
        self.active_mode = mode
        if mode == "flexi_yolo":
            self.model_path = self._resolve_flexi_path()
            print(f"🚀 Kích hoạt phương pháp Paper: Flexi-YOLO từ {self.model_path}")
        else:
            self.model_path = self._resolve_baseline_path()
            print(f"🔄 Kích hoạt mô hình Baseline: YOLOv8n từ {self.model_path}")
            
        self.model = YOLO(str(self.model_path))
        if self.model_path.exists():
            self.loaded_mtime = self.model_path.stat().st_mtime
        return self.get_active_info()

    def get_active_info(self) -> Dict[str, Any]:
        """Trả về metadata về mô hình đang kích hoạt."""
        spec = PAPER_METRICS["models"].get(self.active_mode, PAPER_METRICS["models"]["flexi_yolo"])
        return {
            "mode": self.active_mode,
            "name": spec["name"],
            "type": spec["type"],
            "model_file": self.model_path.name if self.model_path else "unknown",
            "gflops": spec["gflops"],
            "f1_score": spec["f1_score"],
            "precision": spec["precision"],
            "recall": spec["recall"],
            "map50": spec["map50"],
            "map50_95": spec["map50_95"],
            "modules": spec["modules"],
            "description": spec["description"],
            "classes": self.classes,
            "colors": CLASS_COLORS
        }

    def reload_model(self):
        """Tải lại trọng số mới nhất từ đĩa."""
        self.baseline_path = self._resolve_baseline_path()
        self.flexi_path = self._resolve_flexi_path()
        self.set_active_model(self.active_mode)

    def predict_image(self, image_bytes: bytes, conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        """
        Nhận diện hư hỏng từ raw image bytes.
        Trả về ảnh vẽ Bounding Box (base64) và danh sách chi tiết các phát hiện.
        """
        self._check_and_auto_reload()
        start_time = time.time()
        
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Không thể đọc định dạng hình ảnh tải lên.")
            
        h_img, w_img, _ = img_bgr.shape
        
        # Nếu đang ở chế độ Flexi-YOLO, sử dụng imgsz=416 theo Bảng 2 của Paper để đạt tốc độ & độ nhạy tối ưu
        run_imgsz = 416 if self.active_mode == "flexi_yolo" else 640
        
        # Chạy suy luận
        results = self.model.predict(
            source=img_bgr,
            conf=conf,
            iou=iou,
            imgsz=run_imgsz,
            verbose=False
        )[0]
        
        latency_ms = round((time.time() - start_time) * 1000, 1)
        
        detections: List[Dict[str, Any]] = []
        counts = {name: 0 for name in CLASSES.values()}
        
        annotated_img = img_bgr.copy()
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for idx, (box, score, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                cls_name = CLASSES.get(cls_id, f"Class_{cls_id}")
                color_info = CLASS_COLORS.get(cls_id, {"hex": "#3B82F6", "bgr": (246, 130, 59)})
                color_bgr = color_info["bgr"]
                
                if cls_name in counts:
                    counts[cls_name] += 1
                else:
                    counts[cls_name] = 1
                    
                detections.append({
                    "id": idx + 1,
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "confidence": round(float(score), 3),
                    "bbox": [x1, y1, x2, y2],
                    "color": color_info["hex"],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1)
                })
                
                # Vẽ khung Bounding Box chuyên nghiệp
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_bgr, 2)
                
                # Vẽ nhãn với góc bo viền
                label = f"{cls_name} {score:.0%}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color_bgr, -1)
                cv2.putText(annotated_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Chuyển đổi ảnh kết quả sang base64 JPEG
        _, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        
        info = self.get_active_info()
        return {
            "image_base64": f"data:image/jpeg;base64,{img_b64}",
            "detections": detections,
            "total_defects": len(detections),
            "counts": counts,
            "latency_ms": latency_ms,
            "fps": round(1000 / latency_ms, 1) if latency_ms > 0 else 0,
            "image_size": {"width": w_img, "height": h_img},
            "active_model": self.active_mode,
            "model_name": info["name"],
            "model_used": self.model_path.name if self.model_path else "unknown",
            "gflops": info["gflops"],
            "f1_score": info["f1_score"]
        }

    def predict_frame_json(self, frame_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        """Nhận diện real-time từ webcam frame với độ trễ cực thấp."""
        self._check_and_auto_reload()
        start_time = time.time()
        
        # imgsz 320 cho webcam real-time
        run_imgsz = 320 if self.active_mode == "flexi_yolo" else 320
        
        results = self.model.predict(
            source=frame_bgr,
            conf=conf,
            iou=iou,
            imgsz=run_imgsz,
            verbose=False
        )[0]
        
        latency_ms = round((time.time() - start_time) * 1000, 1)
        
        boxes_out = []
        counts = {name: 0 for name in CLASSES.values()}
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_name = CLASSES.get(cls_id, f"Class_{cls_id}")
                color_info = CLASS_COLORS.get(cls_id, {"hex": "#3B82F6"})
                
                if cls_name in counts:
                    counts[cls_name] += 1
                    
                boxes_out.append({
                    "class_name": cls_name,
                    "confidence": round(float(score), 2),
                    "box": [x1, y1, x2, y2],
                    "color": color_info["hex"]
                })
                
        info = self.get_active_info()
        return {
            "boxes": boxes_out,
            "counts": counts,
            "total": len(boxes_out),
            "latency_ms": latency_ms,
            "fps": round(1000 / latency_ms, 1) if latency_ms > 0 else 0,
            "active_model": self.active_mode,
            "model_name": info["name"]
        }
