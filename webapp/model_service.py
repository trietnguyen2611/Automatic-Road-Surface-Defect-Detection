import io
import time
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

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

class DefectDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = self._resolve_model_path(model_path)
        print(f"🔄 Đang tải mô hình YOLOv8 từ: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.classes = CLASSES
        print("✅ Mô hình YOLOv8 đã sẵn sàng phục vụ suy luận!")

    def _resolve_model_path(self, custom_path: Optional[str] = None) -> Path:
        if custom_path and Path(custom_path).exists():
            return Path(custom_path)
        
        # Tìm kiếm trọng số tốt nhất trong toàn bộ thư mục runs/
        best_candidates = sorted(
            list(self.base_dir.glob("runs/**/weights/best.pt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if best_candidates:
            print(f"🎯 Tìm thấy trọng số tốt nhất: {best_candidates[0]}")
            return best_candidates[0]
            
        last_candidates = sorted(
            list(self.base_dir.glob("runs/**/weights/last.pt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if last_candidates:
            return last_candidates[0]
            
        local_yolo = self.base_dir / "yolov8n.pt"
        if local_yolo.exists():
            return local_yolo
            
        return Path("yolov8n.pt")

    def reload_model(self, new_path: Optional[str] = None):
        """Tải lại mô hình khi có trọng số mới sau khi huấn luyện xong."""
        self.model_path = self._resolve_model_path(new_path)
        self.model = YOLO(str(self.model_path))
        print(f"🔄 Đã tải lại mô hình từ: {self.model_path}")

    def predict_image(self, image_bytes: bytes, conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        """
        Nhận diện khuyết tật từ raw image bytes.
        Trả về ảnh có vẽ bounding box (base64) và danh sách chi tiết các phát hiện.
        """
        start_time = time.time()
        
        # Decode ảnh từ bytes
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Không thể đọc định dạng hình ảnh tải lên.")
            
        h_img, w_img, _ = img_bgr.shape
        
        # Chạy suy luận với YOLOv8
        results = self.model.predict(
            source=img_bgr,
            conf=conf,
            iou=iou,
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
                
                # Cập nhật số lượng
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
        
        return {
            "image_base64": f"data:image/jpeg;base64,{img_b64}",
            "detections": detections,
            "total_defects": len(detections),
            "counts": counts,
            "latency_ms": latency_ms,
            "image_size": {"width": w_img, "height": h_img},
            "model_used": self.model_path.name
        }

    def predict_frame_json(self, frame_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        """
        Nhận diện cực nhanh từ webcam frame, trả về tọa độ để client render trên HTML5 Canvas.
        Giảm thiểu tối đa băng thông truyền tải mạng.
        """
        start_time = time.time()
        h_img, w_img, _ = frame_bgr.shape
        
        results = self.model.predict(
            source=frame_bgr,
            conf=conf,
            iou=iou,
            imgsz=320, # Dùng 320 cho real-time CPU stream tốc độ cao
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
                
        return {
            "boxes": boxes_out,
            "counts": counts,
            "total": len(boxes_out),
            "latency_ms": latency_ms,
            "fps": round(1000 / latency_ms, 1) if latency_ms > 0 else 0
        }
