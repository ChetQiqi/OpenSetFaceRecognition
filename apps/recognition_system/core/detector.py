from typing import List, Tuple

import cv2
import numpy as np
from mtcnn import MTCNN  # type: ignore


def _resolve_tf_device(device: str) -> str:
    """将外部设备参数转换为 TensorFlow 可识别的设备字符串。"""
    raw_device = (device or "auto").strip().lower()

    if raw_device == "auto":
        try:
            import tensorflow as tf  # type: ignore
            return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        except Exception:
            return "/CPU:0"

    if raw_device.startswith("/gpu") or raw_device.startswith("/cpu"):
        return raw_device.upper()

    if raw_device.startswith("cuda") or raw_device.startswith("gpu"):
        if ":" in raw_device:
            gpu_id = raw_device.split(":", 1)[1]
            return f"/GPU:{gpu_id}"
        return "/GPU:0"

    if raw_device.startswith("cpu"):
        return "/CPU:0"

    return "/CPU:0"

class FaceDetector:
    def __init__(
        self,
        img_size: int = 112,
        conf_threshold: float = 0.90,
        min_size: int = 40,
        backend: str = "mtcnn",
        yolo_weights: str = "",
        device: str = "cpu",
    ):
        self.img_size = img_size
        self.conf_threshold = float(conf_threshold)
        self.min_size = int(min_size)
        self.backend = "mtcnn"  # 强制使用 MTCNN

        # # 初始化 MTCNN
        # try:
        #     from mtcnn import MTCNN  # type: ignore
        # except ImportError as exc:
        #     raise RuntimeError(
        #         "MTCNN is not installed. Please run: pip install mtcnn"
        #     ) from exc

        tf_device = _resolve_tf_device(device)

        self._mtcnn = MTCNN(device=tf_device)
        self._retinaface = None
        self._yolo_model = None

        if backend not in ["mtcnn", "auto"]:
            print(f"⚠️ 检测器后端已统一改为 MTCNN，忽略参数: {backend}")

    def detect(self, bgr_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """使用 MTCNN 检测人脸"""
        h, w = bgr_image.shape[:2]
        boxes: List[Tuple[int, int, int, int]] = []

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        try:
            detections = self._mtcnn.detect_faces(rgb_image)
        except ValueError as exc:
            msg = str(exc)
            # 兜底：某些运行态下 MTCNN 内部设备仍可能是 "auto"，这里修正后重试一次
            if "Unknown attribute 'auto'" in msg or "device spec: 'auto'" in msg:
                try:
                    self._mtcnn._device = _resolve_tf_device("auto")
                    detections = self._mtcnn.detect_faces(rgb_image)
                except Exception:
                    return []
            else:
                return []
        except (IndexError, AttributeError):
            # MTCNN 在无人脸或特殊图像时会抛出 "pop from empty list" 等内部错误
            return []
        if not isinstance(detections, list):
            return []

        for item in detections:
            score = float(item.get("confidence", 0.0))
            if score < self.conf_threshold:
                continue
            box = item.get("box", None)
            if not box or len(box) != 4:
                continue

            x, y, bw, bh = [int(v) for v in box]
            x1 = max(0, min(x, w - 1))
            y1 = max(0, min(y, h - 1))
            x2 = max(0, min(x + bw, w - 1))
            y2 = max(0, min(y + bh, h - 1))
            bw = x2 - x1
            bh = y2 - y1
            if bw < self.min_size or bh < self.min_size:
                continue
            boxes.append((x1, y1, bw, bh))

        return boxes

    def crop_face(self, bgr_image: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.2) -> np.ndarray:
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        size = max(w, h) * (1.0 + margin)
        x1 = max(0, int(cx - size / 2.0))
        y1 = max(0, int(cy - size / 2.0))
        x2 = min(bgr_image.shape[1], int(cx + size / 2.0))
        y2 = min(bgr_image.shape[0], int(cy + size / 2.0))
        crop = bgr_image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

