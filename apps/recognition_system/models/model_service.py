from pathlib import Path
from threading import RLock
from typing import Any, List, Optional, Tuple

import numpy as np

from apps.recognition_system.config import AppConfig
from apps.recognition_system.core.detector import FaceDetector
from apps.recognition_system.core.operations import build_runtime, extract_face_embedding
from apps.recognition_system.repositories import IdentityRepository


class OnnxEmbeddingModel:
    def __init__(self, weights_path: str, img_size: int = 112):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("当前环境未安装 onnxruntime，请先安装依赖") from exc

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(weights_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = img_size

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        face = face_rgb.astype(np.float32)
        face = (face - 127.5) / 127.5
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0).astype(np.float32)
        output = self.session.run(None, {self.input_name: face})[0]
        vector = np.asarray(output).reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            vector = vector / norm
        return vector


class ModelService:
    """Model 层：负责模型加载、检测器加载、特征库缓存和 embedding 推理。"""

    def __init__(self, config: AppConfig, repository: IdentityRepository):
        self.config = config
        self.repository = repository
        self._current_model_name = config.model_name
        self._current_weights_path = config.weights_path
        self._current_img_size = config.img_size
        self._current_device = config.device
        self._model = None
        self._detector = None
        self._gallery: List[Tuple[str, np.ndarray]] = []
        self._lock = RLock()

    def load(self, force_reload: bool = False) -> None:
        with self._lock:
            if self._model is not None and self._detector is not None and not force_reload:
                return

            if Path(self._current_weights_path).suffix.lower() == ".onnx":
                self._model = OnnxEmbeddingModel(
                    weights_path=self._current_weights_path,
                    img_size=self._current_img_size,
                )
                self._detector = FaceDetector(
                    img_size=self._current_img_size,
                    conf_threshold=self.config.detector_conf_threshold,
                    min_size=self.config.detector_min_size,
                    backend=self.config.detector_backend,
                    device=self._current_device,
                )
            else:
                self._model, self._detector = build_runtime(
                    weights_path=self._current_weights_path,
                    model_name=self._current_model_name,
                    img_size=self._current_img_size,
                    device=self._current_device,
                    det_conf_threshold=self.config.detector_conf_threshold,
                    det_min_size=self.config.detector_min_size,
                    detector_backend=self.config.detector_backend,
                )
            self.refresh_gallery()

    def is_loaded(self) -> bool:
        return self._model is not None and self._detector is not None

    def refresh_gallery(self) -> None:
        with self._lock:
            self._gallery = self.repository.load_gallery(self.config.gallery_mode)

    @property
    def model(self) -> Any:
        self.load()
        return self._model

    @property
    def detector(self) -> Any:
        self.load()
        return self._detector

    @property
    def gallery(self):
        self.load()
        return self._gallery

    def extract_embedding(self, image_bgr: np.ndarray):
        self.load()
        return extract_face_embedding(image_bgr, self._model, self._detector)

    def switch_model(
        self,
        model_name: str,
        weights_path: Optional[str] = None,
        img_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._current_model_name = model_name
            if weights_path:
                self._current_weights_path = weights_path
            if img_size:
                self._current_img_size = int(img_size)
            if device:
                self._current_device = device
            self._model = None
            self._detector = None
        self.load(force_reload=True)

    def get_runtime_info(self) -> dict:
        return {
            "model_name": self._current_model_name,
            "weights_path": self._current_weights_path,
            "framework": Path(self._current_weights_path).suffix.lower().lstrip("."),
            "img_size": self._current_img_size,
            "device": self._current_device,
            "loaded": self.is_loaded(),
            "gallery_size": len(self._gallery),
        }
