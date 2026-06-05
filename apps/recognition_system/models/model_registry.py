from threading import RLock
from typing import Dict, Optional

from apps.recognition_system.models.model_service import ModelService
from apps.recognition_system.repositories import ModelRepository


class ModelRegistry:
    """单例 ModelRegistry：维护激活模型与热切换，避免重复加载。"""

    _instance = None
    _instance_lock = RLock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._lock = RLock()
        self._model_service: Optional[ModelService] = None
        self._model_repository: Optional[ModelRepository] = None
        self._active_model: Optional[Dict[str, object]] = None
        self._loaded_cache: Dict[str, Dict[str, object]] = {}
        self._initialized = True

    def configure(self, model_service: ModelService, model_repository: ModelRepository) -> None:
        with self._lock:
            self._model_service = model_service
            self._model_repository = model_repository

    def initialize_from_db(self) -> None:
        if self._model_repository is None:
            return
        active = self._model_repository.get_active_model()
        if active is not None:
            self.activate(active, persist=False)

    def get_active_model(self) -> Optional[Dict[str, object]]:
        with self._lock:
            return dict(self._active_model) if self._active_model else None

    def activate(self, model_record: Dict[str, object], persist: bool = True) -> Dict[str, object]:
        if self._model_service is None or self._model_repository is None:
            raise RuntimeError("ModelRegistry 未配置 ModelService/ModelRepository")

        with self._lock:
            path = str(model_record["path"])
            framework = str(model_record["framework"]).lower()

            if framework not in ("pt", "onnx"):
                raise ValueError(f"不支持的模型框架: {framework}")

            current_runtime = self._model_service.get_runtime_info()
            if (
                current_runtime.get("weights_path") == path
                and current_runtime.get("model_name") == str(model_record["backbone"])
                and self._model_service.is_loaded()
            ):
                self._active_model = dict(model_record)
                self._loaded_cache[path] = dict(model_record)
                if persist:
                    self._model_repository.set_active_model(int(model_record["id"]))
                return dict(self._active_model)

            self._model_service.switch_model(
                model_name=str(model_record["backbone"]),
                weights_path=path,
                img_size=self._model_service.config.img_size,
                device=self._model_service.config.device,
            )
            self._active_model = dict(model_record)
            self._loaded_cache[path] = dict(model_record)
            if persist:
                self._model_repository.set_active_model(int(model_record["id"]))
            return dict(self._active_model)
