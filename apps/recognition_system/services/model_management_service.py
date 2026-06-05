from pathlib import Path
from typing import Dict, List, Optional

from apps.recognition_system.models import ModelRegistry
from apps.recognition_system.repositories import ModelRepository


class ModelManagementService:
    """Service 层：模型上传、列表、激活、删除。"""

    def __init__(self, model_repository: ModelRepository, model_registry: ModelRegistry, project_root: Path):
        self.model_repository = model_repository
        self.model_registry = model_registry
        self.weights_dir = (project_root / "weights" / "managed").resolve()
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def upload_model(
        self,
        file_name: str,
        content: bytes,
        name: str,
        backbone: str,
        embedding_size: int = 512,
    ) -> Dict[str, object]:
        suffix = Path(file_name).suffix.lower()
        if suffix not in {".pt", ".onnx"}:
            raise ValueError("仅支持上传 .pt 或 .onnx 模型文件")

        framework = "pt" if suffix == ".pt" else "onnx"
        safe_name = name.strip() or Path(file_name).stem
        target_file = self.weights_dir / f"{safe_name}{suffix}"
        target_file.write_bytes(content)

        model = self.model_repository.create_model(
            name=safe_name,
            path=str(target_file),
            backbone=backbone,
            embedding_size=int(embedding_size),
            framework=framework,
        )

        if self.model_repository.get_active_model() is None and framework == "pt":
            self.model_registry.activate(model, persist=True)

        return model

    def list_models(self) -> List[Dict[str, object]]:
        return self.model_repository.list_models()

    def activate_model(self, model_id: int) -> Dict[str, object]:
        model = self.model_repository.get_model(model_id)
        if model is None:
            raise ValueError("模型不存在")
        activated = self.model_registry.activate(model, persist=True)
        return activated

    def delete_model(self, model_id: int) -> Dict[str, object]:
        model = self.model_repository.get_model(model_id)
        if model is None:
            return {"deleted": False, "reason": "模型不存在", "id": model_id}

        if bool(model.get("is_active", False)):
            raise ValueError("不能删除当前激活模型，请先切换到其他模型")

        path = Path(str(model["path"]))
        deleted = self.model_repository.delete_model(model_id)
        if deleted and path.exists():
            path.unlink(missing_ok=True)
        return {"deleted": deleted, "id": model_id}

    def get_active_model(self) -> Optional[Dict[str, object]]:
        return self.model_repository.get_active_model()
