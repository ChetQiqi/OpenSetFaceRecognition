from typing import Dict, List

from apps.recognition_system.models import ModelService
from apps.recognition_system.repositories import IdentityRepository


class DeveloperService:
    """开发者功能：benchmark 查看和轻量级模型评估。"""

    def __init__(self, repository: IdentityRepository, model_service: ModelService):
        self.repository = repository
        self.model_service = model_service

    def benchmark_summary(self) -> Dict[str, object]:
        stats = self.repository.get_stats()
        persons = self.repository.list_persons()
        top_persons = [
            {"name": name, "embedding_count": count}
            for name, count in sorted(persons, key=lambda x: x[1], reverse=True)[:10]
        ]
        return {
            "database_path": self.repository.db_path,
            "person_count": stats["person_count"],
            "embedding_count": stats["embedding_count"],
            "top_persons": top_persons,
        }

    def model_eval(self, topn: int = 10) -> Dict[str, object]:
        self.model_service.load()
        runtime = self.model_service.get_runtime_info()
        persons: List[tuple[str, int]] = self.repository.list_persons()
        sorted_persons = sorted(persons, key=lambda x: x[1], reverse=True)
        heavy_classes = [
            {"name": name, "embedding_count": count}
            for name, count in sorted_persons[: max(1, int(topn))]
        ]
        sparse_classes = [
            {"name": name, "embedding_count": count}
            for name, count in sorted(persons, key=lambda x: x[1])[: max(1, int(topn))]
        ]
        return {
            "runtime": runtime,
            "gallery_size": len(self.model_service.gallery),
            "class_balance_top": heavy_classes,
            "class_balance_bottom": sparse_classes,
            "recommendation": "建议优先补齐 embedding_count 过低的身份样本以提升识别稳定性",
        }
