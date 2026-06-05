from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from apps.recognition_system.models import ModelService
from apps.recognition_system.repositories import IdentityRepository


class IdentityService:
    """Service 层：人员注册、删除、修改、查询。"""

    def __init__(self, repository: IdentityRepository, model_service: ModelService):
        self.repository = repository
        self.model_service = model_service

    def stats(self) -> Dict[str, int]:
        return self.repository.get_stats()

    def list_identities(self) -> List[Dict[str, object]]:
        return [
            {"name": name, "embedding_count": count, "gender": gender, "birth_date": birth_date}
            for name, count, gender, birth_date in self.repository.list_persons()
        ]

    def add_identity(self, person_name: str, images: Iterable[Tuple[str, bytes]],
                     gender: str = "unspecified", birth_date: str = "") -> Dict[str, object]:
        if not person_name.strip():
            raise ValueError("人员 ID 不能为空")

        success_files: List[str] = []
        failed_files: List[Dict[str, str]] = []

        person_name = person_name.strip()
        gallery_dir = Path(self.repository.db_path).resolve().parent / "gallery_images" / person_name
        gallery_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in images:
            image = self._decode_image(content)
            if image is None:
                failed_files.append({"filename": filename, "reason": "图片解码失败"})
                continue

            try:
                feature = self.model_service.extract_embedding(image)
            except Exception as exc:
                failed_files.append({"filename": filename, "reason": str(exc)})
                continue

            if feature is None:
                failed_files.append({"filename": filename, "reason": "未检测到人脸"})
                continue

            safe_name = Path(filename).name or "image.jpg"
            target_path = gallery_dir / safe_name
            stem = target_path.stem
            suffix = target_path.suffix or ".jpg"
            idx = 1
            while target_path.exists():
                target_path = gallery_dir / f"{stem}_{idx}{suffix}"
                idx += 1
            target_path.write_bytes(content)

            self.repository.add_embedding(person_name, feature, image_path=str(target_path),
                                              gender=gender, birth_date=birth_date)
            success_files.append(filename)

        if success_files:
            self.model_service.refresh_gallery()

        return {
            "person_name": person_name,
            "success_count": len(success_files),
            "fail_count": len(failed_files),
            "success_files": success_files,
            "failed_files": failed_files,
        }

    def delete_identity(self, person_name: str) -> Dict[str, object]:
        deleted = self.repository.delete_person(person_name)
        if deleted:
            self.model_service.refresh_gallery()
        return {"deleted": deleted, "person_name": person_name}

    def get_identity(self, person_name: str) -> Optional[Dict[str, object]]:
        detail = self.repository.get_person_detail(person_name)
        if detail is None:
            return None
        name, count, gender, birth_date = detail
        return {"name": name, "embedding_count": count, "gender": gender, "birth_date": birth_date}

    def update_identity(self, old_name: str, new_name: str = "",
                        gender: str = "", birth_date: str = "") -> Dict[str, object]:
        if not old_name:
            raise ValueError("人员名称不能为空")
        if new_name and not new_name.strip():
            raise ValueError("新人员名称不能为空")
        updated = self.repository.update_person(
            old_name, new_name=new_name, gender=gender, birth_date=birth_date
        )
        if updated and new_name:
            self.model_service.refresh_gallery()
        return {"updated": updated, "old_name": old_name, "new_name": new_name or old_name}

    def rename_identity(self, old_name: str, new_name: str) -> Dict[str, object]:
        if not new_name.strip():
            raise ValueError("新人员 ID 不能为空")
        renamed = self.repository.rename_person(old_name, new_name)
        if renamed:
            self.model_service.refresh_gallery()
        return {"renamed": renamed, "old_name": old_name, "new_name": new_name}

    @staticmethod
    def _decode_image(content: bytes):
        array = np.frombuffer(content, dtype=np.uint8)
        if array.size == 0:
            return None
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
