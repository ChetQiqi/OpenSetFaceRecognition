from typing import Dict, List, Optional, Tuple

import numpy as np

from apps.recognition_system.core.feature_db import FeatureDB


class IdentityRepository:
    """Repository 层：统一封装人员和特征库的 SQLite 访问。"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def list_persons(self) -> List[Tuple[str, int, str, str]]:
        with FeatureDB(self.db_path) as db:
            return db.list_persons()

    def get_stats(self) -> Dict[str, int]:
        with FeatureDB(self.db_path) as db:
            return db.get_stats()

    def add_embedding(self, person_name: str, feature: np.ndarray, image_path: str = "",
                      gender: str = "unspecified", birth_date: str = "") -> None:
        with FeatureDB(self.db_path) as db:
            db.add_embedding(person_name, feature, image_path=image_path,
                             gender=gender, birth_date=birth_date)

    def delete_person(self, person_name: str) -> bool:
        with FeatureDB(self.db_path) as db:
            return db.delete_person(person_name)

    def rename_person(self, old_name: str, new_name: str) -> bool:
        with FeatureDB(self.db_path) as db:
            return db.rename_person(old_name, new_name)

    def clear_person_embeddings(self, person_name: str) -> None:
        with FeatureDB(self.db_path) as db:
            db.clear_person_embeddings(person_name)

    def count_embeddings(self, person_name: Optional[str] = None) -> int:
        with FeatureDB(self.db_path) as db:
            return db.count_embeddings(person_name)

    def load_gallery(self, mode: str = "mean"):
        with FeatureDB(self.db_path) as db:
            return db.load_gallery(mode=mode)

    def get_person_latest_image_path(self, person_name: str) -> Optional[str]:
        with FeatureDB(self.db_path) as db:
            return db.get_person_latest_image_path(person_name)

    def list_person_image_paths(self, person_name: str) -> List[str]:
        with FeatureDB(self.db_path) as db:
            return db.list_person_image_paths(person_name)

    def get_person_detail(self, person_name: str) -> Optional[Tuple[str, int, str, str]]:
        with FeatureDB(self.db_path) as db:
            return db.get_person_detail(person_name)

    def update_person(self, old_name: str, new_name: str = "",
                      gender: str = "", birth_date: str = "") -> bool:
        with FeatureDB(self.db_path) as db:
            return db.update_person(old_name, new_name=new_name,
                                    gender=gender, birth_date=birth_date)

    def log_unknown_detection(self, **kwargs) -> None:
        with FeatureDB(self.db_path) as db:
            db.log_unknown_detection(**kwargs)
