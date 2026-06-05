import sqlite3
from typing import Dict, List, Optional


class ModelRepository:
    """Repository 层：模型元数据存储与激活状态管理。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_table()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    path TEXT NOT NULL,
                    backbone TEXT NOT NULL,
                    embedding_size INTEGER NOT NULL DEFAULT 512,
                    framework TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def create_model(
        self,
        name: str,
        path: str,
        backbone: str,
        embedding_size: int,
        framework: str,
    ) -> Dict[str, object]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO model_metadata (name, path, backbone, embedding_size, framework, is_active)
                    VALUES (?, ?, ?, ?, ?, 0)
                    """,
                    (name, path, backbone, int(embedding_size), framework),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"模型名已存在: {name}") from exc
            conn.commit()
            model_id = int(cur.lastrowid)
        finally:
            conn.close()
        model = self.get_model(model_id)
        if model is None:
            raise RuntimeError("模型创建后读取失败")
        return model

    def list_models(self) -> List[Dict[str, object]]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, name, path, backbone, embedding_size, framework, created_at, is_active
                FROM model_metadata
                ORDER BY id DESC
                """
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        return [
            {
                "id": int(row[0]),
                "name": str(row[1]),
                "path": str(row[2]),
                "backbone": str(row[3]),
                "embedding_size": int(row[4]),
                "framework": str(row[5]),
                "created_at": str(row[6]),
                "is_active": bool(row[7]),
            }
            for row in rows
        ]

    def get_model(self, model_id: int) -> Optional[Dict[str, object]]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, name, path, backbone, embedding_size, framework, created_at, is_active
                FROM model_metadata
                WHERE id = ?
                """,
                (int(model_id),),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return {
            "id": int(row[0]),
            "name": str(row[1]),
            "path": str(row[2]),
            "backbone": str(row[3]),
            "embedding_size": int(row[4]),
            "framework": str(row[5]),
            "created_at": str(row[6]),
            "is_active": bool(row[7]),
        }

    def get_active_model(self) -> Optional[Dict[str, object]]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, name, path, backbone, embedding_size, framework, created_at, is_active
                FROM model_metadata
                WHERE is_active = 1
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return {
            "id": int(row[0]),
            "name": str(row[1]),
            "path": str(row[2]),
            "backbone": str(row[3]),
            "embedding_size": int(row[4]),
            "framework": str(row[5]),
            "created_at": str(row[6]),
            "is_active": bool(row[7]),
        }

    def set_active_model(self, model_id: int) -> Optional[Dict[str, object]]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("UPDATE model_metadata SET is_active = 0 WHERE is_active = 1")
            cur.execute("UPDATE model_metadata SET is_active = 1 WHERE id = ?", (int(model_id),))
            if cur.rowcount == 0:
                conn.rollback()
                return None
            conn.commit()
        finally:
            conn.close()
        return self.get_model(model_id)

    def delete_model(self, model_id: int) -> bool:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM model_metadata WHERE id = ?", (int(model_id),))
            deleted = cur.rowcount > 0
            conn.commit()
        finally:
            conn.close()
        return deleted
