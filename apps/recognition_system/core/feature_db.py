import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def _to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class FeatureDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._in_transaction = False
        self._init_tables()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _init_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS person (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                feature BLOB NOT NULL,
                image_path TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(person_id) REFERENCES person(id)
            );
            """
        )
        self._ensure_column("person", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._ensure_column("embedding", "image_path", "TEXT DEFAULT ''")
        self._ensure_column("person", "gender", "TEXT DEFAULT 'unspecified'")
        self._ensure_column("person", "birth_date", "TEXT DEFAULT ''")

        # Adaptive Open-set Recognition Framework tables
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL UNIQUE,
                mean_genuine_score REAL DEFAULT 0.0,
                std_genuine_score REAL DEFAULT 0.0,
                min_genuine_score REAL DEFAULT 0.0,
                max_genuine_score REAL DEFAULT 1.0,
                sample_count INTEGER DEFAULT 0,
                adaptive_threshold REAL DEFAULT 0.5,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(person_id) REFERENCES person(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                similarity_score REAL NOT NULL,
                is_genuine BOOLEAN DEFAULT 1,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(person_id) REFERENCES person(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS unknown_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                top_match_name TEXT DEFAULT '',
                top_match_score REAL DEFAULT 0.0,
                rejection_reason TEXT DEFAULT '',
                z_score REAL DEFAULT 0.0,
                distance_ratio REAL DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, column_def: str) -> None:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cur.fetchall()}
        if column not in columns:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")

    def close(self) -> None:
        self.conn.close()

    def begin_transaction(self) -> None:
        if self._in_transaction:
            return
        self.conn.execute("BEGIN")
        self._in_transaction = True

    def commit_transaction(self) -> None:
        if not self._in_transaction:
            return
        self.conn.commit()
        self._in_transaction = False

    def rollback_transaction(self) -> None:
        if not self._in_transaction:
            return
        self.conn.rollback()
        self._in_transaction = False

    def _get_or_create_person_id(self, name: str, gender: str = "unspecified", birth_date: str = "") -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM person WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is not None:
            return int(row[0])
        cur.execute(
            "INSERT INTO person(name, gender, birth_date) VALUES (?, ?, ?)",
            (name, gender, birth_date),
        )
        if not self._in_transaction:
            self.conn.commit()
        return int(cur.lastrowid)

    def clear_person_embeddings(self, name: str) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM person WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            return
        person_id = int(row[0])
        cur.execute("DELETE FROM embedding WHERE person_id = ?", (person_id,))
        if not self._in_transaction:
            self.conn.commit()

    def delete_person(self, name: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM person WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            return False
        person_id = int(row[0])
        cur.execute("DELETE FROM embedding WHERE person_id = ?", (person_id,))
        cur.execute("DELETE FROM person WHERE id = ?", (person_id,))
        if not self._in_transaction:
            self.conn.commit()
        return True

    def rename_person(self, old_name: str, new_name: str) -> bool:
        """重命名人员（修改人员ID/名字）"""
        cur = self.conn.cursor()

        # 检查旧名字是否存在
        cur.execute("SELECT id FROM person WHERE name = ?", (old_name,))
        row = cur.fetchone()
        if row is None:
            return False

        # 检查新名字是否已被使用
        cur.execute("SELECT id FROM person WHERE name = ?", (new_name,))
        if cur.fetchone() is not None:
            raise ValueError(f"人员 '{new_name}' 已存在！")

        # 更新人员名字
        cur.execute("UPDATE person SET name = ? WHERE name = ?", (new_name, old_name))
        if not self._in_transaction:
            self.conn.commit()
        return True

    def clear_all(self) -> None:
        """清空所有人员和特征数据"""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM embedding")
        cur.execute("DELETE FROM person")
        if not self._in_transaction:
            self.conn.commit()

    def add_embedding(self, name: str, feature: np.ndarray, image_path: str = "",
                      gender: str = "unspecified", birth_date: str = "") -> None:
        person_id = self._get_or_create_person_id(name, gender=gender, birth_date=birth_date)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO embedding(person_id, feature, image_path) VALUES (?, ?, ?)",
            (person_id, _to_blob(feature), image_path),
        )
        if not self._in_transaction:
            self.conn.commit()

    def load_all(self) -> List[Tuple[str, np.ndarray]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT p.name, e.feature
            FROM embedding e
            JOIN person p ON e.person_id = p.id
            """
        )
        rows = cur.fetchall()
        return [(name, _from_blob(blob)) for name, blob in rows]

    def get_person_latest_image_path(self, name: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT e.image_path
            FROM embedding e
            JOIN person p ON e.person_id = p.id
            WHERE p.name = ? AND COALESCE(e.image_path, '') != ''
            ORDER BY e.id DESC
            LIMIT 1
            """,
            (name,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return str(row[0]) if row[0] else None

    def list_person_image_paths(self, name: str) -> List[str]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT e.image_path
            FROM embedding e
            JOIN person p ON e.person_id = p.id
            WHERE p.name = ? AND COALESCE(e.image_path, '') != ''
            ORDER BY e.id DESC
            """,
            (name,),
        )
        return [str(row[0]) for row in cur.fetchall() if row and row[0]]

    def load_gallery(self, mode: str = "mean") -> List[Tuple[str, np.ndarray]]:
        rows = self.load_all()
        if mode == "all":
            return rows
        if mode != "mean":
            raise ValueError(f"Unsupported gallery mode: {mode}")

        grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
        for name, feature in rows:
            grouped[name].append(feature)

        gallery = []
        for name, features in grouped.items():
            prototype = np.mean(np.stack(features, axis=0), axis=0).astype(np.float32)
            gallery.append((name, prototype))
        return gallery

    def list_persons(self) -> List[Tuple[str, int, str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT p.name, COUNT(e.id) AS embedding_count,
                   COALESCE(p.gender, 'unspecified'), COALESCE(p.birth_date, '')
            FROM person p
            LEFT JOIN embedding e ON e.person_id = p.id
            GROUP BY p.id, p.name
            ORDER BY p.name ASC
            """
        )
        return [(str(name), int(count), str(gender), str(birth_date))
                for name, count, gender, birth_date in cur.fetchall()]

    def get_person_detail(self, name: str) -> Optional[Tuple[str, int, str, str]]:
        """Get full person detail by name."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT p.name, COUNT(e.id) AS embedding_count,
                   COALESCE(p.gender, 'unspecified'), COALESCE(p.birth_date, '')
            FROM person p
            LEFT JOIN embedding e ON e.person_id = p.id
            WHERE p.name = ?
            GROUP BY p.id, p.name
            """,
            (name,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return (str(row[0]), int(row[1]), str(row[2]), str(row[3]))

    def update_person(self, old_name: str, new_name: str = "",
                      gender: str = "", birth_date: str = "") -> bool:
        """Update person metadata. Returns True if person was found and updated."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, gender, birth_date FROM person WHERE name = ?", (old_name,))
        row = cur.fetchone()
        if row is None:
            return False

        person_id = int(row[0])
        final_name = new_name if new_name else str(row[1])
        final_gender = gender if gender else str(row[2] or "unspecified")
        final_birth_date = birth_date if birth_date else str(row[3] or "")

        if new_name and new_name != old_name:
            cur.execute("SELECT id FROM person WHERE name = ?", (new_name,))
            if cur.fetchone() is not None:
                raise ValueError(f"Person '{new_name}' already exists!")

        cur.execute(
            "UPDATE person SET name = ?, gender = ?, birth_date = ? WHERE id = ?",
            (final_name, final_gender, final_birth_date, person_id),
        )
        if not self._in_transaction:
            self.conn.commit()
        return True

    def get_stats(self) -> Dict[str, int]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM person")
        person_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM embedding")
        embedding_count = int(cur.fetchone()[0])
        return {
            "person_count": person_count,
            "embedding_count": embedding_count,
        }

    def count_embeddings(self, name: Optional[str] = None) -> int:
        cur = self.conn.cursor()
        if name is None:
            cur.execute("SELECT COUNT(*) FROM embedding")
            return int(cur.fetchone()[0])

        cur.execute(
            """
            SELECT COUNT(*)
            FROM embedding e
            JOIN person p ON e.person_id = p.id
            WHERE p.name = ?
            """,
            (name,),
        )
        return int(cur.fetchone()[0])

    # ========== Adaptive Framework Methods ==========

    def save_identity_statistics(self, stats: "IdentityStatistics") -> None:
        """
        Save or update identity statistics for adaptive thresholding.

        Args:
            stats: IdentityStatistics object with per-identity statistics
        """
        cur = self.conn.cursor()

        # Upsert (INSERT OR REPLACE)
        cur.execute(
            """
            INSERT OR REPLACE INTO identity_statistics (
                person_id, mean_genuine_score, std_genuine_score,
                min_genuine_score, max_genuine_score, sample_count,
                adaptive_threshold, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                stats.person_id,
                stats.mean_genuine_score,
                stats.std_genuine_score,
                stats.min_genuine_score,
                stats.max_genuine_score,
                stats.sample_count,
                stats.adaptive_threshold,
            ),
        )

        if not self._in_transaction:
            self.conn.commit()

    def load_identity_statistics(self) -> Dict[str, "IdentityStatistics"]:
        """
        Load all identity statistics from database.

        Returns:
            Dictionary mapping person_name to IdentityStatistics
        """
        from collections import deque

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                p.name, p.id, s.mean_genuine_score, s.std_genuine_score,
                s.min_genuine_score, s.max_genuine_score, s.sample_count,
                s.adaptive_threshold, s.last_updated
            FROM identity_statistics s
            JOIN person p ON s.person_id = p.id
            """
        )

        # Avoid circular import by using lazy import
        from .adaptive_threshold import IdentityStatistics
        import time

        identity_stats = {}
        for row in cur.fetchall():
            (
                name,
                person_id,
                mean_score,
                std_score,
                min_score,
                max_score,
                sample_count,
                threshold,
                last_updated_str,
            ) = row

            # Convert timestamp string to float (Unix time)
            try:
                last_updated = time.mktime(
                    time.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
                )
            except (ValueError, TypeError):
                last_updated = time.time()

            identity_stats[name] = IdentityStatistics(
                identity_name=name,
                person_id=person_id,
                mean_genuine_score=mean_score,
                std_genuine_score=std_score,
                min_genuine_score=min_score,
                max_genuine_score=max_score,
                sample_count=sample_count,
                adaptive_threshold=threshold,
                last_updated=last_updated,
                recent_scores=deque(maxlen=50),  # Empty for now, will be populated later
            )

        return identity_stats

    def log_recognition_history(
        self, person_id: int, similarity_score: float, is_genuine: bool = True
    ) -> None:
        """
        Log recognition event for temporal adaptation.

        Args:
            person_id: ID of recognized person
            similarity_score: Raw similarity score
            is_genuine: Whether this is a genuine match (default True)
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO recognition_history (person_id, similarity_score, is_genuine)
            VALUES (?, ?, ?)
            """,
            (person_id, similarity_score, 1 if is_genuine else 0),
        )

        if not self._in_transaction:
            self.conn.commit()

    def log_unknown_detection(
        self,
        top_match_name: str,
        top_match_score: float,
        rejection_reason: str,
        z_score: float,
        distance_ratio: float,
    ) -> None:
        """
        Log rejected unknown detection for analysis.

        Args:
            top_match_name: Name of closest match (even though rejected)
            top_match_score: Similarity score of closest match
            rejection_reason: Reason for rejection ("outlier", "below_threshold", etc.)
            z_score: Z-score metric
            distance_ratio: Distance ratio metric
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO unknown_detections (
                top_match_name, top_match_score, rejection_reason,
                z_score, distance_ratio
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (top_match_name, top_match_score, rejection_reason, z_score, distance_ratio),
        )

        if not self._in_transaction:
            self.conn.commit()

    def get_all_persons(self) -> List[Tuple[int, str]]:
        """
        Get all persons in database.

        Returns:
            List of (person_id, person_name) tuples
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, name FROM person ORDER BY name ASC")
        return [(int(pid), str(name)) for pid, name in cur.fetchall()]

