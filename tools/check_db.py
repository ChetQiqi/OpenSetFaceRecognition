#!/usr/bin/env python3
"""快速查看人脸库信息"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apps.recognition_system.core.feature_db import FeatureDB

def check_database(db_path):
    """检查数据库统计信息"""
    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}")
        return

    with FeatureDB(db_path) as db:
        stats = db.get_stats()
        persons = db.list_persons()

        print(f"\n📊 人脸库统计 ({db_path})")
        print("=" * 60)
        print(f"总人数:  {stats['person_count']}")
        print(f"总人脸数: {stats['embedding_count']}")
        print()

        if persons:
            print(f"👥 人员列表:")
            print("-" * 60)
            print(f"{'人名':<30} {'人脸数':>10}")
            print("-" * 60)
            for name, count in persons:
                print(f"{name:<30} {count:>10}")
            print("-" * 60)
        else:
            print("⚠️  数据库为空")
        print()

if __name__ == "__main__":
    # 检查 Streamlit 使用的数据库
    check_database("benchmark\\YTF_100p.db")

    # 也可以检查其他数据库
    # check_database("face_runtime/db/casia_10p.db")
