"""
验证 YTF_100p.db 与实际人脸图片目录的对应关系
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

def check_db_image_correspondence():
    """检查数据库中的人物与实际图片目录的对应关系"""

    db_path = "benchmark/YTF_100p.db"
    image_root = Path(r"G:\YTF_dataset\OpenDataLab___YouTube_Faces\raw\data\YouTubeFaces\aligned_images_DB")

    print("=" * 80)
    print("📊 YTF_100p.db 与图片目录验证")
    print("=" * 80)

    # 1. 从数据库中获取所有人物
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM person ORDER BY id")
    db_persons = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM embedding")
    total_embeddings = cursor.fetchone()[0]

    cursor.execute("SELECT person_id, COUNT(*) FROM embedding GROUP BY person_id")
    embeddings_per_person = dict(cursor.fetchall())

    conn.close()

    print(f"\n【数据库信息】")
    print(f"  总人数: {len(db_persons)}")
    print(f"  总 embedding 数: {total_embeddings}")

    # 2. 检查图片目录
    image_dirs = [d for d in image_root.iterdir() if d.is_dir()]
    print(f"\n【图片目录信息】")
    print(f"  目录中的人物总数: {len(image_dirs)}")

    # 3. 映射关系检查
    print(f"\n【映射关系验证】")

    found_count = 0
    missing_in_db = []
    missing_in_images = []

    # 检查数据库中的人物是否有对应的图片
    for person_id, person_name in db_persons:
        person_dir = image_root / person_name
        embedding_count = embeddings_per_person.get(person_id, 0)

        if person_dir.exists():
            found_count += 1
            video_dirs = [d for d in person_dir.iterdir() if d.is_dir()]
            image_files = []
            for video_dir in video_dirs:
                image_files.extend(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))

            if person_id <= 5 or person_id % 20 == 0:  # 打印样本
                print(f"  ✓ {person_name:30s} | 视频: {len(video_dirs):2d} | 图片: {len(image_files):3d} | embedding: {embedding_count:3d}")
        else:
            missing_in_images.append((person_id, person_name))

    print(f"\n【统计结果】")
    print(f"  数据库中的人物在目录中找到: {found_count}/{len(db_persons)}")

    if missing_in_images:
        print(f"\n  ❌ 数据库中有但目录中缺失的人物 ({len(missing_in_images)}):")
        for person_id, person_name in missing_in_images[:10]:
            print(f"     - {person_name} (ID: {person_id})")
        if len(missing_in_images) > 10:
            print(f"     ... 还有 {len(missing_in_images) - 10} 个")

    # 检查目录中是否有未在数据库中的人物
    db_names = set(name for _, name in db_persons)
    for image_dir in image_dirs:
        if image_dir.name not in db_names:
            missing_in_db.append(image_dir.name)

    if missing_in_db:
        print(f"\n  ⚠️  目录中有但数据库中缺失的人物 ({len(missing_in_db)}):")
        for name in missing_in_db[:10]:
            print(f"     - {name}")
        if len(missing_in_db) > 10:
            print(f"     ... 还有 {len(missing_in_db) - 10} 个")

    # 4. 统计 embedding 分布
    print(f"\n【Embedding 分布统计】")
    embedding_counts = list(embeddings_per_person.values())
    print(f"  每人平均 embedding: {sum(embedding_counts) / len(embedding_counts):.2f}")
    print(f"  embedding 最多的人: {max(embedding_counts)} 个")
    print(f"  embedding 最少的人: {min(embedding_counts)} 个")

    # 5. 详细的对应关系表
    print(f"\n【详细对应关系】")
    print(f"{'ID':>3} {'人名':<30} {'Embedding数':>10} {'图片目录':>10}")
    print("-" * 60)

    for person_id, person_name in db_persons:
        person_dir = image_root / person_name
        embedding_count = embeddings_per_person.get(person_id, 0)
        exists_str = "✓" if person_dir.exists() else "✗"

        if person_id <= 10 or person_id % 10 == 0 or not person_dir.exists():
            print(f"{person_id:3d} {person_name:<30} {embedding_count:>10} {exists_str:>10}")

    print("\n" + "=" * 80)
    print(f"✅ 验证完成！")
    print(f"   实际可用于验证的人物: {found_count} 个")
    print(f"   总 embedding 数: {total_embeddings} 个")

if __name__ == "__main__":
    check_db_image_correspondence()
