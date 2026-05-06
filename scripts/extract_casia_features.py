"""
从CASIA-WebFace数据集提取特征并构建SQLite数据库

文件夹结构：
F:\Dataset\CASIA-WebFace\
├── person_1/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── person_2/
├── ...
└── person_200/

使用方式：
    python extract_casia_features.py --dataset-path F:\Dataset\CASIA-WebFace --num-ids 200 --output benchmark/CASIA_200_features.db
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.recognition_system.core.model import FaceEmbeddingModel
from apps.recognition_system.core.detector import FaceDetector


def init_database(db_path: str):
    """初始化SQLite数据库"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 创建person表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS person (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )
    """)

    # 创建embedding表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embedding (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            feature BLOB NOT NULL,
            image_path TEXT,
            FOREIGN KEY (person_id) REFERENCES person(id)
        )
    """)

    conn.commit()
    conn.close()


def extract_features_from_casia(
    dataset_path: str,
    output_db: str,
    num_ids: int = 200,
    model_weights: str = "weights/model_best.pt",
    device: str = "cuda:0",
):
    """
    从CASIA-WebFace提取特征

    Args:
        dataset_path: CASIA-WebFace数据集根目录
        output_db: 输出数据库路径
        num_ids: 要处理的ID个数（前num_ids个）
        model_weights: 模型权重路径
        device: 设备（cuda:0或cpu）
    """
    dataset_path = Path(dataset_path)
    output_db = Path(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("从CASIA-WebFace提取特征")
    print("=" * 80)

    # 1. 初始化数据库
    print(f"\n1️⃣  初始化数据库: {output_db}")
    init_database(str(output_db))

    # 2. 初始化模型
    print(f"\n2️⃣  初始化模型和检测器...")
    model = FaceEmbeddingModel(
        weights_path=model_weights,
        model_name="iresnet50",
        img_size=112,
        device=device,
    )
    detector = FaceDetector(
        img_size=112,
        conf_threshold=0.9,
        min_size=40,
        backend="mtcnn",
        device=device,
    )
    print(f"   ✓ 模型已加载到: {device}")

    # 3. 扫描CASIA目录
    print(f"\n3️⃣  扫描CASIA目录: {dataset_path}")
    person_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"   ✓ 找到 {len(person_dirs)} 个身份文件夹")
    print(f"   ✓ 只处理前 {num_ids} 个")

    person_dirs = person_dirs[:num_ids]

    # 4. 连接数据库
    conn = sqlite3.connect(str(output_db))
    cur = conn.cursor()

    # 5. 逐个身份处理
    print(f"\n4️⃣  提取特征...")
    total_images = 0
    total_features = 0
    failed_images = 0

    for person_idx, person_dir in enumerate(person_dirs, 1):
        person_name = person_dir.name

        if person_idx % max(1, num_ids // 10) == 0:
            print(f"   ... [{person_idx}/{num_ids}] {person_name}")

        # 插入person
        try:
            cur.execute("INSERT INTO person (name) VALUES (?)", (person_name,))
            person_id = cur.lastrowid
        except sqlite3.IntegrityError:
            cur.execute("SELECT id FROM person WHERE name = ?", (person_name,))
            person_id = cur.fetchone()[0]

        # 获取这个人的所有图片
        image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

        for image_path in image_files:
            total_images += 1

            try:
                # 读取图片
                image = cv2.imread(str(image_path))
                if image is None:
                    failed_images += 1
                    continue

                # 检测人脸
                boxes = detector.detect(image)
                if len(boxes) == 0:
                    failed_images += 1
                    continue

                # 使用最大的人脸
                box = max(boxes, key=lambda b: b[2] * b[3])
                face_rgb = detector.crop_face(image, box)

                # 提取特征
                feature = model.embed(face_rgb)

                # 存储到数据库
                cur.execute(
                    "INSERT INTO embedding (person_id, feature, image_path) VALUES (?, ?, ?)",
                    (person_id, feature.tobytes(), str(image_path)),
                )
                total_features += 1

            except Exception as e:
                failed_images += 1
                continue

    conn.commit()
    conn.close()

    # 6. 统计结果
    print(f"\n5️⃣  统计结果...")
    conn = sqlite3.connect(str(output_db))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM person")
    num_persons = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM embedding")
    num_embeddings = cur.fetchone()[0]

    conn.close()

    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print(f"\n📊 统计信息:")
    print(f"   处理的身份: {num_persons}")
    print(f"   扫描的图片: {total_images}")
    print(f"   提取的特征: {num_embeddings}")
    print(f"   失败的图片: {failed_images}")
    if total_images > 0:
        print(f"   成功率: {100 * num_embeddings / total_images:.1f}%")

    print(f"\n💾 数据库保存到: {output_db}")
    print(f"\n🚀 下一步:")
    print(f"   python prepare_open_set_optimized.py --db casia --input {output_db}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="从CASIA-WebFace提取特征并构建数据库"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="CASIA-WebFace数据集路径 (e.g., F:\\Dataset\\CASIA-WebFace)",
    )
    parser.add_argument(
        "--num-ids",
        type=int,
        default=200,
        help="要处理的身份数量（默认200）",
    )
    parser.add_argument(
        "--output",
        default="benchmark/CASIA_200_features.db",
        help="输出数据库路径",
    )
    parser.add_argument(
        "--model-weights",
        default="weights/model_best.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="设备 (cuda:0 或 cpu)",
    )

    args = parser.parse_args()

    extract_features_from_casia(
        dataset_path=args.dataset_path,
        output_db=args.output,
        num_ids=args.num_ids,
        model_weights=args.model_weights,
        device=args.device,
    )


if __name__ == "__main__":
    main()
