"""
准备Open-Set评估数据集 - 支持多个数据库

使用方式：
  # YTF小数据库（103 persons，快速）
  python prepare_open_set_optimized.py --db small

  # YTF完整数据库（1595 persons，论文用）
  python prepare_open_set_optimized.py --db full --num-known 400

  # CASIA数据库（200 persons，新数据集）
  python prepare_open_set_optimized.py --db casia

  # 自定义数据库
  python prepare_open_set_optimized.py --input benchmark/CASIA_200_features.db --num-known 100
"""
import argparse
import json
import random
import sqlite3
from pathlib import Path
import sys
import numpy as np

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def prepare_open_set(
    db_path: str,
    output_dir: str = "benchmark/open_set_data",
    num_known: int = 80,
    gallery_size: int = 3,
    test_size: int = 2,
    seed: int = 42,
):
    """
    高效准备Open-Set数据集

    Args:
        db_path: 数据库路径
        output_dir: 输出目录
        num_known: known persons数量
        gallery_size: 每个known person用于gallery的特征数
        test_size: 每个known person用于test的特征数
        seed: 随机种子
    """
    print("=" * 80)
    print(f"准备Open-Set评估数据集")
    print("=" * 80)

    # 1. 连接数据库
    print(f"\n1️⃣  加载数据库: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 获取所有person (按ID排序)
    cur.execute("SELECT id, name FROM person ORDER BY id")
    all_persons = cur.fetchall()
    print(f"   ✓ 总计 {len(all_persons)} 个 persons")

    # 2. 分割known/unknown
    print(f"\n2️⃣  分割known/unknown...")
    random.seed(seed)
    indices = list(range(len(all_persons)))
    random.shuffle(indices)

    known_indices = indices[:num_known]
    unknown_indices = indices[num_known:]

    known_persons = [all_persons[i] for i in sorted(known_indices)]
    unknown_persons = [all_persons[i] for i in sorted(unknown_indices)]

    known_ids = set(pid for pid, _ in known_persons)
    unknown_ids = set(pid for pid, _ in unknown_persons)

    print(f"   ✓ Known: {len(known_persons)} persons")
    print(f"   ✓ Unknown: {len(unknown_persons)} persons")

    # 3. 准备Gallery和Test Known (只加载known person的数据)
    print(f"\n3️⃣  准备Gallery和Test Known...")
    gallery = []
    test_known_embs = []
    test_known_labels = []
    valid_known = 0

    for idx, (person_id, person_name) in enumerate(known_persons, 1):
        if idx % max(1, len(known_persons)//10) == 0:
            print(f"   ... {idx}/{len(known_persons)}")

        # 只加载这个person的embeddings
        cur.execute(
            "SELECT feature FROM embedding WHERE person_id = ? ORDER BY id",
            (person_id,)
        )
        rows = cur.fetchall()

        if len(rows) >= gallery_size + test_size:
            # Gallery: 前gallery_size个
            for i in range(gallery_size):
                emb_blob = rows[i][0]
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                gallery.append((person_name, emb))

            # Test Known: 后test_size个
            for i in range(gallery_size, gallery_size + test_size):
                emb_blob = rows[i][0]
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                test_known_embs.append(emb)
                test_known_labels.append(person_name)

            valid_known += 1

    test_known_embs = np.stack(test_known_embs, axis=0) if test_known_embs else np.array([])

    print(f"   ✓ Gallery: {len(gallery)} embeddings from {valid_known} persons")
    print(f"   ✓ Test Known: {len(test_known_labels)} samples")

    # 4. 准备Test Unknown (只加载unknown person的数据)
    print(f"\n4️⃣  准备Test Unknown...")
    test_unknown_embs = []
    test_unknown_labels = []

    for idx, (person_id, person_name) in enumerate(unknown_persons, 1):
        if idx % max(1, len(unknown_persons)//10) == 0:
            print(f"   ... {idx}/{len(unknown_persons)}")

        # 加载这个unknown person的所有embeddings
        cur.execute(
            "SELECT feature FROM embedding WHERE person_id = ? ORDER BY id",
            (person_id,)
        )
        rows = cur.fetchall()

        for emb_blob, in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            test_unknown_embs.append(emb)
            test_unknown_labels.append(person_name)

    test_unknown_embs = np.stack(test_unknown_embs, axis=0) if test_unknown_embs else np.array([])

    print(f"   ✓ Test Unknown: {len(test_unknown_labels)} samples")

    conn.close()

    # 5. 保存数据集
    print(f"\n5️⃣  保存数据集...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gallery
    gallery_embs = np.stack([emb for _, emb in gallery], axis=0)
    gallery_labels = [name for name, _ in gallery]

    np.save(output_dir / "gallery.npy", gallery_embs)
    with open(output_dir / "gallery_labels.txt", "w", encoding="utf-8") as f:
        for label in gallery_labels:
            f.write(f"{label}\n")

    # Test Known
    np.save(output_dir / "test_known.npy", test_known_embs)
    with open(output_dir / "test_known_labels.txt", "w", encoding="utf-8") as f:
        for label in test_known_labels:
            f.write(f"{label}\n")

    # Test Unknown
    np.save(output_dir / "test_unknown.npy", test_unknown_embs)
    with open(output_dir / "test_unknown_labels.txt", "w", encoding="utf-8") as f:
        for label in test_unknown_labels:
            f.write(f"{label}\n")

    # Config
    config = {
        "num_known": len(set(test_known_labels)),
        "num_unknown": len(set(test_unknown_labels)),
        "gallery_size": len(gallery_labels),
        "test_known_size": len(test_known_labels),
        "test_unknown_size": len(test_unknown_labels),
        "known_labels": sorted(set(test_known_labels)),
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"   ✓ 数据已保存到: {output_dir}")

    # 6. 总结
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print(f"\n📊 数据集统计:")
    print(f"   Gallery: {len(gallery_labels)} embeddings")
    print(f"   Test Known: {len(test_known_labels)} samples")
    print(f"   Test Unknown: {len(test_unknown_labels)} samples")
    print(f"   ---")
    print(f"   总计: {len(test_known_labels) + len(test_unknown_labels)} test samples")
    print(f"\n💾 输出目录: {output_dir}")
    print(f"\n🚀 下一步:")
    print(f"   python scripts/compare_with_prepared_data.py --data-dir {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="准备Open-Set评估数据集（支持YTF和CASIA数据库）"
    )
    parser.add_argument(
        "--db",
        choices=["small", "full", "casia"],
        default="small",
        help="选择数据库: small(YTF 103 persons), full(YTF 1595 persons), casia(CASIA)",
    )
    parser.add_argument(
        "--input",
        help="自定义数据库路径（如果指定，覆盖--db选择）",
    )
    parser.add_argument(
        "--num-known",
        type=int,
        default=None,
        help="Known persons数量 (默认: small=80, full=400, casia=100)",
    )
    parser.add_argument(
        "--gallery-size",
        type=int,
        help="Gallery特征数 (默认: small=3, full=25, casia=10)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        help="Test特征数 (默认: small=2, full=25, casia=10)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark/open_set_data",
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    args = parser.parse_args()

    # 根据数据库类型设置默认参数
    if args.input:
        # 使用自定义数据库路径
        db_path = args.input
        num_known = args.num_known if args.num_known else 100
        gallery_size = args.gallery_size if args.gallery_size else 10
        test_size = args.test_size if args.test_size else 10
        print(f"📚 使用自定义数据库: {db_path}")
    elif args.db == "small":
        db_path = "benchmark/YTF_100p.db"
        num_known = args.num_known if args.num_known else 80
        gallery_size = args.gallery_size if args.gallery_size else 3
        test_size = args.test_size if args.test_size else 2
        print("📚 使用小数据库: YTF_100p.db (103 persons)")
    elif args.db == "full":
        db_path = "benchmark/YTF_allID_50features.db"
        num_known = args.num_known if args.num_known else 400
        gallery_size = args.gallery_size if args.gallery_size else 25
        test_size = args.test_size if args.test_size else 25
        print("📚 使用完整数据库: YTF_allID_50features.db (1595 persons)")
    else:  # casia
        db_path = "benchmark/CASIA_200_features.db"
        num_known = args.num_known if args.num_known else 100
        gallery_size = args.gallery_size if args.gallery_size else 10
        test_size = args.test_size if args.test_size else 10
        print("📚 使用CASIA数据库: CASIA_200_features.db (200 persons)")

    prepare_open_set(
        db_path=db_path,
        output_dir=args.output_dir,
        num_known=num_known,
        gallery_size=gallery_size,
        test_size=test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
