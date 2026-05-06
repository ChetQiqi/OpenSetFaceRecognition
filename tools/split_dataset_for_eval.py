#!/usr/bin/env python3
"""
从已注册数据集中分离出测试集
用法: python split_dataset_for_eval.py --source datasets/casia_10p --output datasets/casia_10p_test --ratio 0.3
"""

import argparse
import shutil
from pathlib import Path
import random


def split_dataset(source_dir, output_dir, ratio=0.3, min_train=2):
    """
    将数据集按人员分割：部分用于注册，部分用于测试

    Args:
        source_dir: 原始数据集目录
        output_dir: 测试集输出目录
        ratio: 测试集比例（0-1）
        min_train: 每人最少保留用于注册的图片数
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_persons': 0,
        'total_images': 0,
        'test_images': 0,
        'train_images': 0
    }

    for person_dir in sorted(source_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        images = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))

        if len(images) <= min_train:
            print(f"⚠️  {person_name}: 只有 {len(images)} 张图片，跳过（需要至少 {min_train+1} 张）")
            continue

        # 计算测试集数量
        n_test = max(1, int(len(images) * ratio))
        n_test = min(n_test, len(images) - min_train)  # 确保训练集至少有min_train张

        # 随机选择测试集
        random.shuffle(images)
        test_images = images[:n_test]
        train_images = images[n_test:]

        # 复制测试集图片
        test_person_dir = output_dir / person_name
        test_person_dir.mkdir(parents=True, exist_ok=True)

        for img in test_images:
            shutil.copy2(img, test_person_dir / img.name)

        stats['total_persons'] += 1
        stats['total_images'] += len(images)
        stats['test_images'] += len(test_images)
        stats['train_images'] += len(train_images)

        print(f"✅ {person_name}: {len(images)} 张 → 训练集 {len(train_images)} 张, 测试集 {len(test_images)} 张")

    print("\n" + "="*60)
    print("数据集分割完成")
    print("="*60)
    print(f"总人数: {stats['total_persons']}")
    print(f"总图片: {stats['total_images']}")
    print(f"训练集: {stats['train_images']} ({stats['train_images']/stats['total_images']*100:.1f}%)")
    print(f"测试集: {stats['test_images']} ({stats['test_images']/stats['total_images']*100:.1f}%)")
    print(f"\n测试集保存在: {output_dir}")
    print("\n下一步:")
    print(f"1. 检查数据库是否使用训练集注册")
    print(f"2. 运行评估: python apps/recognition_system/core/eval_comprehensive.py \\")
    print(f"     --weights weights/adasin_best.pt \\")
    print(f"     --db-path face_runtime/db/casia_10p.db \\")
    print(f"     --test-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser("Split dataset into train and test sets")
    parser.add_argument("--source", required=True, help="Source dataset directory")
    parser.add_argument("--output", required=True, help="Output test set directory")
    parser.add_argument("--ratio", type=float, default=0.3, help="Test set ratio (default: 0.3)")
    parser.add_argument("--min-train", type=int, default=2, help="Minimum training images per person")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    split_dataset(args.source, args.output, args.ratio, args.min_train)


if __name__ == "__main__":
    main()
