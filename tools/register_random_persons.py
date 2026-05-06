#!/usr/bin/env python3
"""
从数据集中随机选择N个人进行注册
用法: python register_random_persons.py --dataset datasets/casia --num-persons 100 --db-path face_runtime/db/my_db.db
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List

from apps.recognition_system.core.feature_db import FeatureDB
from apps.recognition_system.core.operations import build_runtime, register_dataset


def get_all_persons(dataset_dir: Path) -> List[Path]:
    """获取数据集中所有人的目录"""
    persons = []
    for person_dir in dataset_dir.iterdir():
        if person_dir.is_dir():
            # 检查是否有图片文件
            image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + \
                         list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.bmp"))
            if len(image_files) > 0:
                persons.append(person_dir)
    return persons


def register_selected_persons(
    db_path: str,
    model,
    detector,
    selected_persons: List[Path],
    max_images_per_person: int = 0,
    clear_first: bool = False
):
    """注册选定的人员"""

    print("\n" + "="*70)
    print(f"📝 开始注册 {len(selected_persons)} 个人")
    print("="*70)

    with FeatureDB(db_path) as db:
        if clear_first:
            print("⚠️  清空现有数据库...")
            db.clear_all()

        total_saved = 0
        total_failed = 0
        success_count = 0

        for idx, person_dir in enumerate(selected_persons, 1):
            person_name = person_dir.name
            print(f"\n[{idx}/{len(selected_persons)}] 处理: {person_name}")

            # 为每个人创建一个临时目录结构供 register_dataset 使用
            from apps.recognition_system.core.operations import iter_images, register_image

            saved = 0
            failed = 0
            processed = 0

            db.begin_transaction()
            try:
                for image_path in iter_images(person_dir):
                    if max_images_per_person > 0 and processed >= max_images_per_person:
                        break
                    processed += 1

                    if register_image(db, model, detector, person_name, image_path):
                        saved += 1
                    else:
                        failed += 1

                db.commit_transaction()

                if saved > 0:
                    success_count += 1
                    total_saved += saved
                    total_failed += failed
                    print(f"  ✅ 成功: {saved}/{processed} 张图片")
                else:
                    print(f"  ❌ 失败: 无法提取特征")

            except Exception as e:
                db.rollback_transaction()
                print(f"  ❌ 错误: {e}")
                total_failed += processed

    print("\n" + "="*70)
    print("📊 注册完成")
    print("="*70)
    print(f"✅ 成功注册: {success_count}/{len(selected_persons)} 人")
    print(f"📸 总图片数: 成功 {total_saved} 张, 失败 {total_failed} 张")
    print(f"💾 数据库路径: {db_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='从数据集中随机选择N个人进行注册')

    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集目录路径 (例如: datasets/casia)')
    parser.add_argument('--num-persons', type=int, default=100,
                       help='要注册的人数 (默认: 100)')
    parser.add_argument('--db-path', type=str, default='face_runtime/db/random_100p.db',
                       help='数据库路径 (默认: face_runtime/db/random_100p.db)')

    # 模型参数
    parser.add_argument('--weights', type=str, default='weights/adasin_best.pt',
                       help='模型权重路径')
    parser.add_argument('--model-name', type=str, default='iresnet50',
                       help='模型名称')
    parser.add_argument('--img-size', type=int, default=112,
                       help='输入图像大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备: auto/cuda/cpu')

    # 注册参数
    parser.add_argument('--max-images-per-person', type=int, default=0,
                       help='每个人最多注册的图片数 (0表示全部, 默认: 0)')
    parser.add_argument('--clear-db', action='store_true',
                       help='清空数据库后再注册')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return 1

    print("\n" + "="*70)
    print("🚀 从数据集中随机选择人员进行注册")
    print("="*70)
    print(f"📁 数据集: {dataset_dir}")
    print(f"👥 目标人数: {args.num_persons}")
    print(f"💾 数据库: {args.db_path}")
    print(f"🎲 随机种子: {args.seed}")
    print("="*70)

    # 获取所有人
    print("\n⏳ 扫描数据集...")
    all_persons = get_all_persons(dataset_dir)
    print(f"✅ 找到 {len(all_persons)} 个人")

    if len(all_persons) == 0:
        print("❌ 数据集为空")
        return 1

    # 随机选择
    num_to_select = min(args.num_persons, len(all_persons))
    if num_to_select < args.num_persons:
        print(f"⚠️  数据集只有 {len(all_persons)} 人，将全部注册")

    selected_persons = random.sample(all_persons, num_to_select)

    print(f"\n✅ 随机选择了 {len(selected_persons)} 个人")
    print("\n前10个人:")
    for i, person in enumerate(selected_persons[:10], 1):
        print(f"  {i}. {person.name}")
    if len(selected_persons) > 10:
        print(f"  ... 还有 {len(selected_persons) - 10} 个人")

    # 确认
    if args.clear_db:
        print("\n⚠️  警告: 将清空现有数据库!")

    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消")
        return 0

    # 加载模型
    print("\n⏳ 加载模型...")
    try:
        model, detector = build_runtime(
            weights_path=args.weights,
            model_name=args.model_name,
            img_size=args.img_size,
            device=args.device
        )
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 1

    # 创建数据库目录
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 注册
    register_selected_persons(
        db_path=str(db_path),
        model=model,
        detector=detector,
        selected_persons=selected_persons,
        max_images_per_person=args.max_images_per_person,
        clear_first=args.clear_db
    )

    # 显示数据库统计
    with FeatureDB(str(db_path)) as db:
        stats = db.get_stats()
        persons = db.list_persons()

    print("\n" + "="*70)
    print("📊 数据库统计")
    print("="*70)
    print(f"总人数: {stats['person_count']}")
    print(f"总特征数: {stats['embedding_count']}")
    print(f"平均每人: {stats['embedding_count'] / stats['person_count']:.1f} 张图片")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
