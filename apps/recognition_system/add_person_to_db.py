#!/usr/bin/env python3
"""
简单的人脸库添加脚本
可以快速添加新的人到数据库
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

from .core.feature_db import FeatureDB
import argparse

# 尝试导入需要cv2的模块（仅在需要时导入）
def get_runtime_modules():
    """懒加载 cv2 相关模块"""
    try:
        from .core.operations import build_runtime, register_image, register_dataset
        return build_runtime, register_image, register_dataset
    except ImportError:
        return None, None, None


def add_single_person_with_image():
    """交互式添加单个人的单张图片"""
    print("\n" + "="*70)
    print("👤 添加单个人 - 单张图片模式")
    print("="*70)

    # 检查依赖
    build_runtime, register_image, _ = get_runtime_modules()
    if build_runtime is None:
        print("❌ 无法导入模型模块，需要安装 OpenCV (opencv-python)")
        print("   请运行: pip install opencv-python")
        return False

    # 获取用户输入
    person_name = input("请输入人的名字（例如：张三）: ").strip()
    if not person_name:
        print("❌ 名字不能为空")
        return False

    image_path = input("请输入图片文件路径: ").strip()
    if not Path(image_path).exists():
        print(f"❌ 图片文件不存在: {image_path}")
        return False

    print(f"\n📝 配置:")
    print(f"  人名: {person_name}")
    print(f"  图片: {image_path}")
    print(f"  数据库: face_runtime/db/casia_10p.db")

    db_path = "face_runtime/db/casia_10p.db"
    weights_path = "weights/model_step999.pt"

    # 检查模型和数据库
    if not Path(weights_path).exists():
        print(f"❌ 模型不存在: {weights_path}")
        return False

    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}")
        return False

    print("\n🔧 加载模型...")
    model, detector = build_runtime(
        weights_path=weights_path,
        model_name="iresnet50",
        img_size=112,
        device="auto",
    )
    print("✓ 模型加载成功")

    print("\n📤 注册人脸...")
    with FeatureDB(db_path) as db:
        success = register_image(
            db=db,
            model=model,
            detector=detector,
            person_name=person_name,
            image_path=Path(image_path),
        )

    if success:
        print(f"\n✅ 成功！已将 {person_name} 的人脸添加到数据库")
        return True
    else:
        print(f"\n❌ 失败！无法检测到人脸或添加失败")
        return False


def add_multiple_persons_from_folder():
    """交互式添加文件夹中的多个人"""
    print("\n" + "="*70)
    print("👥 添加多个人 - 文件夹模式")
    print("="*70)

    # 检查依赖
    _, _, register_dataset = get_runtime_modules()
    if register_dataset is None:
        print("❌ 无法导入模型模块，需要安装 OpenCV (opencv-python)")
        print("   请运行: pip install opencv-python")
        return False

    from .core.operations import build_runtime as _build_runtime

    folder_path = input("请输入包含人脸照片的文件夹路径: ").strip()
    if not Path(folder_path).exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return False

    print(f"\n📁 文件夹结构应该是:")
    print(f"  {folder_path}/")
    print(f"  ├── 张三/")
    print(f"  │   ├── 1.jpg")
    print(f"  │   ├── 2.jpg")
    print(f"  │   └── 3.jpg")
    print(f"  └── 李四/")
    print(f"      ├── 1.jpg")
    print(f"      └── 2.jpg")

    confirm = input("\n文件夹结构是否正确？(y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消")
        return False

    db_path = "face_runtime/db/casia_10p.db"
    weights_path = "weights/model_step999.pt"

    # 检查模型和数据库
    if not Path(weights_path).exists():
        print(f"❌ 模型不存在: {weights_path}")
        return False

    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}")
        return False

    print("\n🔧 加载模型...")
    model, detector = _build_runtime(
        weights_path=weights_path,
        model_name="iresnet50",
        img_size=112,
        device="auto",
    )
    print("✓ 模型加载成功")

    print("\n📤 注册人脸...")
    with FeatureDB(db_path) as db:
        results = register_dataset(
            db=db,
            model=model,
            detector=detector,
            dataset_dir=Path(folder_path),
            clear_first=False,
            max_images_per_person=0,
        )

    # 打印结果
    print("\n✅ 注册完成！")
    print("-" * 70)
    for item in results:
        status = "✓" if item["saved"] > 0 else "✗"
        print(f"{status} {item['person_name']}: "
              f"处理={item['processed']} "
              f"成功={item['saved']} "
              f"失败={item['failed']}")
    print("-" * 70)

    total_saved = sum(item["saved"] for item in results)
    print(f"\n📊 总结: 添加了 {len(results)} 个人，共 {total_saved} 张人脸")
    return True


def list_all_persons():
    """列出数据库中所有的人"""
    print("\n" + "="*70)
    print("📋 数据库中的所有人")
    print("="*70)

    db_path = "face_runtime/db/casia_10p.db"

    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}")
        return False

    with FeatureDB(db_path) as db:
        persons = db.list_persons()
        stats = db.get_stats()

    print(f"\n📊 数据库统计:")
    print(f"  总人数: {stats['person_count']}")
    print(f"  总人脸数: {stats['embedding_count']}")

    if persons:
        print(f"\n👥 所有人:")
        print("-" * 70)
        for name, count in sorted(persons, key=lambda x: x[0]):
            print(f"  {name:20s} : {count:3d} 张人脸")
        print("-" * 70)
    else:
        print("\n❌ 数据库为空")

    return True


def remove_person():
    """删除一个人及其所有人脸"""
    print("\n" + "="*70)
    print("🗑️  删除一个人")
    print("="*70)

    db_path = "face_runtime/db/casia_10p.db"

    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}")
        return False

    # 列出现有的人
    with FeatureDB(db_path) as db:
        persons = db.list_persons()

    if not persons:
        print("❌ 数据库为空")
        return False

    print("\n现有的人:")
    for i, (name, count) in enumerate(sorted(persons, key=lambda x: x[0]), 1):
        print(f"  {i}. {name} ({count} 张人脸)")

    person_name = input("\n请输入要删除的人名: ").strip()

    # 确认
    confirm = input(f"确定要删除 {person_name} 及其所有 {len([p for p in persons if p[0] == person_name])} 张人脸吗? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消")
        return False

    with FeatureDB(db_path) as db:
        deleted = db.delete_person(person_name)

    if deleted:
        print(f"✅ 已删除 {person_name}")
        return True
    else:
        print(f"❌ 删除失败，未找到 {person_name}")
        return False


def main():
    parser = argparse.ArgumentParser(description="人脸库管理工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 子命令
    subparsers.add_parser("add-single", help="添加单张照片")
    subparsers.add_parser("add-folder", help="添加文件夹中的照片")
    subparsers.add_parser("list", help="列出所有人")
    subparsers.add_parser("remove", help="删除一个人")

    args = parser.parse_args()

    if not args.command:
        # 交互式菜单
        while True:
            print("\n" + "="*70)
            print("🎯 人脸库管理工具")
            print("="*70)
            print("1. 添加单张照片 (add-single)")
            print("2. 添加文件夹中的照片 (add-folder)")
            print("3. 列出所有人 (list)")
            print("4. 删除一个人 (remove)")
            print("5. 退出 (exit)")
            print("="*70)

            choice = input("请选择操作 (1-5): ").strip()

            if choice == "1":
                add_single_person_with_image()
            elif choice == "2":
                add_multiple_persons_from_folder()
            elif choice == "3":
                list_all_persons()
            elif choice == "4":
                remove_person()
            elif choice == "5":
                print("👋 再见!")
                break
            else:
                print("❌ 无效的选择")

    elif args.command == "add-single":
        add_single_person_with_image()
    elif args.command == "add-folder":
        add_multiple_persons_from_folder()
    elif args.command == "list":
        list_all_persons()
    elif args.command == "remove":
        remove_person()


if __name__ == "__main__":
    main()
