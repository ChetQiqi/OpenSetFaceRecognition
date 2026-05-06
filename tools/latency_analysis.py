#!/usr/bin/env python3
"""
人脸识别系统端到端延迟分析
分析各个模块的耗时，评估是否达到实时处理标准

测量内容：
1. 人脸检测（MTCNN）耗时
2. 特征提取（iResNet50）耗时
3. 底库检索（1:N）耗时
4. 总延迟（是否达到 30 FPS 标准）
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

import cv2
import numpy as np
from tqdm import tqdm

from apps.recognition_system.core.operations import (
    build_runtime, load_gallery, extract_face_embedding
)
from apps.recognition_system.core.feature_db import FeatureDB
from apps.recognition_system.core.matcher import find_best_match


class LatencyProfiler:
    """延迟分析器"""

    def __init__(self):
        self.timings = {
            'detection': [],      # 人脸检测耗时
            'extraction': [],     # 特征提取耗时
            'matching': [],       # 底库检索耗时
            'total': []          # 总耗时
        }

    def add_timing(self, stage: str, duration: float):
        """添加一次计时记录（毫秒）"""
        if stage in self.timings:
            self.timings[stage].append(duration * 1000)  # 转换为毫秒

    def get_stats(self, stage: str) -> Dict[str, float]:
        """获取某个阶段的统计信息"""
        if stage not in self.timings or not self.timings[stage]:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p95': 0, 'p99': 0}

        data = np.array(self.timings[stage])
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'p50': np.percentile(data, 50),
            'p95': np.percentile(data, 95),
            'p99': np.percentile(data, 99)
        }

    def print_report(self, db_size: int):
        """打印延迟分析报告"""
        print("\n" + "="*70)
        print("⏱️  端到端延迟分析报告")
        print("="*70)
        print(f"📊 测试样本数: {len(self.timings['total'])} 张图片")
        print(f"💾 底库规模: {db_size:,} 人")
        print("="*70)

        stages = [
            ('detection', '🔍 人脸检测 (MTCNN)'),
            ('extraction', '🧠 特征提取 (iResNet50)'),
            ('matching', '🔎 底库检索 (1:N)'),
            ('total', '⚡ 总延迟')
        ]

        print(f"\n{'阶段':<25} {'平均':<10} {'标准差':<10} {'P50':<10} {'P95':<10} {'P99':<10}")
        print("-"*70)

        for stage, name in stages:
            stats = self.get_stats(stage)
            print(f"{name:<25} {stats['mean']:>7.2f} ms {stats['std']:>7.2f} ms "
                  f"{stats['p50']:>7.2f} ms {stats['p95']:>7.2f} ms {stats['p99']:>7.2f} ms")

        print("="*70)

        # 计算FPS和实时性能
        total_stats = self.get_stats('total')
        avg_latency = total_stats['mean']
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0

        print(f"\n🎯 性能评估:")
        print(f"   平均端到端延迟: {avg_latency:.2f} ms")
        print(f"   理论处理帧率: {fps:.2f} FPS")

        # 判断是否达到实时标准
        if fps >= 30:
            print(f"   ✅ 达到 30 FPS 实时标准 (实际: {fps:.2f} FPS)")
        else:
            print(f"   ❌ 未达到 30 FPS 实时标准 (实际: {fps:.2f} FPS, 差距: {30-fps:.2f} FPS)")

        if fps >= 25:
            print(f"   ✅ 满足视频处理要求 (25 FPS)")

        # 各阶段耗时占比
        print(f"\n📊 耗时占比:")
        detection_stats = self.get_stats('detection')
        extraction_stats = self.get_stats('extraction')
        matching_stats = self.get_stats('matching')

        total = detection_stats['mean'] + extraction_stats['mean'] + matching_stats['mean']
        if total > 0:
            print(f"   人脸检测: {detection_stats['mean']:>7.2f} ms ({detection_stats['mean']/total*100:>5.1f}%)")
            print(f"   特征提取: {extraction_stats['mean']:>7.2f} ms ({extraction_stats['mean']/total*100:>5.1f}%)")
            print(f"   底库检索: {matching_stats['mean']:>7.2f} ms ({matching_stats['mean']/total*100:>5.1f}%)")

        print("="*70 + "\n")

    def save_to_json(self, output_path: str, db_size: int):
        """保存结果到JSON文件"""
        result = {
            'database_size': db_size,
            'sample_count': len(self.timings['total']),
            'stages': {}
        }

        for stage in self.timings.keys():
            result['stages'][stage] = self.get_stats(stage)

        # 计算FPS
        total_stats = self.get_stats('total')
        result['fps'] = float(1000.0 / total_stats['mean'] if total_stats['mean'] > 0 else 0)
        result['realtime_30fps'] = bool(result['fps'] >= 30)
        result['video_25fps'] = bool(result['fps'] >= 25)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def analyze_from_images(images_dir: Path, model, detector, gallery: Dict,
                       profiler: LatencyProfiler, max_samples: int = -1, skip_detection: bool = False):
    """从图像目录分析延迟"""

    print("\n⏳ 正在分析图像...")

    # 收集所有测试图像
    test_images = []
    for person_dir in images_dir.iterdir():
        if not person_dir.is_dir():
            continue

        # 支持两种目录结构:
        # 1. PersonID/VideoNumber/*.jpg (嵌套)
        # 2. PersonID/*.jpg (平铺)

        # 先检查是否直接有图片文件
        direct_images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + \
                       list(person_dir.glob("*.png")) + list(person_dir.glob("*.bmp"))

        if direct_images:
            # 平铺结构
            test_images.extend(direct_images)
        else:
            # 嵌套结构
            for video_dir in person_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                for img_path in video_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        test_images.append(img_path)

    if not test_images:
        print("❌ 未找到测试图像")
        return

    # 如果指定了最大样本数
    if max_samples > 0 and len(test_images) > max_samples:
        import random
        test_images = random.sample(test_images, max_samples)

    print(f"📸 找到 {len(test_images)} 张测试图像")

    success_count = 0

    for img_path in tqdm(test_images, desc="延迟分析", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            # 记录总开始时间
            total_start = time.time()

            if skip_detection:
                # 跳过人脸检测，假设图像已经是对齐好的人脸
                profiler.add_timing('detection', 0.0)  # 检测时间为0
                face_img = img
            else:
                # 1. 人脸检测
                detect_start = time.time()
                faces = detector.detect(img)
                detect_end = time.time()
                profiler.add_timing('detection', detect_end - detect_start)

                if len(faces) == 0:
                    continue

                # 使用第一个检测到的人脸 (返回格式: (x1, y1, x2, y2))
                x1, y1, x2, y2 = faces[0]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                face_img = img[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

            # 2. 特征提取
            extract_start = time.time()
            if skip_detection:
                # 直接对整张图像提取特征（已对齐的人脸）
                # 需要调整大小并转换为RGB
                face_resized = cv2.resize(face_img, (112, 112))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                feature = model.embed(face_rgb)
            else:
                # 已经裁剪好了人脸，直接调整大小并提取特征
                face_resized = cv2.resize(face_img, (112, 112))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                feature = model.embed(face_rgb)
            extract_end = time.time()
            profiler.add_timing('extraction', extract_end - extract_start)

            if feature is None:
                continue

            # 3. 底库检索
            match_start = time.time()
            result = find_best_match(feature, gallery, reduce='topk_mean', topk=3)
            match_end = time.time()
            profiler.add_timing('matching', match_end - match_start)

            # 记录总时间
            total_end = time.time()
            profiler.add_timing('total', total_end - total_start)

            success_count += 1

        except Exception as e:
            continue

    print(f"✅ 成功分析 {success_count} / {len(test_images)} 张图片")


def analyze_from_video(video_path: Path, model, detector, gallery: Dict,
                      profiler: LatencyProfiler, max_frames: int = -1):
    """从视频分析延迟"""

    print(f"\n⏳ 正在分析视频: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 视频信息: {total_frames} 帧, {fps:.2f} FPS")

    frame_count = 0
    success_count = 0

    # 创建进度条
    pbar = tqdm(total=min(max_frames, total_frames) if max_frames > 0 else total_frames,
                desc="延迟分析", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if max_frames > 0 and frame_count > max_frames:
            break

        try:
            # 记录总开始时间
            total_start = time.time()

            # 1. 人脸检测
            detect_start = time.time()
            faces = detector.detect_faces(frame)
            detect_end = time.time()
            profiler.add_timing('detection', detect_end - detect_start)

            if len(faces) == 0:
                pbar.update(1)
                continue

            # 使用第一个检测到的人脸
            face = faces[0]
            x1, y1, x2, y2 = map(int, face['box'])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                pbar.update(1)
                continue

            # 2. 特征提取
            extract_start = time.time()
            if skip_detection:
                # 直接对整张图像提取特征（已对齐的人脸）
                # 需要调整大小并转换为RGB
                face_resized = cv2.resize(face_img, (112, 112))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                feature = model.embed(face_rgb)
            else:
                # 已经裁剪好了人脸，直接调整大小并提取特征
                face_resized = cv2.resize(face_img, (112, 112))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                feature = model.embed(face_rgb)
            extract_end = time.time()
            profiler.add_timing('extraction', extract_end - extract_start)

            if feature is None:
                pbar.update(1)
                continue

            # 3. 底库检索
            match_start = time.time()
            result = find_best_match(feature, gallery, reduce='topk_mean', topk=3)
            match_end = time.time()
            profiler.add_timing('matching', match_end - match_start)

            # 记录总时间
            total_end = time.time()
            profiler.add_timing('total', total_end - total_start)

            success_count += 1

        except Exception as e:
            pass

        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"✅ 成功分析 {success_count} / {frame_count} 帧")


def main():
    parser = argparse.ArgumentParser(description='人脸识别系统端到端延迟分析')

    # 输入源（图像或视频）
    parser.add_argument('--images-dir', type=str,
                       help='测试图像目录')
    parser.add_argument('--video', type=str,
                       help='测试视频路径')
    parser.add_argument('--max-samples', type=int, default=-1,
                       help='最大测试样本数 (图像模式, -1表示全部)')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='最大测试帧数 (视频模式, -1表示全部)')

    # 数据库
    parser.add_argument('--db-path', type=str, required=True,
                       help='特征库路径')
    parser.add_argument('--gallery-mode', type=str, default='mean',
                       choices=['mean', 'max', 'all'],
                       help='特征库模式')

    # 模型参数
    parser.add_argument('--weights', type=str, default='weights/adasin_best.pt',
                       help='模型权重路径')
    parser.add_argument('--model-name', type=str, default='iresnet50',
                       help='模型名称')
    parser.add_argument('--img-size', type=int, default=112,
                       help='输入图像大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备: auto/cuda/cpu')

    # 测试选项
    parser.add_argument('--skip-detection', action='store_true',
                       help='跳过人脸检测步骤（用于已对齐的人脸图像）')

    # 输出
    parser.add_argument('--output', type=str, default='latency_analysis.json',
                       help='结果输出路径')

    args = parser.parse_args()

    # 检查输入
    if not args.images_dir and not args.video:
        print("❌ 请指定 --images-dir 或 --video")
        return 1

    print("\n" + "="*70)
    print("⏱️  人脸识别系统端到端延迟分析")
    print("="*70)

    # 加载模型
    print("\n⏳ 加载模型...")
    model, detector = build_runtime(
        weights_path=args.weights,
        model_name=args.model_name,
        img_size=args.img_size,
        device=args.device
    )
    print("✅ 模型加载完成")

    # 加载特征库
    print("\n⏳ 加载特征库...")
    gallery = load_gallery(args.db_path, args.gallery_mode)

    # 获取数据库人数
    with FeatureDB(args.db_path) as db:
        db_stats = db.get_stats()
        db_size = db_stats['person_count']

    print(f"✅ 特征库加载完成: {db_size:,} 人, {len(gallery)} 个模板")

    # 创建分析器
    profiler = LatencyProfiler()

    # 执行分析
    if args.images_dir:
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            print(f"❌ 图像目录不存在: {images_dir}")
            return 1

        analyze_from_images(images_dir, model, detector, gallery, profiler,
                           args.max_samples, args.skip_detection)

    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"❌ 视频不存在: {video_path}")
            return 1

        analyze_from_video(video_path, model, detector, gallery, profiler, args.max_frames)

    # 打印报告
    profiler.print_report(db_size)

    # 保存结果
    profiler.save_to_json(args.output, db_size)
    print(f"💾 结果已保存到: {args.output}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
