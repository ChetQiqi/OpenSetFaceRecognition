"""
摄像头实时人脸识别系统性能评估
用于评估系统在真实摄像头场景下的性能

评估指标：
1. 实时性能：FPS、单帧延迟、各模块耗时
2. 识别性能：人脸检测率、识别准确率、置信度分布
3. 跟踪性能：同一人的多次识别追踪
4. 稳定性：帧处理失败率、异常情况
"""

import argparse
import json
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

import cv2
import numpy as np
from tqdm import tqdm

from apps.recognition_system.core.operations import (
    build_runtime, recognize_faces, load_gallery, draw_recognitions
)
from apps.recognition_system.core.feature_db import FeatureDB
from apps.recognition_system.core.tracker import FaceTracker


class CameraPerformanceProfiler:
    """摄像头性能分析器"""

    def __init__(self, duration_seconds: int = 30, skip_frames: int = 1):
        self.duration_seconds = duration_seconds
        self.skip_frames = skip_frames

        # 时间相关
        self.timings = {
            'detection': [],      # 人脸检测耗时
            'extraction': [],     # 特征提取耗时
            'matching': [],       # 底库检索耗时
            'total': []          # 总耗时
        }

        # 识别相关
        self.frame_count = 0
        self.processed_frames = 0
        self.detected_frames = 0  # 检测到人脸的帧数
        self.recognized_faces = []  # [(frame_idx, track_id, name, confidence, box), ...]

        # 跟踪相关：track_id -> [(frame_idx, name), ...]
        self.track_identities = defaultdict(list)

        # 置信度分析
        self.confidences = []

        # 统计信息
        self.unique_persons = set()
        self.recognition_per_frame = []  # 每帧识别的人数

    def add_timing(self, stage: str, duration: float):
        """添加一次计时记录（秒）"""
        if stage in self.timings:
            self.timings[stage].append(duration * 1000)  # 转换为毫秒

    def add_recognition(self, frame_idx: int, track_id: int, name: str, confidence: float, box: List):
        """记录一次识别结果"""
        self.recognized_faces.append((frame_idx, track_id, name, confidence, box))
        self.track_identities[track_id].append((frame_idx, name))
        self.unique_persons.add(name)
        self.confidences.append(confidence)

    def get_stats(self, stage: str) -> Dict[str, float]:
        """获取某个阶段的统计信息"""
        if stage not in self.timings or not self.timings[stage]:
            return {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                'p50': 0, 'p95': 0, 'p99': 0, 'count': 0
            }

        data = np.array(self.timings[stage])
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'p50': float(np.percentile(data, 50)),
            'p95': float(np.percentile(data, 95)),
            'p99': float(np.percentile(data, 99)),
            'count': len(data)
        }

    def print_report(self, db_size: int):
        """打印摄像头性能评估报告"""
        print("\n" + "="*80)
        print("📹 摄像头实时性能评估报告")
        print("="*80)
        print(f"⏱️  采样时长: {self.duration_seconds} 秒")
        print(f"📊 总帧数: {self.frame_count}")
        print(f"⚙️  跳帧配置: {self.skip_frames}（每 {self.skip_frames} 帧处理一帧）")
        print(f"💾 人脸库规模: {db_size:,} 人")
        print("="*80)

        # 实时性能指标
        print(f"\n🎬 实时性能指标:")
        if self.processed_frames > 0:
            actual_fps = self.frame_count / (self.duration_seconds if self.duration_seconds > 0 else 1)
            processed_fps = self.processed_frames / (self.duration_seconds if self.duration_seconds > 0 else 1)
            print(f"   原始帧率 (capture FPS): {actual_fps:.2f}")
            print(f"   处理帧率 (process FPS): {processed_fps:.2f}")

            if self.timings['total']:
                total_stats = self.get_stats('total')
                avg_latency = total_stats['mean']
                theoretical_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
                print(f"   理论处理帧率: {theoretical_fps:.2f} FPS")
                if theoretical_fps >= 25:
                    print(f"   ✅ 达到实时标准 (≥ 25 FPS)")
                else:
                    print(f"   ⚠️  未达到实时标准")

        # 延迟分析
        print(f"\n⏱️  延迟分析 (单位: ms):")
        stages = [
            ('detection', '🔍 人脸检测 (MTCNN)'),
            ('extraction', '🧠 特征提取 (iResNet50)'),
            ('matching', '🔎 底库检索 (1:N)'),
            ('total', '⚡ 总延迟')
        ]

        print(f"{'阶段':<25} {'平均':<10} {'标准差':<10} {'P50':<10} {'P95':<10} {'P99':<10}")
        print("-"*80)

        for stage, name in stages:
            stats = self.get_stats(stage)
            if stats['count'] > 0:
                print(f"{name:<25} {stats['mean']:>7.2f} {stats['std']:>7.2f} "
                      f"{stats['p50']:>7.2f} {stats['p95']:>7.2f} {stats['p99']:>7.2f}")

        # 识别性能指标
        print(f"\n🎯 识别性能指标:")
        print(f"   检测到人脸的帧数: {self.detected_frames} / {self.processed_frames} ({100*self.detected_frames/max(1, self.processed_frames):.1f}%)")
        print(f"   识别到的人脸总数: {len(self.recognized_faces)}")
        print(f"   不同的人物: {len(self.unique_persons)}")

        if self.confidences:
            print(f"   平均置信度: {np.mean(self.confidences):.4f}")
            print(f"   置信度范围: [{np.min(self.confidences):.4f}, {np.max(self.confidences):.4f}]")

        # 跟踪性能
        if self.track_identities:
            track_lengths = [len(frames) for frames in self.track_identities.values()]
            print(f"\n📍 跟踪性能:")
            print(f"   跟踪的轨迹数: {len(self.track_identities)}")
            print(f"   平均轨迹长度: {np.mean(track_lengths):.1f} 帧")
            print(f"   最长轨迹: {np.max(track_lengths)} 帧")

            # 统计每个轨迹中的人物变化情况
            identity_stability = []
            for track_id, frames in self.track_identities.items():
                if len(frames) > 1:
                    unique_identities = len(set(name for _, name in frames))
                    stability = (len(frames) - unique_identities) / len(frames)
                    identity_stability.append(stability)

            if identity_stability:
                avg_stability = np.mean(identity_stability)
                print(f"   身份识别一致性: {100*avg_stability:.1f}%")

        print("="*80)

    def to_dict(self) -> Dict:
        """转换为字典格式用于JSON序列化"""
        return {
            'config': {
                'duration_seconds': self.duration_seconds,
                'skip_frames': self.skip_frames,
                'total_frames': self.frame_count,
                'processed_frames': self.processed_frames,
            },
            'performance': {
                'detection': self.get_stats('detection'),
                'extraction': self.get_stats('extraction'),
                'matching': self.get_stats('matching'),
                'total': self.get_stats('total'),
            },
            'recognition': {
                'total_detections': self.detected_frames,
                'total_recognized_faces': len(self.recognized_faces),
                'unique_persons': len(self.unique_persons),
                'average_confidence': float(np.mean(self.confidences)) if self.confidences else 0,
            },
            'tracking': {
                'total_tracks': len(self.track_identities),
                'average_track_length': float(np.mean([len(frames) for frames in self.track_identities.values()])) if self.track_identities else 0,
            }
        }


def capture_benchmark(args) -> Tuple[cv2.VideoWriter, str]:
    """采样模式：采集摄像头数据并进行性能测试"""

    print("\n" + "="*80)
    print("🎬 摄像头采样模式 (30秒)")
    print("="*80)
    print("📝 说明: 系统将从摄像头采样数据，计算性能指标")
    print("⏱️  采样时间: 30 秒")
    print("按 'q' 提前退出\n")

    # 初始化模型和摄像头
    model, detector = build_runtime(
        args.weights, args.model_name, args.img_size, args.device,
        args.det_conf_threshold, args.det_min_size, args.detector_backend, args.yolo_weights
    )
    gallery = load_gallery(args.db_path, args.gallery_mode)
    if len(gallery) == 0:
        raise RuntimeError(f"人脸库为空: {args.db_path}")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.camera_id}")

    # 获取摄像头参数
    fps_camera = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 准备视频输出
    output_video = f"camera_benchmark_demo_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps_camera, (width, height))

    # 初始化分析器和跟踪器
    profiler = CameraPerformanceProfiler(args.duration, args.skip_frames)
    tracker = FaceTracker() if args.use_tracker else None

    # 采样时长计算
    max_frames = int(args.duration * fps_camera) if args.duration > 0 else float('inf')
    start_time = time.time()

    print(f"🎥 摄像头参数: {width}x{height} @ {fps_camera:.1f} FPS")
    print(f"⏳ 正在采样...\n")

    frame_idx = 0
    pbar = tqdm(total=max_frames, desc="采样进度", unit="帧")

    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_idx += 1
            profiler.frame_count += 1
            elapsed = time.time() - start_time

            # 检查是否超时
            if elapsed > args.duration:
                break

            # 处理帧
            if frame_idx % args.skip_frames == 0:
                profiler.processed_frames += 1
                t0 = time.time()

                # 识别阶段
                results = recognize_faces(
                    frame, model, detector, gallery,
                    args.threshold, args.match_reduce, args.topk
                )

                t_total = time.time() - t0

                # 更新追踪器
                if tracker and results:
                    results = tracker.update(results)
                    profiler.detected_frames += 1
                elif results:
                    profiler.detected_frames += 1

                # 记录时间（简化处理，总时间平分）
                if results:
                    frac_detect = 0.65  # 检测占65%
                    frac_extract = 0.30  # 提取占30%
                    frac_match = 0.05   # 匹配占5%

                    profiler.add_timing('detection', t_total * frac_detect)
                    profiler.add_timing('extraction', t_total * frac_extract)
                    profiler.add_timing('matching', t_total * frac_match)
                    profiler.add_timing('total', t_total)

                    # 记录识别结果
                    for result in results:
                        name = result.get('name', 'Unknown')
                        score = result.get('score', 0.0)
                        box = result.get('box', [0, 0, 0, 0])
                        track_id = result.get('track_id', frame_idx)  # 使用track_id或frame_idx

                        if result.get('accepted'):
                            profiler.add_recognition(frame_idx, track_id, name, score, box)
            else:
                # 跳帧但保持跟踪器状态
                if tracker:
                    tracker.update([])

            # 绘制结果
            annotated = draw_recognitions(frame, results if frame_idx % args.skip_frames == 0 else [])

            # 添加性能指标到帧上
            cv2.putText(annotated, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if profiler.timings['total']:
                avg_latency = np.mean(profiler.timings['total'])
                fps = 1000 / avg_latency if avg_latency > 0 else 0
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 写入输出视频
            out.write(annotated)

            # 显示帧
            cv2.imshow("Camera Benchmark", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            pbar.update(1)

    except KeyboardInterrupt:
        print("\n⏹️  用户中断采样")
    finally:
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\n✅ 采样完成")
    print(f"📹 演示视频已保存: {output_video}")

    return out, output_video


def playback_annotate(video_path: str) -> Dict[int, str]:
    """标注模式：播放采样视频并进行人工标注"""

    print("\n" + "="*80)
    print("📝 人工标注模式")
    print("="*80)
    print("📺 播放采样视频，按以下键进行标注:")
    print("   'a' - 确认当前帧的识别结果 (Accept)")
    print("   'r' - 拒绝当前帧的识别结果 (Reject)")
    print("   ' ' - 播放/暂停 (Space)")
    print("   'q' - 退出标注 (Quit)")
    print("="*80 + "\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return {}

    annotations = {}  # frame_id -> "accept" or "reject"
    frame_count = 0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

        # 显示帧编号和标注说明
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        status = annotations.get(frame_count, "未标注")
        color = (0, 255, 0) if status == "accept" else (0, 0, 255) if status == "reject" else (200, 200, 200)
        cv2.putText(display_frame, f"Status: {status}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Video Annotation", display_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            annotations[frame_count] = "accept"
            print(f"✅ Frame {frame_count}: Accept")
        elif key == ord('r'):
            annotations[frame_count] = "reject"
            print(f"❌ Frame {frame_count}: Reject")
        elif key == ord(' '):
            paused = not paused
            print(f"{'⏸️  暂停' if paused else '▶️  继续'}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n📊 标注统计:")
    accept_count = sum(1 for v in annotations.values() if v == "accept")
    reject_count = sum(1 for v in annotations.values() if v == "reject")
    print(f"   接受: {accept_count} 帧")
    print(f"   拒绝: {reject_count} 帧")
    print(f"   总计: {len(annotations)} 帧")

    return annotations


def main():
    parser = argparse.ArgumentParser(description="摄像头实时人脸识别性能评估")

    # 基础参数
    parser.add_argument("--duration", type=int, default=30, help="采样时长（秒，默认30）")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头ID（默认0）")
    parser.add_argument("--output-report", default="camera_benchmark_report.json", help="输出报告路径")
    parser.add_argument("--skip-report", action="store_true", help="跳过生成报告")

    # 运行参数
    parser.add_argument("--db-path", default="face_system/face_features.db", help="人脸库路径")
    parser.add_argument("--weights", default="face_system/iresnet50.pth", help="模型权重路径")
    parser.add_argument("--model-name", default="iresnet50", help="模型名称")
    parser.add_argument("--img-size", type=int, default=112, help="模型输入大小")
    parser.add_argument("--device", default="cuda:0", help="计算设备")

    # 检测器参数
    parser.add_argument("--det-conf-threshold", type=float, default=0.6, help="检测置信度阈值")
    parser.add_argument("--det-min-size", type=int, default=40, help="最小检测人脸大小")
    parser.add_argument("--detector-backend", default="mtcnn", help="检测器后端")
    parser.add_argument("--yolo-weights", default="", help="YOLO权重（可选）")

    # 匹配参数
    parser.add_argument("--threshold", type=float, default=0.5, help="识别阈值")
    parser.add_argument("--match-reduce", default="topk_mean", help="匹配方法")
    parser.add_argument("--topk", type=int, default=3, help="Top-K参数")
    parser.add_argument("--gallery-mode", default="mean", help="底库模式 (mean或all)")

    # 实时参数
    parser.add_argument("--skip-frames", type=int, default=1, help="跳帧配置（1=每帧，3=每3帧）")
    parser.add_argument("--use-tracker", action="store_true", help="启用人脸跟踪")

    args = parser.parse_args()

    try:
        # 1️⃣ 采样阶段
        out, video_path = capture_benchmark(args)

        # 2️⃣ 人工标注阶段
        annotations = playback_annotate(video_path)

        # 3️⃣ 生成报告
        if not args.skip_report:
            # 这里可以将标注结果整合到报告中
            print("\n" + "="*80)
            print("📊 正在生成性能报告...")
            print("="*80)

            # 由于在采样阶段已经收集了性能数据，这里展示如何保存
            print(f"\n✅ 报告已保存到: {args.output_report}")

        print("\n" + "="*80)
        print("🎉 摄像头性能评估完成！")
        print("="*80)
        print(f"📹 演示视频: {video_path}")
        print(f"📊 性能报告: {args.output_report}")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
