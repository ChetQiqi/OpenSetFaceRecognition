"""
摄像头基准测试交互式工具
支持：采样 -> 标注 -> 生成报告 -> 可视化
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import cv2
import numpy as np
from tqdm import tqdm

from apps.recognition_system.core.operations import (
    build_runtime, recognize_faces, load_gallery, draw_recognitions
)
from apps.recognition_system.core.feature_db import FeatureDB
from apps.recognition_system.core.tracker import FaceTracker


class CameraMetricsCollector:
    """摄像头性能指标收集器 - 完整版本"""

    def __init__(self, duration_seconds: int = 30, skip_frames: int = 1):
        self.duration_seconds = duration_seconds
        self.skip_frames = skip_frames
        self.start_time = None

        # 帧级数据
        self.frames = []  # [{'frame_idx': int, 'timestamp': float, 'detected': bool, 'recognized': int, ...}, ...]

        # 时间分析
        self.timings = {
            'detection': [],
            'extraction': [],
            'matching': [],
            'total': []
        }

        # 识别数据
        self.recognitions = []  # [{'frame_idx': int, 'track_id': int, 'name': str, 'confidence': float}, ...]
        self.detections = 0
        self.unique_persons = set()
        self.confidences = []

        # 跟踪数据
        self.tracks = {}  # track_id -> {'frames': [...], 'names': set(), 'length': int}

    def record_frame_start(self, frame_idx: int) -> Tuple[float, int]:
        """记录帧处理开始"""
        t0 = time.time()
        return t0, frame_idx

    def record_frame_complete(self, frame_idx: int, t0: float, detected: int, results: List):
        """记录帧处理完成"""
        elapsed = (time.time() - t0) * 1000  # ms

        frame_record = {
            'frame_idx': frame_idx,
            'timestamp': time.time() - self.start_time if self.start_time else 0,
            'elapsed_ms': elapsed,
            'detected': detected > 0,
            'detected_count': detected,
            'recognized_count': len([r for r in results if r.get('accepted')])
        }
        self.frames.append(frame_record)

        if detected > 0:
            self.detections += detected

    def record_timing(self, stage: str, duration_ms: float):
        """记录各阶段耗时"""
        if stage in self.timings:
            self.timings[stage].append(duration_ms)

    def record_recognition(self, frame_idx: int, track_id: int, name: str, confidence: float):
        """记录识别结果"""
        self.recognitions.append({
            'frame_idx': frame_idx,
            'track_id': track_id,
            'name': name,
            'confidence': confidence
        })
        self.unique_persons.add(name)
        self.confidences.append(confidence)

        # 更新轨迹信息
        if track_id not in self.tracks:
            self.tracks[track_id] = {'frames': [], 'names': set(), 'start_frame': frame_idx}

        self.tracks[track_id]['frames'].append(frame_idx)
        self.tracks[track_id]['names'].add(name)
        self.tracks[track_id]['length'] = len(self.tracks[track_id]['frames'])

    def get_stats(self, stage: str) -> Dict:
        """计算统计信息"""
        if stage not in self.timings or not self.timings[stage]:
            return {}

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

    def to_report(self, db_size: int, total_frames: int, processed_frames: int) -> Dict:
        """生成完整报告字典"""
        fps_actual = total_frames / self.duration_seconds if self.duration_seconds > 0 else 0
        fps_processed = processed_frames / self.duration_seconds if self.duration_seconds > 0 else 0

        total_stats = self.get_stats('total')
        fps_theoretical = 1000 / total_stats['mean'] if total_stats.get('mean', 0) > 0 else 0

        return {
            'config': {
                'duration_seconds': self.duration_seconds,
                'skip_frames': self.skip_frames,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'db_size': db_size
            },
            'performance_metrics': {
                'fps_actual': round(fps_actual, 2),
                'fps_processed': round(fps_processed, 2),
                'fps_theoretical': round(fps_theoretical, 2),
                'latency_stages': {
                    'detection': self.get_stats('detection'),
                    'extraction': self.get_stats('extraction'),
                    'matching': self.get_stats('matching'),
                    'total': self.get_stats('total')
                }
            },
            'recognition_metrics': {
                'total_detections': self.detections,
                'total_recognitions': len(self.recognitions),
                'unique_persons': len(self.unique_persons),
                'detection_rate': round(100 * self.detections / max(1, processed_frames), 2),
                'confidence_stats': {
                    'mean': round(float(np.mean(self.confidences)), 4) if self.confidences else 0,
                    'std': round(float(np.std(self.confidences)), 4) if self.confidences else 0,
                    'min': round(float(np.min(self.confidences)), 4) if self.confidences else 0,
                    'max': round(float(np.max(self.confidences)), 4) if self.confidences else 0
                }
            },
            'tracking_metrics': {
                'total_tracks': len(self.tracks),
                'avg_track_length': round(float(np.mean([t['length'] for t in self.tracks.values()])), 2) if self.tracks else 0,
                'max_track_length': max([t['length'] for t in self.tracks.values()]) if self.tracks else 0,
                'identity_consistency': self._calc_identity_consistency()
            }
        }

    def _calc_identity_consistency(self) -> float:
        """计算身份识别一致性（百分比）"""
        if not self.tracks:
            return 0.0

        consistencies = []
        for track in self.tracks.values():
            if len(track['names']) > 0:
                consistency = 1.0 / len(track['names'])  # 越多不同的身份，一致性越低
                consistencies.append(consistency)

        return round(100 * float(np.mean(consistencies)) if consistencies else 0.0, 2)


class InteractiveCamera:
    """交互式摄像头采样工具"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.detector = None
        self.gallery = None
        self.tracker = None
        self.collector = None
        self.identity_stats = None  # Adaptive framework statistics
        self.feature_db = None  # Database for logging

    def setup(self):
        """初始化模型和资源"""
        print("\n🔧 正在初始化系统...")
        self.model, self.detector = build_runtime(
            self.args.weights, self.args.model_name, self.args.img_size, self.args.device,
            self.args.det_conf_threshold, self.args.det_min_size,
            self.args.detector_backend, self.args.yolo_weights
        )
        self.gallery = load_gallery(self.args.db_path, self.args.gallery_mode)
        if len(self.gallery) == 0:
            raise RuntimeError(f"❌ 人脸库为空")

        if self.args.use_tracker:
            self.tracker = FaceTracker()

        self.collector = CameraMetricsCollector(self.args.duration, self.args.skip_frames)

        # Adaptive framework initialization
        if self.args.adaptive_mode == "gaussian":
            from apps.recognition_system.core.adaptive_threshold import compute_adaptive_thresholds
            from apps.recognition_system.core.feature_db import FeatureDB

            print("🧠 正在初始化自适应阈值框架...")
            self.feature_db = FeatureDB(self.args.db_path)

            # Try to load existing statistics from database
            self.identity_stats = self.feature_db.load_identity_statistics()

            if len(self.identity_stats) == 0:
                # No statistics yet, compute from scratch
                print("   计算per-identity自适应阈值...")
                self.identity_stats = compute_adaptive_thresholds(self.args.db_path)

                # Save to database
                for stats in self.identity_stats.values():
                    self.feature_db.save_identity_statistics(stats)

                print(f"   ✅ 已为 {len(self.identity_stats)} 个身份计算自适应阈值")
            else:
                print(f"   ✅ 已加载 {len(self.identity_stats)} 个身份的自适应阈值")

            if self.args.enable_temporal_adaptation:
                print(f"   📈 在线学习已启用 (学习率={self.args.temporal_learning_rate})")

        print("✅ 初始化完成\n")

    def capture_benchmark(self) -> str:
        """采样摄像头并收集性能数据"""
        print("="*100)
        print("📹 摄像头采样模式")
        print("="*100)
        print(f"⏱️  采样时长: {self.args.duration} 秒")
        print(f"⚙️  跳帧配置: {self.args.skip_frames}（每 {self.args.skip_frames} 帧处理一帧）")
        print(f"💾 人脸库规模: {len(self.gallery)} 人")
        print("按 'q' 提前退出\n")

        cap = cv2.VideoCapture(self.args.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"❌ 无法打开摄像头 {self.args.camera_id}")

        fps_camera = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 准备视频输出（保存到与报告相同的文件夹）
        output_dir = Path(self.args.output_report).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = str(output_dir / f"camera_demo_{int(time.time())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps_camera, (width, height))

        max_frames = int(self.args.duration * fps_camera) if self.args.duration > 0 else float('inf')
        self.collector.start_time = time.time()

        print(f"🎥 分辨率: {width}x{height} @ {fps_camera:.1f} FPS")
        print(f"⏳ 正在采样...\n")

        frame_idx = 0
        processed_frames = 0
        pbar = tqdm(total=max_frames, desc="采样进度", unit="帧", unit_scale=True)

        try:
            while frame_idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame_idx += 1
                elapsed = time.time() - self.collector.start_time

                if elapsed > self.args.duration:
                    break

                # 处理帧
                if frame_idx % self.args.skip_frames == 0:
                    processed_frames += 1
                    t0 = time.time()
                    results = recognize_faces(
                        frame, self.model, self.detector, self.gallery,
                        self.args.threshold, self.args.match_reduce, self.args.topk,
                        adaptive_mode=self.args.adaptive_mode,
                        identity_stats=self.identity_stats,
                        feature_db=self.feature_db
                    )
                    elapsed_ms = (time.time() - t0) * 1000

                    detected_count = len(results)
                    if self.tracker and detected_count > 0:
                        results = self.tracker.update(results)

                    # 记录性能数据
                    self.collector.record_frame_complete(frame_idx, t0, detected_count, results)
                    self.collector.record_timing('total', elapsed_ms)

                    # 记录识别结果
                    for result in results:
                        if result.get('accepted'):
                            name = result.get('name', 'Unknown')
                            score = result.get('score', 0.0)
                            track_id = result.get('track_id', frame_idx)
                            self.collector.record_recognition(frame_idx, track_id, name, score)

                    # Temporal adaptation: update statistics based on accepted recognitions
                    if self.args.adaptive_mode == "gaussian" and self.args.enable_temporal_adaptation:
                        from apps.recognition_system.core.adaptive_threshold import update_identity_statistics

                        for result in results:
                            if result.get('accepted') and result['name'] != "Unknown":
                                identity_name = result['name']
                                raw_score = result.get('raw_score', 0.0)

                                if identity_name in self.identity_stats:
                                    self.identity_stats[identity_name] = update_identity_statistics(
                                        self.identity_stats[identity_name],
                                        raw_score,
                                        learning_rate=self.args.temporal_learning_rate
                                    )

                                    # Log to database every 10 frames to reduce overhead
                                    if frame_idx % 10 == 0:
                                        person_id = self.identity_stats[identity_name].person_id
                                        self.feature_db.log_recognition_history(
                                            person_id, raw_score, is_genuine=True
                                        )
                else:
                    if self.tracker:
                        self.tracker.update([])

                # 绘制并保存帧
                annotated = draw_recognitions(frame, results if frame_idx % self.args.skip_frames == 0 else [])

                # 显示性能指标
                cv2.putText(annotated, f"Frame: {frame_idx}/{int(max_frames)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if self.collector.timings['total']:
                    avg_latency = np.mean(self.collector.timings['total'])
                    fps = 1000 / avg_latency if avg_latency > 0 else 0
                    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                out.write(annotated)
                cv2.imshow("Camera Demo", annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pbar.update(1)

        except KeyboardInterrupt:
            print("\n⏹️  用户中断")
        finally:
            # Save updated adaptive statistics
            if self.args.adaptive_mode == "gaussian" and self.identity_stats and self.feature_db:
                print("\n💾 保存更新的自适应阈值...")
                for stats in self.identity_stats.values():
                    self.feature_db.save_identity_statistics(stats)
                print(f"   ✅ 已保存 {len(self.identity_stats)} 个身份的统计信息")

            pbar.close()
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"\n✅ 采样完成")
        print(f"📹 演示视频: {output_video}")
        print(f"📊 采集数据: {processed_frames} 帧，{len(self.collector.recognitions)} 次识别")

        return output_video

    def annotate_video(self, video_path: str) -> Dict[int, str]:
        """交互式标注采样视频 - 支持逐帧手动浏览"""
        print("\n" + "="*100)
        print("📝 人工标注模式 (帧级标注)")
        print("="*100)
        print("操作说明:")
        print("  'a' - 接受：本帧识别正确 (Accept) → 自动进到下一帧")
        print("  'r' - 拒绝：本帧识别错误 (Reject) → 自动进到下一帧")
        print("  'n' - 下一帧 (Next Frame)")
        print("  'p' - 上一帧 (Previous Frame)")
        print("  ' ' - 切换 自动播放/手动模式 (Toggle Auto Play)")
        print("  'q' - 退出标注 (Quit)")
        print("="*100 + "\n")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return {}

        # 读取所有帧到内存（方便前后跳转）
        print("⏳ 加载视频帧到内存...")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        cap.release()
        total_frames = len(frames)
        print(f"✅ 已加载 {total_frames} 帧\n")

        annotations = {}
        frame_idx = 0  # 当前帧索引（0-based）
        auto_play = False
        auto_play_speed = 200  # 毫秒，控制自动播放速度

        while frame_idx < total_frames:
            # 显示当前帧
            frame = frames[frame_idx]
            display = frame.copy()

            # 帧号（显示为1-based）
            frame_num = frame_idx + 1
            progress = f"Frame #{frame_num}/{total_frames}"
            cv2.putText(display, progress, (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 标注状态
            status = annotations.get(frame_num, "?")
            if status == "accept":
                color, text = (0, 255, 0), "✅ Accept"
            elif status == "reject":
                color, text = (0, 0, 255), "❌ Reject"
            else:
                color, text = (200, 200, 200), "● Pending"

            cv2.putText(display, text, (15, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 播放模式提示
            if auto_play:
                mode_text = "⏯️ Auto Play (按 ' ' 切换到手动)"
                mode_color = (0, 200, 0)
            else:
                mode_text = "⏸️ Manual Mode (按 'n' 下一帧，'p' 上一帧)"
                mode_color = (200, 100, 0)
            cv2.putText(display, mode_text, (15, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)

            cv2.imshow("Annotation", display)

            # 等待按键
            wait_time = auto_play_speed if auto_play else 0  # 0 表示等待按键
            key = cv2.waitKey(wait_time) & 0xFF

            # 处理按键
            if key == ord('q'):
                print("\n⏹️  用户退出标注")
                break
            elif key == ord('a'):
                # 标注为正确，自动跳到下一帧
                annotations[frame_num] = "accept"
                print(f"✅ Frame {frame_num}: Accept")
                if frame_idx < total_frames - 1:
                    frame_idx += 1
            elif key == ord('r'):
                # 标注为错误，自动跳到下一帧
                annotations[frame_num] = "reject"
                print(f"❌ Frame {frame_num}: Reject")
                if frame_idx < total_frames - 1:
                    frame_idx += 1
            elif key == ord('n'):
                # 下一帧
                if frame_idx < total_frames - 1:
                    frame_idx += 1
                    print(f"→ 下一帧: {frame_idx + 1}")
            elif key == ord('p'):
                # 上一帧
                if frame_idx > 0:
                    frame_idx -= 1
                    print(f"← 上一帧: {frame_idx + 1}")
            elif key == ord(' '):
                # 切换自动播放/手动模式
                auto_play = not auto_play
                if auto_play:
                    print(f"⏯️  启用自动播放 (每 {auto_play_speed}ms 一帧)")
                else:
                    print("⏸️  切换到手动模式 (按 'n' 下一帧，'p' 上一帧)")
            elif auto_play and key != 255:
                # 自动播放模式下，有无按键都继续
                if frame_idx < total_frames - 1:
                    frame_idx += 1
                else:
                    # 到达最后一帧，停止自动播放
                    auto_play = False
                    print("\n✅ 自动播放至末尾，已切换为手动模式")
            elif auto_play and key == 255:
                # 自动播放模式下，无按键时继续下一帧
                if frame_idx < total_frames - 1:
                    frame_idx += 1
                else:
                    auto_play = False

        cv2.destroyAllWindows()

        print(f"\n📊 标注完成:")
        accept_count = sum(1 for v in annotations.values() if v == 'accept')
        reject_count = sum(1 for v in annotations.values() if v == 'reject')
        print(f"   接受: {accept_count} 帧 ✅")
        print(f"   拒绝: {reject_count} 帧 ❌")
        print(f"   标注率: {len(annotations)}/{total_frames} ({100*len(annotations)/max(1,total_frames):.1f}%)")

        return annotations

    def generate_report(self, video_path: str):
        """生成性能报告"""
        print("\n" + "="*100)
        print("📊 生成性能报告")
        print("="*100)

        report = self.collector.to_report(
            db_size=len(self.gallery),
            total_frames=self.collector.frames[-1]['frame_idx'] if self.collector.frames else 0,
            processed_frames=len(self.collector.timings['total'])
        )

        # 保存JSON报告
        report['video_file'] = video_path
        report['generated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 创建output目录（支持camera_eval）
        output_path = Path(self.args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON报告已保存: {output_path}")

        # 保存CSV表格版本
        self._save_report_csv(report, output_path.parent)

        # 打印报告摘要
        self._print_report_summary(report)

        return report

    def _save_report_csv(self, report: Dict, output_dir: Path):
        """保存表格格式的报告 (CSV)"""
        import csv

        csv_file = output_dir / "performance_metrics.csv"

        config = report['config']
        perf = report['performance_metrics']
        recog = report['recognition_metrics']
        track = report['tracking_metrics']

        # 安全获取延迟数据（处理空字典的情况）
        def get_latency_value(stage_name, key='mean', default=0.0):
            stage = perf['latency_stages'].get(stage_name, {})
            return f"{stage.get(key, default):.2f}" if stage else "N/A"

        metrics_data = [
            ['配置项', '值'],
            ['', ''],
            ['采样时长', f"{config['duration_seconds']}秒"],
            ['总帧数', f"{config['total_frames']}"],
            ['处理帧数', f"{config['processed_frames']}"],
            ['人脸库规模', f"{config['db_size']}人"],
            ['', ''],
            ['实时性能', ''],
            ['实际帧率 (FPS)', f"{perf['fps_actual']:.2f}"],
            ['处理帧率 (FPS)', f"{perf['fps_processed']:.2f}"],
            ['理论帧率 (FPS)', f"{perf['fps_theoretical']:.2f}"],
            ['', ''],
            ['延迟分析 (ms)', ''],
            ['检测平均延迟', get_latency_value('detection', 'mean')],
            ['提取平均延迟', get_latency_value('extraction', 'mean')],
            ['匹配平均延迟', get_latency_value('matching', 'mean')],
            ['总平均延迟', get_latency_value('total', 'mean')],
            ['总P95延迟', get_latency_value('total', 'p95')],
            ['', ''],
            ['识别性能', ''],
            ['总检测数', f"{recog['total_detections']}"],
            ['总识别数', f"{recog['total_recognitions']}"],
            ['不同人物', f"{recog['unique_persons']}"],
            ['检测率 (%)', f"{recog['detection_rate']:.2f}"],
            ['平均置信度', f"{recog['confidence_stats']['mean']:.4f}"],
            ['', ''],
            ['跟踪性能', ''],
            ['轨迹总数', f"{track['total_tracks']}"],
            ['平均轨迹长度', f"{track['avg_track_length']:.1f}"],
            ['最长轨迹', f"{track['max_track_length']}"],
            ['身份一致性 (%)', f"{track['identity_consistency']:.2f}"],
        ]

        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_data)

        print(f"✅ CSV表格已保存: {csv_file}")

    def _print_report_summary(self, report: Dict):
        """打印报告摘要"""
        print("\n" + "="*100)
        print("📈 性能评估摘要")
        print("="*100)

        config = report['config']
        perf = report['performance_metrics']
        recog = report['recognition_metrics']
        track = report['tracking_metrics']

        print(f"\n⏱️  采样信息:")
        print(f"   采样时长: {config['duration_seconds']}秒")
        print(f"   总帧数: {config['total_frames']}")
        print(f"   处理帧数: {config['processed_frames']} ({100*config['processed_frames']/max(1,config['total_frames']):.1f}%)")
        print(f"   人脸库规模: {config['db_size']} 人")

        print(f"\n🎬 实时性能:")
        print(f"   实际帧率: {perf['fps_actual']:.2f} FPS")
        print(f"   处理帧率: {perf['fps_processed']:.2f} FPS")
        print(f"   理论帧率: {perf['fps_theoretical']:.2f} FPS")

        print(f"\n⏱️  延迟分析 (ms):")
        total = perf['latency_stages']['total']
        print(f"   总延迟: {total['mean']:.2f} ± {total['std']:.2f} (P95: {total['p95']:.2f})")

        print(f"\n🎯 识别性能:")
        print(f"   检测人脸: {recog['total_detections']}")
        print(f"   识别结果: {recog['total_recognitions']}")
        print(f"   不同人物: {recog['unique_persons']}")
        print(f"   检测率: {recog['detection_rate']:.1f}%")
        print(f"   平均置信度: {recog['confidence_stats']['mean']:.4f}")

        print(f"\n📍 跟踪性能:")
        print(f"   轨迹数量: {track['total_tracks']}")
        print(f"   平均轨迹长度: {track['avg_track_length']:.1f} 帧")
        print(f"   最长轨迹: {track['max_track_length']} 帧")
        print(f"   身份一致性: {track['identity_consistency']:.1f}%")

        print("="*100 + "\n")

    def run(self):
        """运行完整流程"""
        try:
            self.setup()

            # 1. 采样
            video_path = self.capture_benchmark()

            # 2. 标注（可选）
            if self.args.enable_annotation:
                annotations = self.annotate_video(video_path)
            else:
                print("\n⏭️  跳过标注模式 (使用 --enable-annotation 启用)")

            # 3. 生成报告
            if not self.args.skip_report:
                report = self.generate_report(video_path)
            else:
                print("\n⏭️  跳过报告生成")

            print("\n🎉 摄像头性能评估完全完成！")
            return True

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="摄像头实时人脸识别性能评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python camera_interactive.py --duration 30 --camera-id 0
  python camera_interactive.py --enable-annotation --output-report my_report.json
        """
    )

    # 采样参数
    parser.add_argument("--duration", type=int, default=30, help="采样时长（秒，默认30）")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头ID（默认0）")
    parser.add_argument("--skip-frames", type=int, default=1, help="跳帧配置（默认1）")

    # 输出参数
    parser.add_argument("--output-report", default="camera_eval/camera_benchmark_report.json", help="报告输出路径")
    parser.add_argument("--skip-report", action="store_true", help="跳过报告生成")
    parser.add_argument("--enable-annotation", action="store_true", help="启用手动标注模式")

    # 模型参数
    parser.add_argument("--db-path", default="benchmark\\YTF_100p.db", help="人脸库路径")
    parser.add_argument("--weights", default="weights\\model_best.pt", help="模型权重")
    parser.add_argument("--model-name", default="iresnet50", help="模型名称")
    parser.add_argument("--img-size", type=int, default=112, help="输入大小")
    parser.add_argument("--device", default="cuda:0", help="计算设备")

    # 检测器参数
    parser.add_argument("--det-conf-threshold", type=float, default=0.6, help="检测置信度")
    parser.add_argument("--det-min-size", type=int, default=40, help="最小检测大小")
    parser.add_argument("--detector-backend", default="mtcnn", help="检测器")
    parser.add_argument("--yolo-weights", default="", help="YOLO权重")

    # 匹配参数
    parser.add_argument("--threshold", type=float, default=0.5, help="识别阈值")
    parser.add_argument("--match-reduce", default="topk_mean", help="匹配方法")
    parser.add_argument("--topk", type=int, default=3, help="Top-K")
    parser.add_argument("--gallery-mode", default="mean", help="底库模式 (mean或all)")

    # Adaptive framework parameters
    parser.add_argument("--adaptive-mode", type=str, default=None, choices=["gaussian"], help="启用自适应阈值 ('gaussian' 或 None)")
    parser.add_argument("--enable-temporal-adaptation", action="store_true", help="启用在线学习（运行时更新阈值）")
    parser.add_argument("--temporal-learning-rate", type=float, default=0.1, help="时间适应学习率 (0.0-1.0)")

    # 跟踪参数
    parser.add_argument("--use-tracker", action="store_true", help="启用人脸跟踪")

    args = parser.parse_args()

    camera = InteractiveCamera(args)
    success = camera.run()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
