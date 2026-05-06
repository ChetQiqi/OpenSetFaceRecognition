"""
人脸跟踪器 - 时序平滑机制，避免短暂误识别
"""
from collections import deque, Counter
from typing import List, Dict, Optional, Tuple
import numpy as np


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """计算两个边界框的IoU (Intersection over Union)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class TrackedFace:
    """单个被跟踪的人脸"""

    def __init__(self, track_id: int, box: Tuple[int, int, int, int],
                 name: str, score: float, history_size: int = 5):
        self.track_id = track_id
        self.box = box
        self.history = deque(maxlen=history_size)  # 保存最近N帧的识别结果
        self.history.append((name, score))
        self.frames_missing = 0
        self.max_missing = 3  # 最多丢失3帧还保留跟踪

    def update(self, box: Tuple[int, int, int, int], name: str, score: float):
        """更新跟踪信息"""
        self.box = box
        self.history.append((name, score))
        self.frames_missing = 0

    def get_stable_identity(self, min_count: int = 3) -> Tuple[str, float]:
        """
        获取稳定的身份识别结果

        Args:
            min_count: 最少出现次数才认为稳定

        Returns:
            (name, average_score)
        """
        if len(self.history) == 0:
            return "Unknown", 0.0

        # 统计每个名字的出现次数
        name_counter = Counter(name for name, _ in self.history)

        # 找出现次数最多的名字
        most_common_name, count = name_counter.most_common(1)[0]

        # 如果出现次数不够，标记为不稳定
        if count < min_count:
            return "Unknown", 0.0

        # 计算该名字的平均分数
        scores = [score for name, score in self.history if name == most_common_name]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return most_common_name, avg_score

    def mark_missing(self):
        """标记该帧未检测到"""
        self.frames_missing += 1

    def is_lost(self) -> bool:
        """是否已经丢失跟踪"""
        return self.frames_missing > self.max_missing


class FaceTracker:
    """
    人脸跟踪器 - 跨帧跟踪人脸并进行时序平滑

    Args:
        history_size: 保存最近N帧的识别历史
        min_stable_count: 至少N帧识别为同一ID才认为稳定
        iou_threshold: IoU阈值，用于匹配同一人脸
    """

    def __init__(self, history_size: int = 5, min_stable_count: int = 3,
                 iou_threshold: float = 0.3):
        self.tracked_faces: List[TrackedFace] = []
        self.next_track_id = 0
        self.history_size = history_size
        self.min_stable_count = min_stable_count
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新跟踪器并返回稳定的识别结果

        Args:
            detections: 当前帧的检测结果列表

        Returns:
            稳定后的识别结果列表
        """
        # 标记所有已跟踪人脸为"本帧缺失"
        for tracked in self.tracked_faces:
            tracked.mark_missing()

        # 匹配当前检测结果到已跟踪的人脸
        matched_tracks = set()
        stable_results = []

        for detection in detections:
            box = detection['box']
            name = detection['name']
            score = detection['score']

            # 尝试匹配到已有的跟踪
            best_match = None
            best_iou = 0.0

            for tracked in self.tracked_faces:
                if tracked.track_id in matched_tracks:
                    continue

                iou = calculate_iou(box, tracked.box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_match = tracked

            if best_match is not None:
                # 更新已有跟踪
                best_match.update(box, name, score)
                matched_tracks.add(best_match.track_id)
                tracked_face = best_match
            else:
                # 创建新的跟踪
                new_track = TrackedFace(
                    track_id=self.next_track_id,
                    box=box,
                    name=name,
                    score=score,
                    history_size=self.history_size
                )
                self.next_track_id += 1
                self.tracked_faces.append(new_track)
                tracked_face = new_track

            # 获取稳定的身份
            stable_name, stable_score = tracked_face.get_stable_identity(self.min_stable_count)

            # 判断是否被接受
            accepted = stable_name != "Unknown"
            display_name = stable_name if accepted else "陌生人"

            # 构建稳定的结果
            stable_result = {
                "box": box,
                "name": stable_name,
                "display_name": display_name,
                "score": stable_score,
                "accepted": accepted,
                "label": f"{display_name} ({stable_score:.3f})",
                "track_id": tracked_face.track_id,
                "match": detection.get("match"),  # 保留原始match信息
                "raw_score": detection.get("raw_score", score),
            }
            stable_results.append(stable_result)

        # 移除丢失的跟踪
        self.tracked_faces = [t for t in self.tracked_faces if not t.is_lost()]

        return stable_results

    def reset(self):
        """重置跟踪器"""
        self.tracked_faces.clear()
        self.next_track_id = 0
