from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np

from .detector import FaceDetector
from .feature_db import FeatureDB
from .matcher import MatchResult, find_best_match
from .model import FaceEmbeddingModel


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def similarity_con(score: float) -> float:
    if score < 0.0:
        return 0.0
    if score <= 0.25:
        return score * 2.0
    if score <= 0.34:
        return 3.33 * score - 0.33
    if score < 0.6:
        return 0.38 * score + 0.67
    return 0.25 * score + 0.75


def iter_images(folder: Path) -> Iterator[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            yield path


def build_runtime(
    weights_path: str,
    model_name: str,
    img_size: int,
    device: str,
    det_conf_threshold: float = 0.90,
    det_min_size: int = 40,
    detector_backend: str = "mtcnn",
    yolo_weights: str = "",
):
    """
    构建运行时环境。

    注意：检测器后端已统一为 MTCNN。
          其他参数（detector_backend, yolo_weights）保留用于向后兼容，但不再使用。
    """
    model = FaceEmbeddingModel(
        weights_path=weights_path,
        model_name=model_name,
        img_size=img_size,
        device=device,
    )
    detector = FaceDetector(
        img_size=img_size,
        conf_threshold=det_conf_threshold,
        min_size=det_min_size,
        backend="mtcnn",  # 强制使用 MTCNN
        yolo_weights="",  # 忽略 yolo_weights
        device=device,
    )
    return model, detector


def extract_face_embedding(
    image_bgr: np.ndarray,
    model: FaceEmbeddingModel,
    detector: FaceDetector,
    box: Optional[tuple] = None,
) -> Optional[np.ndarray]:
    boxes = detector.detect(image_bgr)
    if len(boxes) == 0:
        return None
    selected_box = box if box is not None else max(boxes, key=lambda item: item[2] * item[3])
    face_rgb = detector.crop_face(image_bgr, selected_box)
    return model.embed(face_rgb)


def register_image(
    db: FeatureDB,
    model: FaceEmbeddingModel,
    detector: FaceDetector,
    person_name: str,
    image_path: Path,
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    feature = extract_face_embedding(image, model, detector)
    if feature is None:
        return False

    db.add_embedding(person_name, feature, image_path=str(image_path))
    return True


def register_dataset(
    db: FeatureDB,
    model: FaceEmbeddingModel,
    detector: FaceDetector,
    dataset_dir: Path,
    clear_first: bool = False,
    max_images_per_person: int = 0,
) -> List[Dict[str, int]]:
    results: List[Dict[str, int]] = []

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        saved = 0
        failed = 0
        processed = 0

        db.begin_transaction()
        try:
            if clear_first:
                db.clear_person_embeddings(person_name)

            for image_path in iter_images(person_dir):
                if max_images_per_person > 0 and processed >= max_images_per_person:
                    break
                processed += 1
                if register_image(db, model, detector, person_name, image_path):
                    saved += 1
                else:
                    failed += 1
            db.commit_transaction()
        except Exception:
            db.rollback_transaction()
            raise

        results.append(
            {
                "person_name": person_name,
                "processed": processed,
                "saved": saved,
                "failed": failed,
            }
        )

    return results


def load_gallery(db_path: str, gallery_mode: str):
    with FeatureDB(db_path) as db:
        return db.load_gallery(gallery_mode)


def recognize_faces(
    frame: np.ndarray,
    model: FaceEmbeddingModel,
    detector: FaceDetector,
    gallery,
    threshold: float,
    match_reduce: str,
    topk: int,
    # Adaptive framework parameters (backward compatible)
    adaptive_mode: Optional[str] = None,
    identity_stats: Optional[Dict] = None,
    feature_db=None,
):
    """
    Recognize faces in a frame with optional adaptive thresholding.

    Args:
        frame: Input image frame
        model: Face embedding model
        detector: Face detector
        gallery: Gallery of known faces
        threshold: Global threshold (used if adaptive_mode is None)
        match_reduce: Score reduction method
        topk: Top-k for aggregation
        adaptive_mode: "gaussian" for adaptive thresholding, None for fixed threshold
        identity_stats: Dictionary of per-identity statistics (required for adaptive mode)
        feature_db: Database instance for logging (optional)

    Returns:
        List of recognition results with bounding boxes and scores
    """
    results = []

    for box in detector.detect(frame):
        x, y, w, h = box
        face_rgb = detector.crop_face(frame, box)
        feature = model.embed(face_rgb)
        match: MatchResult = find_best_match(feature, gallery, reduce=match_reduce, topk=topk)
        raw_score = float(match.score)
        calibrated_score = similarity_con(raw_score)

        # ADAPTIVE MODE: Use per-identity adaptive thresholds
        if adaptive_mode == "gaussian" and identity_stats is not None:
            from .adaptive_threshold import (
                adaptive_recognize_face,
                get_all_scores_for_query,
            )

            # Get all scores for uncertainty computation
            _, all_scores = get_all_scores_for_query(
                feature, gallery, reduce=match_reduce, topk=topk
            )

            # Adaptive decision with multi-layer logic
            decision = adaptive_recognize_face(
                query_embedding=feature,
                gallery=gallery,
                identity_stats=identity_stats,
                best_match_name=match.name,
                best_match_score=raw_score,
                all_scores=all_scores,
                calibrated_score=calibrated_score,
                global_fallback=threshold,
            )

            accepted = decision.accepted
            display_name = decision.identity_name if accepted else "陌生人"

            # Log unknown detection for analysis
            if not accepted and feature_db is not None:
                feature_db.log_unknown_detection(
                    top_match_name=match.name,
                    top_match_score=raw_score,
                    rejection_reason=decision.decision_reason,
                    z_score=decision.z_score,
                    distance_ratio=decision.distance_ratio,
                )

            results.append(
                {
                    "box": box,
                    "match": match,
                    "name": decision.identity_name if accepted else "Unknown",
                    "display_name": display_name,
                    "score": calibrated_score,
                    "raw_score": raw_score,
                    "accepted": accepted,
                    "label": f"{display_name} ({calibrated_score:.3f})",
                    # Adaptive-specific fields
                    "adaptive_threshold": decision.adaptive_threshold,
                    "z_score": decision.z_score,
                    "decision_reason": decision.decision_reason,
                    "distance_ratio": decision.distance_ratio,
                }
            )

        # FIXED THRESHOLD MODE: Original logic (backward compatible)
        else:
            accepted = calibrated_score >= threshold and match.name != "Unknown"

            # 陌生人使用中文标签
            display_name = match.name if accepted else "陌生人"

            results.append(
                {
                    "box": box,
                    "match": match,
                    "name": match.name if accepted else "Unknown",  # 内部使用英文
                    "display_name": display_name,  # 显示使用中文
                    "score": calibrated_score,
                    "raw_score": raw_score,
                    "accepted": accepted,
                    "label": f"{display_name} ({calibrated_score:.3f})",
                }
            )

    return results


def _draw_label_pill(img: np.ndarray, text: str, x: int, y: int, color, font_scale: float = 0.5):
    """在框上方绘制胶囊形标签（深色背景 + 白色文字）。"""
    thickness = max(1, int(font_scale * 2.5))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = int(th * 0.55)
    rx, ry = max(0, x), max(0, y - th - pad * 2)
    rw, rh = tw + pad * 2, th + pad * 2
    # 深色半透明背景
    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (20, 20, 30), -1)
    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), color, 1)
    # 白色文字
    cv2.putText(img, text, (rx + pad, ry + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def _draw_corner_brackets(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, thickness: int):
    """绘制四角 L 形括号（现代风格识别框）。"""
    bw, bh = x2 - x1, y2 - y1
    corner = int(min(bw, bh) * 0.22)
    # 左上
    cv2.line(img, (x1, y1), (x1 + corner, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner), color, thickness)
    # 右上
    cv2.line(img, (x2, y1), (x2 - corner, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner), color, thickness)
    # 左下
    cv2.line(img, (x1, y2), (x1 + corner, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner), color, thickness)
    # 右下
    cv2.line(img, (x2, y2), (x2 - corner, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner), color, thickness)


def draw_recognitions(frame: np.ndarray, results) -> np.ndarray:
    """在帧上绘制现代风格的人脸识别结果。

    使用四角括号 + 半透明填充 + 胶囊标签替代传统粗矩形框。
    """
    annotated = frame.copy()
    overlay = annotated.copy()
    h_img, w_img = annotated.shape[:2]
    line_thickness = max(2, min(4, min(h_img, w_img) // 300))

    for item in results:
        x, y, w, h = item["box"]
        x2, y2 = x + w, y + h
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(annotated.shape[1], x2)
        y2 = min(annotated.shape[0], y2)

        if item["accepted"]:
            color = (100, 230, 140)  # 柔和翠绿
            fill_color = (45, 100, 65)
        else:
            color = (100, 100, 240)  # 柔和红紫
            fill_color = (80, 35, 50)

        # 半透明填充（先画在 overlay 上，稍后混合）
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)

        # 四角括号
        _draw_corner_brackets(annotated, x1, y1, x2, y2, color, line_thickness)

        # 胶囊标签
        label = item.get("label", "")
        if label:
            _draw_label_pill(annotated, label, x1, y1 - 4, color)

    # 混合半透明填充
    cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)

    return annotated