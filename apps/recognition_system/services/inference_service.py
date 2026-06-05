import base64
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from apps.recognition_system.config import AppConfig
from apps.recognition_system.core.operations import draw_recognitions, recognize_faces
from apps.recognition_system.models import ModelService


class InferenceService:
    """Service 层：图片和单帧识别编排。"""

    def __init__(self, config: AppConfig, model_service: ModelService):
        self.config = config
        self.model_service = model_service

    def recognize_image(
        self,
        image_bytes: bytes,
        threshold: Optional[float] = None,
        draw: bool = True,
    ) -> Dict[str, object]:
        frame = self._decode_image(image_bytes)
        if frame is None:
            raise ValueError("图片解码失败")

        results = self.recognize_frame(frame, threshold=threshold)
        response: Dict[str, object] = {"results": results}

        if draw:
            raw_results = self._recognize_frame_raw(frame, threshold=threshold)
            annotated = draw_recognitions(frame, raw_results)
            ok, buffer = cv2.imencode(".jpg", annotated)
            if ok:
                response["annotated_image_base64"] = base64.b64encode(buffer.tobytes()).decode("utf-8")
            response["matched_pairs"] = self._build_matched_pairs(frame, raw_results)
            response["person_cards"] = self.build_person_cards(raw_results)

        return response

    def recognize_frame(self, frame: np.ndarray, threshold: Optional[float] = None) -> List[Dict[str, object]]:
        raw_results = self._recognize_frame_raw(frame, threshold=threshold)
        return [self._serialize_result(item) for item in raw_results]

    def _recognize_frame_raw(self, frame: np.ndarray, threshold: Optional[float] = None):
        self.model_service.load()
        return recognize_faces(
            frame=frame,
            model=self.model_service.model,
            detector=self.model_service.detector,
            gallery=self.model_service.gallery,
            threshold=self.config.recognition_threshold if threshold is None else threshold,
            match_reduce=self.config.match_reduce,
            topk=self.config.topk,
        )

    @staticmethod
    def _decode_image(content: bytes):
        array = np.frombuffer(content, dtype=np.uint8)
        if array.size == 0:
            return None
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    @staticmethod
    def _serialize_result(item) -> Dict[str, object]:
        box = item.get("box", (0, 0, 0, 0))
        match = item.get("match")
        return {
            "box": [int(v) for v in box],
            "name": str(item.get("name", "Unknown")),
            "display_name": str(item.get("display_name", item.get("name", "Unknown"))),
            "score": float(item.get("score", 0.0)),
            "raw_score": float(item.get("raw_score", item.get("score", 0.0))),
            "accepted": bool(item.get("accepted", False)),
            "label": str(item.get("label", "")),
            "support": int(getattr(match, "support", 0)) if match is not None else 0,
        }

    def _build_matched_pairs(self, frame: np.ndarray, raw_results) -> List[Dict[str, str]]:
        pairs: List[Dict[str, str]] = []
        repo = self.model_service.repository
        for item in raw_results:
            if not item.get("accepted"):
                continue
            name = str(item.get("name", ""))
            if not name or name == "Unknown":
                continue

            box = item.get("box", (0, 0, 0, 0))
            x, y, w, h = [int(v) for v in box]
            h_img, w_img = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + max(0, w))
            y2 = min(h_img, y + max(0, h))
            if x2 <= x1 or y2 <= y1:
                continue

            # 被识别图展示原图，并仅标注当前命中的人脸框。
            # 先统一输出尺寸再绘制，避免不同原图分辨率导致标签大小不一致。
            query_annotated, scale = self._resize_for_pair_display(frame)
            rx1 = int(round(x1 * scale))
            ry1 = int(round(y1 * scale))
            rx2 = int(round(x2 * scale))
            ry2 = int(round(y2 * scale))
            # 现代风格：四角括号 + 半透明填充 + 胶囊标签
            color = (100, 230, 140)
            line_thickness = 3
            bw, bh = rx2 - rx1, ry2 - ry1
            corner = int(min(bw, bh) * 0.22)
            # 半透明填充
            overlay = query_annotated.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (45, 100, 65), -1)
            cv2.addWeighted(overlay, 0.12, query_annotated, 0.88, 0, query_annotated)
            # 四角括号
            cv2.line(query_annotated, (rx1, ry1), (rx1 + corner, ry1), color, line_thickness)
            cv2.line(query_annotated, (rx1, ry1), (rx1, ry1 + corner), color, line_thickness)
            cv2.line(query_annotated, (rx2, ry1), (rx2 - corner, ry1), color, line_thickness)
            cv2.line(query_annotated, (rx2, ry1), (rx2, ry1 + corner), color, line_thickness)
            cv2.line(query_annotated, (rx1, ry2), (rx1 + corner, ry2), color, line_thickness)
            cv2.line(query_annotated, (rx1, ry2), (rx1, ry2 - corner), color, line_thickness)
            cv2.line(query_annotated, (rx2, ry2), (rx2 - corner, ry2), color, line_thickness)
            cv2.line(query_annotated, (rx2, ry2), (rx2, ry2 - corner), color, line_thickness)
            # 胶囊标签
            label = str(item.get("display_name", name))
            font_scale = 0.5
            t_thick = max(1, int(font_scale * 2.5))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, t_thick)
            pad = int(th * 0.55)
            lx, ly = max(0, rx1), max(0, ry1 - th - pad * 2)
            cv2.rectangle(query_annotated, (lx, ly), (lx + tw + pad * 2, ly + th + pad * 2), (20, 20, 30), -1)
            cv2.rectangle(query_annotated, (lx, ly), (lx + tw + pad * 2, ly + th + pad * 2), color, 1)
            cv2.putText(query_annotated, label, (lx + pad, ly + th + pad),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), t_thick)
            ok, qbuf = cv2.imencode(".jpg", query_annotated)
            if not ok:
                continue
            query_image_base64 = base64.b64encode(qbuf.tobytes()).decode("utf-8")

            gallery_face_base64 = None
            gallery = self._load_person_gallery_image(name)
            if gallery is not None:
                # 图库图仅输出人脸区域（优先检测最大人脸）
                gallery_face = None
                try:
                    face_boxes = self.model_service.detector.detect(gallery)
                    if face_boxes:
                        gx, gy, gw, gh = max(face_boxes, key=lambda b: int(b[2]) * int(b[3]))
                        gh_img, gw_img = gallery.shape[:2]
                        gx1 = max(0, int(gx))
                        gy1 = max(0, int(gy))
                        gx2 = min(gw_img, int(gx + gw))
                        gy2 = min(gh_img, int(gy + gh))
                        if gx2 > gx1 and gy2 > gy1:
                            gallery_face = gallery[gy1:gy2, gx1:gx2]
                except Exception:
                    gallery_face = None

                if gallery_face is None:
                    # 检测失败时不返回整图，做中心裁切，尽量保持“只看人脸”的效果
                    gh_img, gw_img = gallery.shape[:2]
                    side = max(1, min(gw_img, gh_img))
                    cx, cy = gw_img // 2, gh_img // 2
                    gx1 = max(0, cx - side // 2)
                    gy1 = max(0, cy - side // 2)
                    gx2 = min(gw_img, gx1 + side)
                    gy2 = min(gh_img, gy1 + side)
                    gallery_face = gallery[gy1:gy2, gx1:gx2]

                gok, gbuf = cv2.imencode(".jpg", gallery_face)
                if gok:
                    gallery_face_base64 = base64.b64encode(gbuf.tobytes()).decode("utf-8")

            pair = {
                "name": name,
                "query_image_base64": query_image_base64,
            }
            if gallery_face_base64:
                pair["gallery_face_base64"] = gallery_face_base64
            else:
                pair["gallery_error"] = f"未找到或无法读取 {name} 的图库原图"
            pairs.append(pair)
        return pairs

    def _load_person_gallery_image(self, person_name: str):
        repo = self.model_service.repository
        db_path = Path(repo.db_path).resolve()
        gallery_base = db_path.parent / "gallery_images"

        # 1) 先尝试数据库里记录的路径（新老数据都可能有）
        candidate_paths = []
        latest = repo.get_person_latest_image_path(person_name)
        if latest:
            candidate_paths.append(latest)
        candidate_paths.extend(repo.list_person_image_paths(person_name))

        for raw in candidate_paths:
            path_obj = Path(raw)
            # 绝对路径
            if path_obj.is_absolute() and path_obj.exists():
                img = self._read_image_file(path_obj)
                if img is not None:
                    return img
            # 相对文件名（旧数据常见）
            rel_candidate = gallery_base / person_name / path_obj.name
            if rel_candidate.exists():
                img = self._read_image_file(rel_candidate)
                if img is not None:
                    return img
            # 再兜底：按文件名在 gallery_images 下递归查找（兼容历史迁移后目录变化）
            if path_obj.name:
                for any_candidate in gallery_base.rglob(path_obj.name):
                    if any_candidate.is_file():
                        img = self._read_image_file(any_candidate)
                        if img is not None:
                            return img

        # 2) 兜底：扫描该人员目录中的常见图片
        person_dir = gallery_base / person_name
        if person_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for img_path in sorted(person_dir.glob(ext)):
                    img = self._read_image_file(img_path)
                    if img is not None:
                        return img
        return None

    @staticmethod
    def _resize_for_pair_display(image: np.ndarray, target_long_side: int = 640):
        h, w = image.shape[:2]
        long_side = max(h, w)
        if long_side <= 0:
            return image.copy(), 1.0
        scale = target_long_side / float(long_side)
        resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
        return resized, scale

    def build_person_cards(self, raw_results, score_key: str = "score") -> List[Dict[str, object]]:
        """从识别结果构建人物资料卡列表（去重，每人一张卡）。"""
        cards: List[Dict[str, object]] = []
        seen: set = set()
        repo = self.model_service.repository

        for item in raw_results:
            name = str(item.get("name", "Unknown"))
            if name == "Unknown" or not name:
                continue
            if name in seen:
                continue
            seen.add(name)

            detail = repo.get_person_detail(name)
            embedding_count = detail[1] if detail else 0
            gender = detail[2] if detail else "unspecified"
            birth_date = detail[3] if detail else ""

            score = float(item.get(score_key, 0.0))
            accepted = bool(item.get("accepted", False))

            gallery_face_base64 = None
            gallery_img = self._load_person_gallery_image(name)
            if gallery_img is not None:
                gok, gbuf = cv2.imencode(".jpg", gallery_img)
                if gok:
                    gallery_face_base64 = base64.b64encode(gbuf.tobytes()).decode("utf-8")

            cards.append({
                "name": name,
                "gender": gender,
                "birth_date": birth_date,
                "embedding_count": embedding_count,
                "gallery_face_base64": gallery_face_base64,
                "score": score,
                "accepted": accepted,
            })

        return cards

    @staticmethod
    def _read_image_file(path: Path):
        """Windows 下 cv2.imread 对中文/特殊路径不稳定，优先使用 fromfile 解码。"""
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size > 0:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is not None:
                    return image
        except Exception:
            pass
        return cv2.imread(str(path))
