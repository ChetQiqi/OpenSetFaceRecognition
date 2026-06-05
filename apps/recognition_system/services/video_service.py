import base64
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2

from apps.recognition_system.core.operations import draw_recognitions
from apps.recognition_system.core.tracker import FaceTracker
from apps.recognition_system.services.inference_service import InferenceService


class VideoService:
    """Service 层：视频文件处理和摄像头任务编排。"""

    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
        self._camera_thread = None
        self._camera_data = None

    def recognize_video(
        self,
        video_bytes: bytes,
        filename: str,
        skip_frames: int = 5,
        threshold: float = 0.45,
        stable_frames: int = 3,
        mode: str = "视频中找人",
        verify_target: Optional[str] = None,
    ) -> Dict[str, object]:
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(video_bytes)
            input_path = Path(tmp_file.name)

        try:
            return self._process_video_file(
                input_path=input_path,
                output_name=filename,
                skip_frames=skip_frames,
                threshold=threshold,
                stable_frames=stable_frames,
                mode=mode,
                verify_target=verify_target,
            )
        finally:
            input_path.unlink(missing_ok=True)

    def start_camera(
        self,
        camera_id: int = 0,
        skip_frames: int = 3,
        threshold: float = 0.45,
        stable_frames: int = 3,
        mode: str = "视频中找人",
        verify_target: Optional[str] = None,
    ) -> Dict[str, object]:
        from apps.recognition_system.core.camera_thread import CameraThread

        if self._camera_thread is not None and self._camera_thread.is_alive():
            return {"running": True, "message": "摄像头已在运行"}

        self.inference_service.model_service.load()
        self._camera_data = {
            "results": [],
            "stats": defaultdict(int),
            "total_frames": 0,
            "processed_frames": 0,
            "fps": 0.0,
            "running": True,
            "verify_status": None,
            "verify_count": 0,
        }
        self._camera_thread = CameraThread(
            camera_id=camera_id,
            skip_frames=skip_frames,
            threshold=threshold,
            model=self.inference_service.model_service.model,
            detector=self.inference_service.model_service.detector,
            gallery=self.inference_service.model_service.gallery,
            data_dict=self._camera_data,
            stable_frames=stable_frames,
            mode=mode,
            verify_target=verify_target,
        )
        self._camera_thread.start()
        return {"running": True, "message": "摄像头已启动"}

    def stop_camera(self) -> Dict[str, object]:
        if self._camera_thread is not None:
            self._camera_thread.stop()
            self._camera_thread = None
        if self._camera_data is not None:
            self._camera_data["running"] = False
        return {"running": False, "message": "摄像头已停止"}

    def clear_camera(self) -> Dict[str, object]:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return {"running": True, "message": "摄像头运行中，请先停止再清空"}
        self._camera_data = None
        return {"running": False, "message": "摄像头数据已清空"}

    def camera_status(self) -> Dict[str, object]:
        if self._camera_thread is not None and not self._camera_thread.is_alive():
            self._camera_thread = None
            if self._camera_data is not None:
                self._camera_data["running"] = False

        data = self._camera_data or {
            "results": [],
            "stats": {},
            "total_frames": 0,
            "processed_frames": 0,
            "fps": 0.0,
            "running": False,
            "verify_status": None,
            "verify_count": 0,
        }
        raw_results = data.get("results", [])
        return {
            "running": bool(data.get("running", False)),
            "results": [InferenceService._serialize_result(item) for item in raw_results],
            "stats": dict(data.get("stats", {})),
            "total_frames": int(data.get("total_frames", 0)),
            "processed_frames": int(data.get("processed_frames", 0)),
            "fps": float(data.get("fps", 0.0)),
            "verify_status": data.get("verify_status"),
            "verify_count": int(data.get("verify_count", 0)),
            "person_cards": self.inference_service.build_person_cards(raw_results),
        }

    def _process_video_file(
        self,
        input_path: Path,
        output_name: str,
        skip_frames: int,
        threshold: float,
        stable_frames: int,
        mode: str,
        verify_target: Optional[str],
    ) -> Dict[str, object]:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_output = Path(tempfile.gettempdir()) / f"temp_{int(time.time())}.mp4"
        output_path = Path(tempfile.gettempdir()) / f"recognized_{int(time.time())}_{Path(output_name).name}"
        writer = cv2.VideoWriter(
            str(temp_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        tracker = FaceTracker(
            history_size=stable_frames,
            min_stable_count=stable_frames,
            iou_threshold=0.3,
        )

        person_counts: Dict[str, int] = {}
        stranger_count = 0
        frame_idx = 0
        processed_frames = 0
        last_results = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % max(1, skip_frames) == 0:
                    raw_results = self.inference_service._recognize_frame_raw(frame, threshold=threshold)
                    stable_results = tracker.update(raw_results)
                    last_results = stable_results
                    processed_frames += 1

                    for item in stable_results:
                        if item.get("accepted", False):
                            name = item["name"]
                            person_counts[name] = person_counts.get(name, 0) + 1
                        else:
                            stranger_count += 1

                display_frame = draw_recognitions(frame, last_results) if last_results else frame
                writer.write(display_frame)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        self._transcode_for_browser(temp_output, output_path)
        video_bytes = output_path.read_bytes()

        verify_result = None
        if mode == "验证ID" and verify_target:
            verify_frames = person_counts.get(verify_target, 0)
            verify_result = {
                "target": verify_target,
                "passed": verify_frames >= 10,
                "frames": verify_frames,
            }

        person_cards = self.inference_service.build_person_cards([
            {"name": name, "accepted": True, "score": 1.0}
            for name in person_counts.keys()
        ], score_key="score")

        return {
            "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
            "output_filename": output_path.name,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "written_frames": frame_idx,
            "fps": fps,
            "width": width,
            "height": height,
            "person_counts": person_counts,
            "stranger_count": stranger_count,
            "verify_result": verify_result,
            "person_cards": person_cards,
        }

    @staticmethod
    def _transcode_for_browser(temp_output: Path, output_path: Path) -> None:
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(temp_output),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                shutil.copy(str(temp_output), str(output_path))
        except Exception:
            shutil.copy(str(temp_output), str(output_path))
        finally:
            temp_output.unlink(missing_ok=True)
