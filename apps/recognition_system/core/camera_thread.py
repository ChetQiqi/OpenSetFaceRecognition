"""
摄像头线程模块 - cv2.imshow() 弹窗显示，识别统计回传 Web 界面
"""
import threading
import time
import traceback

import cv2

from .operations import recognize_faces, draw_recognitions
from .tracker import FaceTracker


class CameraThread(threading.Thread):
    """后台摄像头线程：弹窗显示识别画面，统计数据写入 data_dict 供 Web 读取"""

    def __init__(self, camera_id, skip_frames, threshold, model, detector, gallery, data_dict,
                 stable_frames=3, mode="视频中找人", verify_target=None):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.skip_frames = skip_frames
        self.threshold = threshold
        self.model = model
        self.detector = detector
        self.gallery = gallery if gallery else []
        self.data_dict = data_dict
        self.stable_frames = stable_frames
        self.mode = mode
        self.verify_target = verify_target
        self._stop_event = threading.Event()
        self.lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        processed_count = 0
        last_results = []
        fps_time = time.time()
        fps_frames = 0
        current_fps = 0.0
        has_gallery = len(self.gallery) > 0

        # 创建人脸跟踪器（时序平滑）
        tracker = FaceTracker(
            history_size=self.stable_frames,
            min_stable_count=self.stable_frames,
            iou_threshold=0.3
        )

        print(f"摄像头启动 (id={self.camera_id}, gallery={len(self.gallery)} 人, 稳定帧数={self.stable_frames})")

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                fps_frames += 1

                # 每秒更新一次 FPS
                now = time.time()
                if now - fps_time >= 1.0:
                    current_fps = fps_frames / (now - fps_time)
                    fps_time = now
                    fps_frames = 0

                # 跳帧识别
                if frame_count % self.skip_frames == 0 and has_gallery:
                    try:
                        # 原始识别结果
                        raw_results = recognize_faces(
                            frame=frame,
                            model=self.model,
                            detector=self.detector,
                            gallery=self.gallery,
                            threshold=self.threshold,
                            match_reduce="topk_mean",
                            topk=3
                        )

                        # 使用跟踪器进行时序平滑
                        stable_results = tracker.update(raw_results)
                        last_results = stable_results
                        processed_count += 1

                        # 统计数据回传给 Web 界面
                        with self.lock:
                            for r in stable_results:
                                if r.get("accepted", False):
                                    self.data_dict["stats"][r["name"]] += 1
                            self.data_dict["results"] = last_results
                            self.data_dict["processed_frames"] = processed_count

                            # 验证ID模式：检查是否达到验证条件
                            if self.mode == "验证ID" and self.verify_target:
                                verify_count = self.data_dict["stats"].get(self.verify_target, 0)

                                # 验证通过：>=5次
                                if verify_count >= 5:
                                    self.data_dict["verify_status"] = "success"
                                    self.data_dict["verify_count"] = verify_count
                                    print(f"✅ 验证通过: {self.verify_target} ({verify_count} 次)")
                                    self._stop_event.set()
                                    break

                                # 超时失败：>=50帧仍未达到5次
                                elif processed_count >= 50:
                                    self.data_dict["verify_status"] = "failure"
                                    self.data_dict["verify_count"] = verify_count
                                    print(f"❌ 验证失败: {self.verify_target} ({verify_count} 次 < 5 次)")
                                    self._stop_event.set()
                                    break

                    except Exception as e:
                        print(f"识别失败: {e}")
                        traceback.print_exc()

                # 绘制识别结果和 HUD 信息
                display = frame.copy()
                if last_results:
                    display = draw_recognitions(display, last_results)

                hud = f"FPS:{current_fps:.1f}  Frames:{frame_count}  Processed:{processed_count}"
                cv2.putText(display, hud, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if not has_gallery:
                    cv2.putText(display, "人脸库为空，请先注册人脸",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 更新帧数统计（供 Web 显示）
                with self.lock:
                    self.data_dict["total_frames"] = frame_count
                    self.data_dict["fps"] = current_fps

                # 弹窗显示
                cv2.imshow("人脸识别 - 按 Q 退出", display)

                # 按 Q 键退出弹窗
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._stop_event.set()
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            with self.lock:
                self.data_dict["running"] = False
            print("摄像头已停止")
