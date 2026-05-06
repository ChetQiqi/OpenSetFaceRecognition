#!/usr/bin/env python3
"""
摄像头丢帧诊断脚本
检查采集速度是否跟上摄像头输出
"""

import cv2
import time
import numpy as np

def diagnose_camera():
    """诊断摄像头采集性能"""
    print("=" * 80)
    print("📹 摄像头采集诊断")
    print("=" * 80)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    # 获取摄像头参数
    fps_camera = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"🎥 摄像头参数:")
    print(f"   FPS: {fps_camera}")
    print(f"   分辨率: {width}x{height}")
    print(f"   期望每帧耗时: {1000/fps_camera:.2f} ms")
    print()

    # 采样10秒，测量采集速度
    print("⏳ 正在采样10秒...")
    print()

    frame_count = 0
    total_read_time = 0
    max_read_time = 0
    min_read_time = float('inf')

    start_time = time.time()

    while time.time() - start_time < 10:
        t0 = time.time()
        ok, frame = cap.read()
        read_time = (time.time() - t0) * 1000  # ms

        if not ok:
            continue

        frame_count += 1
        total_read_time += read_time
        max_read_time = max(max_read_time, read_time)
        min_read_time = min(min_read_time, read_time)

    cap.release()

    elapsed = time.time() - start_time
    avg_read_time = total_read_time / max(1, frame_count)
    actual_fps = frame_count / elapsed

    print("📊 采集统计:")
    print(f"   实际采集帧数: {frame_count} 帧")
    print(f"   实际采集FPS: {actual_fps:.2f} FPS")
    print(f"   期望采集FPS: {fps_camera}")
    print()

    print("⏱️ 单帧采集耗时:")
    print(f"   平均: {avg_read_time:.2f} ms")
    print(f"   最小: {min_read_time:.2f} ms")
    print(f"   最大: {max_read_time:.2f} ms")
    print()

    # 诊断
    print("🔍 诊断结果:")

    if actual_fps >= fps_camera * 0.9:
        print("   ✅ 采集性能良好 (跟上摄像头输出)")
    else:
        loss_rate = 100 * (1 - actual_fps/fps_camera)
        print(f"   ⚠️ 采集丢帧 ({loss_rate:.1f}%)")
        print(f"      原因: 采集太慢 (实际 {actual_fps:.2f} vs 期望 {fps_camera})")

    print()

    # 检查处理时间
    if avg_read_time > 1000/fps_camera * 0.5:
        print("⚠️  摄像头驱动可能有问题:")
        print(f"   单帧采集耗时 {avg_read_time:.2f}ms > {1000/fps_camera*.5:.2f}ms")

    print()
    print("=" * 80)

if __name__ == "__main__":
    diagnose_camera()
