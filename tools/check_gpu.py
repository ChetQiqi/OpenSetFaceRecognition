
# import torch
# from apps.recognition_system.core.operations import build_runtime

# print("=" * 60)
# print("🎮 GPU使用检查")
# print("=" * 60)

# # 检查CUDA可用性
# print(f"\n✓ CUDA可用: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"  设备: {torch.cuda.get_device_name(0)}")
#     print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# # 加载模型
# print("\n⏳ 加载模型...")
# model, detector = build_runtime(
#     weights_path="weights\\model_best.pt",
#     model_name="iresnet50",
#     img_size=112,
#     device="cuda:0",
#     det_conf_threshold=0.6,
#     det_min_size=40,
#     detector_backend="mtcnn",
#     yolo_weights=""
# )

# # 检查模型在哪个设备
# import inspect
# try:
#     model_device = next(model.parameters()).device
#     print(f"\n✓ 人脸模型设备: {model_device}")
# except:
#     print("\n✗ 无法确定模型设备")

# # 测试一次前向传播
# import cv2
# import numpy as np

# print("\n⏳ 测试前向传播...")
# import time

# # 创建虚拟输入
# dummy_input = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

# t0 = time.time()
# try:
#     from apps.recognition_system.core.operations import extract_face_embedding
#     feature = extract_face_embedding(dummy_input, model, detector)
#     elapsed = (time.time() - t0) * 1000
#     print(f"✓ 处理耗时: {elapsed:.2f} ms")
#     if elapsed > 100:
#         print("  ⚠️ 耗时过长，可能没有GPU加速")
# except Exception as e:
#     print(f"✗ 错误: {e}")

# print("\n" + "=" * 60)

import cv2

# 打开摄像头（通常 0 是内置摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
else:
    # 获取常见参数
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # 宽度
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 高度
    fps = cap.get(cv2.CAP_PROP_FPS)             # 帧率
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)       # 编码格式

    print(f"--- 摄像头当前设置 ---")
    print(f"分辨率: {int(width)} x {int(height)}")
    print(f"设定帧率: {fps}")
    
    # 解码 FOURCC 编码（如 MJPG, YUY2）
    codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
    print(f"当前编码格式: {codec}")

cap.release()