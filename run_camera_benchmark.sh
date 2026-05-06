#!/bin/bash
# 摄像头实时人脸识别性能评估 - 完整工作流脚本
# 用途: 30秒采样 -> 人工标注 -> 生成报告

echo "=================================================="
echo "📹 摄像头实时人脸识别系统性能评估"
echo "=================================================="
echo ""

# 脚本配置
DURATION=30              # 采样时长（秒）
CAMERA_ID=0              # 摄像头ID
SKIP_FRAMES=2            # 跳帧配置（1=每帧，3=每3帧）
DB_PATH="benchmark\\YTF_100p.db"        # 人脸库路径
WEIGHTS_PATH="weights\\model_best.pt"   # 模型权重路径
OUTPUT_DIR="camera_eval"  # 输出文件夹
OUTPUT_REPORT="${OUTPUT_DIR}/camera_benchmark_report.json"  # 输出报告
ENABLE_ANNOTATION="--enable-annotation"  # 启用手动标注
USE_TRACKER="--use-tracker"             # 启用人脸追踪

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "⚙️  运行配置:"
echo "   采样时长: ${DURATION}秒"
echo "   摄像头ID: ${CAMERA_ID}"
echo "   跳帧策略: ${SKIP_FRAMES}"
echo "   人脸库: ${DB_PATH}"
echo "   模型权重: ${WEIGHTS_PATH}"
echo "   输出目录: ${OUTPUT_DIR}"
echo ""

# ======================================================================
# 第1步：采样 + 标注 + 报告生成
# ======================================================================
echo "【第1步】🎬 开始摄像头采样、标注、报告生成..."
echo "======================================================================="

python camera_interactive.py \
    --duration ${DURATION} \
    --camera-id ${CAMERA_ID} \
    --skip-frames ${SKIP_FRAMES} \
    --db-path "${DB_PATH}" \
    --weights "${WEIGHTS_PATH}" \
    --model-name "iresnet50" \
    --img-size 112 \
    --device "cuda:0" \
    --det-conf-threshold 0.6 \
    --det-min-size 40 \
    --detector-backend "mtcnn" \
    --threshold 0.5 \
    --match-reduce "topk_mean" \
    --topk 3 \
    --gallery-mode "mean" \
    --output-report "${OUTPUT_REPORT}" \
    ${ENABLE_ANNOTATION} \
    ${USE_TRACKER}

RESULT=$?

if [ $RESULT -ne 0 ]; then
    echo ""
    echo "❌ 采样过程失败，退出"
    exit 1
fi

echo ""
echo "✅ 采样完成！"
echo ""

# ======================================================================
# 完成
# ======================================================================
echo "======================================================================="
echo "🎉 摄像头性能评估完全完成！"
echo "======================================================================="
echo ""
echo "📁 生成的文件:"
echo "   📹 演示视频: camera_demo_*.mp4"
echo "   📄 性能报告: ${OUTPUT_DIR}/camera_benchmark_report.json"
echo "   📊 性能表格: ${OUTPUT_DIR}/performance_metrics.csv"
echo ""
echo "======================================================================="
echo ""

