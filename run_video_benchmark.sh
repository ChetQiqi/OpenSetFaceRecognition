#!/bin/bash
# 视频人脸识别系统性能评估 - 快速运行脚本
# 支持YouTube Faces数据集结构

IMAGES_DIR="G:\\YTF_dataset\\OpenDataLab___YouTube_Faces\\raw\\data\\YouTubeFaces\\aligned_images_DB"
VIDEOS_DIR="G:\\YTF_dataset\\OpenDataLab___YouTube_Faces\\raw\\data\\YouTubeFaces\\videos"

echo "========================================"
echo "🚀 视频人脸识别系统性能评估"
echo "========================================"
echo ""

# 检查目录是否存在
if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ 错误: 找不到 aligned_images_DB 目录"
    echo "路径: $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$VIDEOS_DIR" ]; then
    echo "❌ 错误: 找不到 videos 目录"
    echo "路径: $VIDEOS_DIR"
    exit 1
fi

echo "✅ 目录检查通过"
echo "📁 图像目录: $IMAGES_DIR"
echo "📁 视频目录: $VIDEOS_DIR"
echo ""

# 标准评估
echo "📊 运行标准评估（阈值0.45，启用时序平滑）..."
echo ""
#---------------------------------------------
# 如果要重新生成db 就删除 --skip-register 参数
#---------------------------------------------

python video_benchmark.py \
    --videos-dir "$VIDEOS_DIR" \
    --db-path benchmark\\YTF_100p.db \
    --use-tracker \
    --skip-frames 3 \
    --only-in-gallery \
    --output-dir results_with_tracker \
    --skip-register


# # 限制测试视频数量
#     python video_benchmark.py \
#     --videos-dir "$VIDEOS_DIR" \
#     --db-path benchmark\\YTF_100p.db \
#     --use-tracker \
#     --skip-frames 3 \
#     --only-in-gallery \
#     --max-videos 100 \
#     --output-dir results_only_in_gallery

# # 既过滤库人员又限制数量
#     python video_benchmark.py \
#     --videos-dir "$VIDEOS_DIR" \
#     --db-path benchmark\\YTF_100p.db \
#     --use-tracker \
#     --skip-frames 3 \
#     --only-in-gallery \
#     --max-videos 100 \
#     --output-dir results_filtered

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 评估完成！"
    echo "========================================"
    echo ""
    echo "📁 结果保存在: video_benchmark_results/"
    echo ""
    echo "📄 查看报告:"
    echo "   - JSON:     video_benchmark_results/video_benchmark_results.json"
    echo "   - Markdown: video_benchmark_results/video_benchmark_report.md"
    echo "   - LaTeX:    video_benchmark_results/video_benchmark_table.tex"
    echo ""
else
    echo ""
    echo "❌ 评估失败，请检查错误信息"
    exit 1
fi
