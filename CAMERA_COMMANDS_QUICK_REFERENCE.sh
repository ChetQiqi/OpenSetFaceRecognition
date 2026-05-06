#!/bin/bash
# 摄像头性能评估 - 快速命令参考

# 【场景1】最小化：30秒快速采样（无标注、无图表）
echo "场景1: 快速采样 (无标注)"
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --skip-report

# 【场景2】标准工作流：采样 + 标注 + 报告
echo "场景2: 完整工作流 (采样+标注+报告)"
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --enable-annotation \
    --output-report "camera_report.json"

# 【场景3】包含追踪和优化
echo "场景3: 优化版 (启用追踪和跳帧)"
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --skip-frames 1 \
    --use-tracker \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --enable-annotation \
    --output-report "camera_report_optimized.json"

# 【场景4】仅生成图表（从已有报告）
echo "场景4: 仅生成可视化"
python generate_camera_charts.py --report "camera_report.json"

# 【场景5】完整答辩准备流程
echo "场景5: 完整答辩准备"
bash run_camera_benchmark.sh

# 【场景6】长时间采样 (60秒)
echo "场景6: 长时间采样"
python camera_interactive.py \
    --duration 60 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --use-tracker \
    --output-report "camera_report_60s.json"

# 【场景7】性能优化测试 (每3帧处理)
echo "场景7: 性能优化采样"
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --skip-frames 3 \
    --use-tracker \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --output-report "camera_report_3frame_skip.json"
