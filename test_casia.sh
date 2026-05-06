# python run_complete_benchmark.py \
#   --casia-dir "F:\\Dataset\\CASIA-WebFace" \
#   --num-persons 20 \
#   --train-ratio 0.7 \
#   --skip-camera

# 使用已有数据库，快速重测
# python run_benchmark_flexible.py \
#   --mode test-only \
#   --db-path E:\\MyRecoCode\\benchmark_casia\\face_features.db \
#   --test-dir E:\\MyRecoCode\\benchmark_casia\\test \
#   --skip-camera

# 没有数据库
# python run_benchmark_flexible.py \
#     --mode full \
#     --casia-dir "F:\\Dataset\\CASIA-WebFace" \
#     --num-persons 100 \
#     --weights "E:\\MyRecoCode\\weights\\adasin_best.pt" \
#     --output-dir "benchmark_casia" \
#     --skip-camera

# python merge_benchmark_results.py \
#   --benchmark-json benchmark_casia/benchmark_results.json \
#   --output benchmark_casia/performance_table.tex

python -m apps.recognition_system.core.cli list-persons \
  --db-path benchmark\feat_casia.db