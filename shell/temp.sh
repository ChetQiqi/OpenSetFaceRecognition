# # 完整数据库，600个known persons
# python prepare_open_set_optimized.py --db full --num-known 600

# # 自定义gallery/test比例
# python prepare_open_set_optimized.py --db full --num-known 400 --gallery-size 20 --test-size 30

# # 自定义输出目录
# python prepare_open_set_optimized.py \
#     --db full \
#     --num-known 400 \
#     --output-dir benchmark/open_set_data_full

# # 运行评估
# python scripts/compare_with_prepared_data.py \
#     --data-dir benchmark/open_set_data_full

# python prepare_open_set_optimized.py \
#     --db small \
#     --output-dir benchmark/open_set_data_small

# python extract_casia_features.py \
#     --dataset-path F:\\Dataset\\CASIA-WebFace \
#     --num-ids 200 \
#     --output benchmark/CASIA_200_features.db \
#     --device cuda:0

# python prepare_open_set_optimized.py \
#     --db casia \
#     --num-known 100

python scripts/compare_with_prepared_data.py \
    --data-dir benchmark/open_set_data_full \
    --output-dir evaluation_results_YTF_full \
    --fixed-threshold 0.45