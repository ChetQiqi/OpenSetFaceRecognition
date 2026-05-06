# 📁 RecognitionSystem 项目结构说明

## 🎯 项目概述

人脸识别系统，包含实时相机识别和开放集识别评估框架。

---

## 📂 目录结构

```
RecognitionSystem/
├── 📂 apps/                          # 核心应用代码
│   └── recognition_system/
│       └── core/
│           ├── adaptive_threshold.py  # ✨ 自适应阈值框架
│           ├── detector.py            # 人脸检测器 (MTCNN)
│           ├── feature_db.py          # SQLite特征数据库
│           ├── matcher.py             # 特征匹配器
│           ├── model.py               # 特征提取模型 (iResNet50)
│           └── operations.py          # 核心操作 (已删除similarity_con)
│
├── 📂 scripts/                       # 实验与评估脚本
│   ├── extract_casia_features.py     # 从CASIA-WebFace提取特征
│   ├── prepare_open_set_optimized.py # 准备开放集数据(支持YTF/CASIA)
│   └── compare_with_prepared_data.py # 固定 vs 自适应对比评估
│
├── 📂 tools/                         # 分析和辅助工具
│   ├── analyze_genuine_variance.py   # 分析genuine样本方差
│   ├── analyze_threshold_roc.py      # ROC曲线分析
│   ├── check_db.py                   # 数据库检查工具
│   ├── check_gpu.py                  # GPU检查工具
│   ├── combine_images.py             # 图像合并工具
│   ├── generate_camera_charts.py     # 生成相机性能图表
│   ├── generate_latency_report.py    # 生成延迟报告
│   ├── latency_analysis.py           # 延迟分析
│   ├── merge_benchmark_results.py    # 合并benchmark结果
│   ├── plot_normalized_only.py       # 绘制归一化分布
│   ├── register_random_persons.py    # 随机注册persons
│   ├── split_dataset_for_eval.py     # 分割数据集
│   ├── verify_roc_with_images.py     # ROC验证工具
│   └── verify_ytf_database.py        # YTF数据库验证
│
├── 📂 benchmark/                     # 数据库和评估数据
│   ├── YTF_100p.db                   # YTF小数据库 (103 persons, Web演示用)
│   ├── YTF_allID_50features.db       # YTF完整数据库 (1595 persons, 论文用)
│   ├── CASIA_200_features.db         # CASIA数据库 (200 persons, 实验用)
│   └── open_set_data/                # 开放集评估数据
│       ├── gallery.npy
│       ├── test_known.npy
│       ├── test_unknown.npy
│       └── config.json
│
├── 📂 thesis_eval/                   # 论文评估结果
│   ├── fixed_vs_adaptive_results.csv # 对比结果表格
│   ├── fixed_vs_adaptive_results.json
│   └── fixed_vs_adaptive_comparison.png
│
├── 📂 weights/                       # 模型权重
│   └── model_best.pt                 # iResNet50预训练权重
│
├── 📂 camera_eval/                   # 相机benchmark结果
│   ├── performance_metrics.csv
│   ├── camera_benchmark_report.json
│   └── camera_demo_*.mp4
│
├── 📄 camera_benchmark.py            # 相机性能评估主脚本
├── 📄 camera_interactive.py          # 实时相机识别 (支持adaptive_mode)
├── 📄 diagnose_camera.py             # 相机诊断工具
├── 📄 run.py                         # Web演示主入口
├── 📄 video_benchmark.py             # 视频benchmark
└── 📄 download.py                    # 下载工具
```

---

## 🚀 核心功能

### 1. Web实时识别 (生产环境)

```bash
# 启动Web演示
python run.py

# 相机benchmark
bash run_camera_benchmark.sh
```

**数据库**: `benchmark/YTF_100p.db` (103 persons)
**模式**: 固定阈值 (threshold=0.5)
**用途**: 实时演示

---

### 2. 自适应开放集识别框架 (试验/论文)

#### Step 1: 提取特征

```bash
# 从CASIA-WebFace提取特征
python scripts/extract_casia_features.py \
    --dataset-path F:\Dataset\CASIA-WebFace \
    --num-ids 200 \
    --output benchmark/CASIA_200_features.db
```

#### Step 2: 准备开放集数据

```bash
# CASIA数据库 (推荐)
python scripts/prepare_open_set_optimized.py --db casia --num-known 100

# YTF小数据库 (快速测试)
python scripts/prepare_open_set_optimized.py --db small

# YTF完整数据库 (大规模)
python scripts/prepare_open_set_optimized.py --db full --num-known 400
```

#### Step 3: 运行对比评估

```bash
python scripts/compare_with_prepared_data.py --data-dir benchmark/open_set_data
```

**输出**: `thesis_eval/fixed_vs_adaptive_results.csv`

---

## 📊 关键指标说明

| 指标 | 含义 | 目标 |
|------|------|------|
| **OSR** | Open-Set Recognition Rate | 总体识别率 ↑ |
| **KCA** | Known Class Accuracy | 已知人准确率 ↑ |
| **UDR** | Unknown Detection Rate | 陌生人检测率 ↑ |
| **Precision** | 拒绝决策准确度 | ↑ |
| **F1-Score** | 综合指标 | ↑ |

---

## 🔧 常用命令

### 数据库相关

```bash
# 检查数据库
python tools/check_db.py benchmark/YTF_100p.db

# 验证YTF数据库
python tools/verify_ytf_database.py
```

### GPU检测

```bash
# 检查GPU可用性
python tools/check_gpu.py
```

### 分析工具

```bash
# 分析genuine样本方差
python tools/analyze_genuine_variance.py

# 生成ROC曲线
python tools/analyze_threshold_roc.py
```

---

## 📝 重要说明

### ✅ 已修复的Bug

**similarity_con校准函数问题** (2026-04-03修复):
- ❌ 旧版本: 使用非线性校准导致自适应阈值失效
- ✅ 新版本: 直接使用原始cosine相似度 (0-1范围)
- 📍 修改位置: `apps/recognition_system/core/operations.py` line 193

### 🎯 当前状态

- ✅ 核心算法已开发完成
- ✅ Bug已修复 (删除similarity_con)
- ✅ 离线评估框架完善
- ⏳ 尚未集成到Web实时系统

### 📚 数据库选择建议

| 场景 | 数据库 | 特点 |
|------|--------|------|
| Web演示 | YTF_100p.db | 轻量快速 |
| 论文实验 | CASIA_200_features.db | 多样性强 ⭐ 推荐 |
| 大规模验证 | YTF_allID_50features.db | 数据量大 |

**推荐使用CASIA**: 人脸多样性更好，更接近真实场景

---

## 🎓 论文相关

### 创新点

1. **Per-Identity自适应阈值**: 每个人单独计算阈值 (μ - 2σ)
2. **多层决策机制**: Z-score + 自适应阈值 + 不确定性门控
3. **原始相似度空间**: 避免非线性变换带来的问题

### 实验数据路径

```
thesis_eval/
├── fixed_vs_adaptive_results.csv       ← 论文表格
├── fixed_vs_adaptive_results.json      ← 详细数据
└── fixed_vs_adaptive_comparison.png    ← 论文图表
```

---

## ⚙️ 环境要求

```bash
# Python包
torch>=1.9.0
opencv-python
numpy
sqlite3
matplotlib
pandas
```

```bash
# 硬件
GPU: NVIDIA (支持CUDA)
RAM: 8GB+
```

---

## 🔗 快速开始

### 方式1: Web演示

```bash
python run.py
```

### 方式2: 相机Benchmark

```bash
bash run_camera_benchmark.sh
```

### 方式3: 论文实验

```bash
# 完整流程
python scripts/extract_casia_features.py --dataset-path F:\Dataset\CASIA-WebFace --num-ids 200
python scripts/prepare_open_set_optimized.py --db casia --num-known 100
python scripts/compare_with_prepared_data.py

# 查看结果
cat thesis_eval/fixed_vs_adaptive_results.csv
```

---

## 📞 联系与反馈

如有问题，请参考：
- 项目内存: `.claude/projects/e--RecognitionSystem/memory/MEMORY.md`
- 实验指南: 运行脚本时查看输出提示

---

**最后更新**: 2026-04-03
**项目状态**: ✅ 核心功能完成，试验阶段
