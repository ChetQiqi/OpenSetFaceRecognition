# 🎯 RecognitionSystem - 人脸识别系统

基于深度学习的人脸识别系统，支持实时相机识别和开放集识别评估。

---

## ✨ 主要功能

### 1. 实时人脸识别 (Web演示)
- ✅ 基于iResNet50的高精度特征提取
- ✅ MTCNN人脸检测
- ✅ 实时相机识别与性能评估

### 2. 自适应开放集识别框架 (论文创新)
- ✅ Per-Identity自适应阈值
- ✅ 多层决策机制 (Z-score + 自适应阈值 + 不确定性)
- ✅ 支持陌生人检测 (Unknown Detection)

---

## 🚀 快速开始

### Web演示

```bash
python run.py
```

### 相机性能评估

```bash
bash run_camera_benchmark.sh
```

### 论文实验 (推荐使用CASIA数据集)

```bash
# Step 1: 提取CASIA特征 (~30分钟)
python scripts/extract_casia_features.py \
    --dataset-path F:\Dataset\CASIA-WebFace \
    --num-ids 200

# Step 2: 准备开放集数据 (<1分钟)
python scripts/prepare_open_set_optimized.py --db casia --num-known 100

# Step 3: 运行对比评估 (5-10分钟)
python scripts/compare_with_prepared_data.py

# 查看结果
cat thesis_eval/fixed_vs_adaptive_results.csv
```

---

## 📊 项目结构

```
RecognitionSystem/
├── apps/                    # 核心代码
│   └── recognition_system/
│       └── core/            # 核心模块
├── scripts/                 # 实验脚本
│   ├── extract_casia_features.py
│   ├── prepare_open_set_optimized.py
│   └── compare_with_prepared_data.py
├── tools/                   # 分析工具
├── benchmark/               # 数据库
├── thesis_eval/             # 论文结果
└── weights/                 # 模型权重
```

详细说明请查看: [📁 PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 📈 性能指标

### 开放集识别 (CASIA数据集)

| 指标 | 固定阈值 | 自适应阈值 | 改进 |
|------|---------|-----------|------|
| OSR  | ~97%    | ~98%      | +1%  |
| KCA  | ~95%    | ~96%      | +1%  |
| UDR  | ~98%    | ~100%     | +2%  |

**注**: 实际结果取决于数据集和参数配置

---

## 💡 核心创新

### 自适应阈值框架

**问题**: 固定全局阈值对所有人都一样，不够灵活

**解决方案**:
```
为每个已知身份单独计算阈值:
τ_i = μ_i - 2σ_i

其中:
- μ_i: 该身份genuine样本的平均相似度
- σ_i: 该身份genuine样本的标准差
```

**优势**:
- ✅ 适应不同身份的特征分布差异
- ✅ 提高陌生人检测率
- ✅ 保持已知人识别准确率

---

## 🗂️ 数据库

### Web演示用
- **YTF_100p.db**: 103 persons, 轻量快速

### 论文实验用 (推荐)
- **CASIA_200_features.db**: 200 persons from CASIA-WebFace
  - 特点: 人脸多样性强，更真实
  - 推荐用于论文实验

### 大规模验证用
- **YTF_allID_50features.db**: 1595 persons, 79750 features
  - 特点: 数据量大
  - 适合大规模性能验证

---

## 🔧 环境要求

### Python包
```bash
pip install torch opencv-python numpy pandas matplotlib
```

### 硬件
- GPU: NVIDIA (推荐)
- RAM: 8GB+
- 磁盘: 2GB+ (用于数据库和权重)

---

## 📝 使用说明

### 选择数据集

**推荐使用CASIA-WebFace**，原因:
- ✅ 人脸角度/光照/质量多样化
- ✅ 更接近真实应用场景
- ✅ 自适应阈值计算更合理

YTF数据集问题:
- ❌ 同一人的特征来自1-4个视频，高度相似
- ❌ 导致σ很小，自适应阈值过高
- ❌ 容易产生false rejection

### 参数调整

```bash
# 调整已知/未知比例
python scripts/prepare_open_set_optimized.py --db casia --num-known 150

# 调整gallery/test特征数
python scripts/prepare_open_set_optimized.py --db casia --gallery-size 15 --test-size 15
```

---

## 🎓 论文应用

本框架作为**论文创新点**使用:
- 提出自适应开放集识别方法
- 在多个数据集上验证有效性
- 与固定阈值基线对比

**注**: 当前Web演示仍使用固定阈值（稳定性考虑）

---

## 📂 关键文件

| 文件 | 用途 |
|------|------|
| `run.py` | Web演示入口 |
| `camera_benchmark.py` | 相机性能评估 |
| `scripts/extract_casia_features.py` | 提取CASIA特征 |
| `scripts/prepare_open_set_optimized.py` | 准备开放集数据 |
| `scripts/compare_with_prepared_data.py` | 运行对比评估 |
| `PROJECT_STRUCTURE.md` | 详细项目结构 |

---

## 🐛 已知问题与修复

### ✅ 已修复: similarity_con校准函数
- **问题**: 非线性校准破坏自适应阈值
- **修复**: 删除校准，直接使用原始cosine相似度
- **影响**: KCA从75%恢复到96%

---

## 📞 更多信息

详细文档:
- [项目结构](PROJECT_STRUCTURE.md)
- [内存文档](.claude/projects/e--RecognitionSystem/memory/MEMORY.md)

---

**项目状态**: ✅ 核心功能完成
**最后更新**: 2026-04-03
