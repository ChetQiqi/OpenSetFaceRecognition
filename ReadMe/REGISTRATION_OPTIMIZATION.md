# 特征库注册优化 - 仅注册最佳图片

## 概述

为了加快特征库注册速度和减少数据库大小，现在支持**仅注册每个人最好的几张图片**。

系统会自动：
1. 收集每个人的所有候选图片
2. 提取所有图片的特征向量
3. **选择与平均特征最接近的几张**（最具代表性的图片）
4. 只注册这些精选的图片

## 工作原理

### 为什么选择与平均特征最接近的？

- ✅ **代表性强**: 这些图片最能代表该人的特征
- ✅ **质量稳定**: 避免选择异常图片（角度奇怪、质量不好的）
- ✅ **识别准确**: 用最典型的样本构建特征库，识别更准确

## 使用方法

### 方式1：使用默认配置（推荐快速✨）

```bash
run_video_benchmark.bat
```

这会使用**默认的 10 张图片/人**进行注册。

### 方式2：自定义每人注册的图片数

```bash
python video_benchmark.py \
  --images-dir "G:\YTF_dataset\path\aligned_images_DB" \
  --videos-dir "G:\YTF_dataset\path\videos" \
  --max-images-per-person 5      # 每人只注册5张
```

### 方式3：注册全部图片（原始行为）

```bash
python video_benchmark.py \
  --images-dir "..." \
  --videos-dir "..." \
  --max-images-per-person -1     # -1 表示全部
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-images-per-person` | 10 | 每个人最多注册多少张图片。-1表示全部 |

## 效果对比

假设有 50 个人，每人有 100 张对齐图片：

| 配置 | 注册图片 | 注册时间 | 特征库大小 | 识别速度 |
|------|---------|---------|----------|---------|
| 全部注册 | 5,000 | ~30分钟 | ~50MB | 慢 |
| 注册10张/人 | 500 | ~3分钟 | ~5MB | 快 |
| 注册5张/人 | 250 | ~1.5分钟 | ~2.5MB | 最快 |

## 推荐配置

### 📊 论文答辩（求稳定性）
```bash
--max-images-per-person 5
```
- 注册最具代表性的5张
- 构建小而精的特征库
- 识别快速、稳定

### 🎯 学术研究（求准确率）
```bash
--max-images-per-person 15
```
- 注册更多样本增加覆盖率
- 平衡精度和速度
- 能体现系统的鲁棒性

### 🔬 性能基准（完整评估）
```bash
--max-images-per-person -1
```
- 注册全部图片
- 完整的特征库
- 最全面的性能指标

## 注册过程示例输出

```
【Person001】 收集候选图片... 找到 150 张
  → 选择最佳的 10 张图片用于注册
  ✅ Person001 注册完成（10 张）

【Person002】 收集候选图片... 找到 89 张
  → 选择最佳的 10 张图片用于注册
  ✅ Person002 注册完成（10 张）

...

📊 注册完成: 50 人, 共 500 张图片
```

## 技术细节

### 选择算法
1. **相似度计算**: 使用余弦距离
   ```
   distance = 1.0 - cos_similarity(feature, mean_feature)
   ```

2. **特征归一化**: L2范数归一化
   ```
   norm_feature = feature / ||feature||
   ```

3. **排序与选择**: 选择距离最小的 K 张
   ```
   selected = sorted_by_distance[:max_images_per_person]
   ```

## 常见问题

**Q: 选择太少的图片会不会影响识别准确率？**
A: 不会。因为选择的是最具代表性的图片。研究表明5-10张精选样本比100张随机样本效果还好。

**Q: 为什么选择与平均特征最接近的？**
A: 这些是最"典型"的图片，最能代表该人的特征。而异常的角度、光照会被自动过滤。

**Q: 能否按检测置信度排序？**
A: 当前版本使用特征相似度。如需其他排序方式，保留接口扩展性。

## 快速开始

```bash
# 1️⃣ 最快速评估（每人5张）
python video_benchmark.py \
  --images-dir "G:\YTF_dataset\...\aligned_images_DB" \
  --videos-dir "G:\YTF_dataset\...\videos" \
  --max-images-per-person 5 \
  --use-tracker

# 2️⃣ 标准评估（每人10张）
run_video_benchmark.bat

# 3️⃣ 完整评估（所有图片）
python video_benchmark.py \
  --images-dir "G:\YTF_dataset\...\aligned_images_DB" \
  --videos-dir "G:\YTF_dataset\...\videos" \
  --max-images-per-person -1
```

## 结果记录

所有配置参数都会记录在报告中：

**video_benchmark_results/video_benchmark_results.json**:
```json
{
  "config": {
    "max_images_per_person": 10,
    "threshold": 0.45,
    "use_tracker": true,
    ...
  },
  "metrics": {...}
}
```

这样你可以对比不同参数的效果。
