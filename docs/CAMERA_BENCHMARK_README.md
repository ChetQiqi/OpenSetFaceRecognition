# 📹 摄像头实时性能评估系统

## 系统概述

用于评估人脸识别系统在真实摄像头场景下的性能，包括：
- **实时性能**：FPS、单帧延迟、各模块耗时分解
- **识别性能**：人脸检测率、识别准确率、置信度分析
- **跟踪性能**：同一人的多次识别追踪、身份一致性
- **稳定性**：帧处理失败率、性能波动等

## 三个核心脚本

### 1️⃣ `camera_interactive.py` ⭐ 推荐使用

**完整的交互式工具，包含采样→标注→报告→可视化全流程**

#### 基础使用
```bash
# 30秒采样 (最基础)
python camera_interactive.py

# 30秒采样 + 启用人工标注
python camera_interactive.py --enable-annotation

# 自定义采样时长
python camera_interactive.py --duration 60 --enable-annotation
```

#### 完整工作流
```bash
# 30秒采样，生成演示视频和报告，启用标注
python camera_interactive.py \
  --duration 30 \
  --camera-id 0 \
  --enable-annotation \
  --output-report camera_demo_report.json \
  --use-tracker
```

#### 可配置参数

**采样参数：**
- `--duration INT` - 采样时长（秒，默认30）
- `--camera-id INT` - 摄像头ID（默认0）
- `--skip-frames INT` - 跳帧配置（默认1=每帧处理，3=每3帧处理一次）
- `--use-tracker` - 启用人脸追踪，用于跟踪同一人的多次识别

**输出参数：**
- `--output-report PATH` - JSON报告输出路径（默认camera_benchmark_report.json）
- `--skip-report` - 跳过报告生成
- `--enable-annotation` - 启用手动标注模式

**模型参数：**
- `--db-path PATH` - 人脸库路径（默认face_system/face_features.db）
- `--weights PATH` - 模型权重（默认face_system/iresnet50.pth）
- `--model-name STR` - 模型名称（默认iresnet50）
- `--device STR` - 计算设备（默认cuda:0）

**检测器参数：**
- `--det-conf-threshold FLOAT` - 检测置信度（默认0.6）
- `--det-min-size INT` - 最小检测大小（默认40）
- `--detector-backend STR` - 检测器（默认mtcnn）

**匹配参数：**
- `--threshold FLOAT` - 识别阈值（默认0.5）
- `--match-reduce STR` - 匹配方法（默认topk_mean）
- `--topk INT` - Top-K（默认3）

#### 工作流程

**阶段1：采样** (30秒，自动)
```
✅ 初始化系统
🎥 打开摄像头
⏳ 采样30秒数据
📹 保存演示视频: camera_demo_1234567890.mp4
```
输出：高分辨率演示视频 + 原始性能数据

**阶段2：标注** (交互式，可选)
```
按键说明:
  'a' - 确认本帧识别正确 (Accept)
  'r' - 拒绝本帧识别错误 (Reject)
  ' ' - 暂停/继续播放
  'q' - 退出标注

✅ Frame 1: Accept
❌ Frame 2: Reject
...
```
用途：人工筛选准确的识别结果

**阶段3：报告生成** (自动)
```
✅ 报告已保存: camera_benchmark_report.json
📈 性能评估摘要:
   采样时长: 30秒
   总帧数: 900
   处理帧数: 900 (100.0%)
   实际帧率: 30.00 FPS
   理论帧率: 32.15 FPS
   ...
```

#### 自动生成的数据

**JSON报告结构** (camera_benchmark_report.json):
```json
{
  "config": {
    "duration_seconds": 30,
    "total_frames": 900,
    "processed_frames": 900,
    "db_size": 101
  },
  "performance_metrics": {
    "fps_actual": 30.0,
    "fps_processed": 30.0,
    "fps_theoretical": 32.15,
    "latency_stages": {
      "detection": {"mean": 65.2, "std": 5.3, "p95": 72.1, ...},
      "extraction": {"mean": 25.1, ...},
      "matching": {"mean": 2.3, ...},
      "total": {"mean": 92.6, ...}
    }
  },
  "recognition_metrics": {
    "total_detections": 234,
    "total_recognitions": 198,
    "unique_persons": 15,
    "detection_rate": 26.0,
    "confidence_stats": {...}
  },
  "tracking_metrics": {
    "total_tracks": 18,
    "avg_track_length": 15.2,
    "max_track_length": 45,
    "identity_consistency": 94.2
  }
}
```

### 2️⃣ `generate_camera_charts.py`

**从报告生成5张答辩级可视化图表**

#### 使用
```bash
# 使用默认报告文件
python generate_camera_charts.py

# 指定报告文件
python generate_camera_charts.py --report my_report.json
```

#### 生成的图表

| 序号 | 图表名称 | 用途 |
|------|--------|------|
| 1 | chart_camera_01_latency_stages.png | 🎯 延迟分解 - 检测/提取/匹配各阶段耗时柱状图 |
| 2 | chart_camera_02_realtime_performance.png | 🎬 实时性能 - FPS对比和延迟分布 |
| 3 | chart_camera_03_recognition_dashboard.png | 📊 识别仪表盘 - 6个关键KPI卡片 |
| 4 | chart_camera_04_tracking_analysis.png | 📍 跟踪分析 - 轨迹长度和身份一致性 |
| 5 | chart_camera_05_config_summary.png | ⚙️  配置汇总 - 完整测试参数和结果表 |

**推荐展示顺序**（答辩时）：
1. chart_camera_03_recognition_dashboard.png - 快速概览全景
2. chart_camera_02_realtime_performance.png - 展示实时性能达成
3. chart_camera_01_latency_stages.png - 详细性能分解
4. chart_camera_04_tracking_analysis.png - 跟踪能力展示
5. chart_camera_05_config_summary.png - 提供完整技术数据

### 3️⃣ `camera_benchmark.py`

**轻量级采样脚本** (如果只需要快速采样，不需要完整流程)

```bash
# 基础采样
python camera_benchmark.py

# 自定义配置
python camera_benchmark.py \
  --duration 60 \
  --skip-frames 3 \
  --use-tracker
```

---

## 📊 完整工作流示例

### 场景：准备论文答辩的摄像头性能评估

```bash
# 第1步：30秒采样 + 人工标注 + 生成报告
python camera_interactive.py \
  --duration 30 \
  --enable-annotation \
  --use-tracker \
  --output-report camera_demo_report.json

# 第2步：从报告生成5张答辩级可视化图表
python generate_camera_charts.py --report camera_demo_report.json

# 第3步：查看生成的图表
# 输出文件：
#   camera_demo_1234567890.mp4 - 演示视频
#   camera_demo_report.json - 性能报告
#   chart_camera_01_*.png 到 chart_camera_05_*.png - 5张图表
```

### 输出文件说明

**采样过程生成：**
- `camera_demo_*.mp4` - 高分辨率演示视频（含识别结果和性能指标）
- `camera_benchmark_report.json` - 完整性能数据JSON

**可视化生成：**
- `chart_camera_01_latency_stages.png` (128K, 300dpi) - 延迟分解
- `chart_camera_02_realtime_performance.png` (111K, 300dpi) - 实时性能
- `chart_camera_03_recognition_dashboard.png` (193K, 300dpi) - 仪表盘
- `chart_camera_04_tracking_analysis.png` (394K, 300dpi) - 跟踪分析
- `chart_camera_05_config_summary.png` (432K, 300dpi) - 配置汇总

所有图表均为答辩级质量（300 DPI），支持投影展示和打印。

---

## 🎯 关键性能指标解释

| 指标 | 含义 | 标准 | 说明 |
|------|------|------|------|
| **fps_actual** | 实际帧率 | ≥30 | 摄像头捕获速率（影响系统整体流畅度） |
| **fps_processed** | 处理帧率 | ≥25 | 系统处理速率（需≥25才能实时） |
| **fps_theoretical** | 理论帧率 | 越高越好 | 单帧平均延迟换算成FPS（理想情况） |
| **总延迟mean** | 平均延迟 | <50ms | 包含检测、提取、匹配的总耗时 |
| **detection_rate** | 检测率 | 越高越好 | 成功检测到人脸的帧数比例 |
| **unique_persons** | 不同人物 | 越多越好 | 在采样中识别到的不同人员数 |
| **avg_confidence** | 平均置信度 | ≥0.5 | 识别结果的平均置信度（0-1） |
| **identity_consistency** | 身份一致性 | ≥80% | 同一轨迹中识别结果的一致性 |
| **avg_track_length** | 平均轨迹长度 | 越长越好 | 同一人脸被连续跟踪的帧数 |

---

## ⚡ 性能优化建议

### 如果FPS不够（<25）

1. **启用跳帧** - 每3帧处理一次
   ```bash
   python camera_interactive.py --skip-frames 3
   ```

2. **启用人脸追踪** - 减少重复计算
   ```bash
   python camera_interactive.py --use-tracker
   ```

3. **调整检测器参数**
   ```bash
   python camera_interactive.py \
     --det-conf-threshold 0.7 \
     --det-min-size 60
   ```

### 如果识别准确率不够

1. **提高识别阈值**
   ```bash
   python camera_interactive.py --threshold 0.6
   ```

2. **增加底库特征数** - 注册更多样本
   ```bash
   python register_random_persons.py --num-persons 500
   ```

---

## 🐛 常见问题

### Q: 摄像头打不开怎么办？
A: 检查摄像头ID，尝试 `--camera-id 1` 或更高的数字

### Q: 报告JSON生成失败？
A: 检查输出路径是否存在，或指定完整路径：
```bash
python camera_interactive.py --output-report /absolute/path/report.json
```

### Q: 想跳过标注快速生成报告？
A: 不使用 `--enable-annotation` 参数即可

### Q: 只想测试延迟不想要演示视频？
A: 使用 `camera_benchmark.py` 的简化版本

---

## 📌 答辩展示建议

1. **第一张图** (chart_camera_03...): 快速展示系统性能全景
2. **第二张图** (chart_camera_02...): 强调实时性能（FPS和延迟）
3. **第三张图** (chart_camera_01...): 详细技术分析
4. **第四张图** (chart_camera_04...): 展示人脸跟踪能力
5. **第五张图** (chart_camera_05...): 回答评委关于测试规模的问题

配合演示视频 (MP4) 展示实时效果！

---

## 与视频基准的对比

| 特性 | 视频基准 | 摄像头评估 |
|------|---------|-----------|
| 数据源 | 预录视频文件 | 实时摄像头流 |
| 样本规模 | 208个视频，33791帧 | 可自定义（默认30秒） |
| 测试模式 | 离线批处理 | 在线实时流 |
| 人脸跟踪 | ✅ 支持 | ✅ 支持 |
| 人工标注 | ❌ | ✅ 支持 |
| 演示视频 | ❌ | ✅ 自动生成 |
| 性能指标 | 全局统计 | 全局 + 帧级详细 |

---

生成日期：2026-03-22
