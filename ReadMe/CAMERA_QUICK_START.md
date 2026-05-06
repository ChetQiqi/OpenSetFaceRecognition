# 🚀 摄像头性能评估 - 快速运行指南

## 📌 你的配置

```
人脸库:  benchmark/YTF_100p.db
模型:    weights/model_best.pt
采样:    30秒
```

---

## ⚡ 最快上手 (3 种选择)

### 【选项1】一键完整流程 ⭐ 推荐答辩用

```bash
bash run_camera_benchmark.sh
```

**自动执行：**
1. 30秒摄像头采样
2. 交互式人工标注
3. 生成数据报告
4. 生成5张答辩级图表

**输出：**
- `camera_demo_*.mp4` - 演示视频
- `camera_benchmark_report.json` - 性能报告
- `chart_camera_0X_*.png` - 5张答辩图表

---

### 【选项2】仅采样+标注 (最常用)

```bash
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --enable-annotation \
    --use-tracker
```

**功能：**
- ✅ 30秒摄像头采样
- ✅ 生成演示视频
- ✅ 交互式标注 (a/r/q)
- ✅ 自动生成报告JSON

**按键说明：**
- `a` - 确认本帧识别正确
- `r` - 拒绝本帧识别错误
- ` ` - 暂停/继续
- `q` - 退出

---

### 【选项3】快速采样 (跳过标注和图表)

```bash
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --skip-report
```

**输出：**
- `camera_demo_*.mp4` - 演示视频
- 性能指标打印到终端

---

## 🎯 7 种常见场景

### 1️⃣ 快速验证系统 (3分钟)
```bash
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --skip-report
```

### 2️⃣ 标准评估 + 标注 (5分钟)
```bash
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --enable-annotation
```

### 3️⃣ 启用人脸追踪 (更精准)
```bash
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --use-tracker \
    --enable-annotation
```

### 4️⃣ 性能优化测试 (每3帧处理)
```bash
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --skip-frames 3 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --use-tracker
```

### 5️⃣ 长时间采样 (60秒)
```bash
python camera_interactive.py \
    --duration 60 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --enable-annotation
```

### 6️⃣ 生成可视化图表 (从已有报告)
```bash
python generate_camera_charts.py \
    --report camera_benchmark_report.json
```

### 7️⃣ 完整答辩准备 (一键全套)
```bash
bash run_camera_benchmark.sh
```

---

## 📊 完整参数说明

### 采样参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--duration` | 30 | 采样时长（秒）|
| `--camera-id` | 0 | 摄像头ID |
| `--skip-frames` | 1 | 跳帧（1=每帧，3=每3帧） |
| `--use-tracker` | ❌ | 启用人脸追踪 |

### 输出参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-report` | camera_benchmark_report.json | 报告输出路径 |
| `--skip-report` | ❌ | 跳过报告生成 |
| `--enable-annotation` | ❌ | 启用人工标注 |

### 模型参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--db-path` | benchmark/YTF_100p.db | 人脸库路径 |
| `--weights` | weights/model_best.pt | 模型权重 |
| `--model-name` | iresnet50 | 模型名称 |
| `--img-size` | 112 | 输入大小 |
| `--device` | cuda:0 | 计算设备 |

### 检测器参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--det-conf-threshold` | 0.6 | 检测置信度 |
| `--det-min-size` | 40 | 最小检测大小 |
| `--detector-backend` | mtcnn | 检测器 |

### 匹配参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--threshold` | 0.5 | 识别阈值 |
| `--match-reduce` | topk_mean | 匹配方法 |
| `--topk` | 3 | Top-K |

---

## 📈 输出说明

### 采样过程生成

**演示视频**
- 文件: `camera_demo_*.mp4`
- 分辨率: 原摄像头分辨率
- 内容:
  - 识别框 (绿色=已识别，红色=未知)
  - 帧号显示
  - 实时FPS数值
  - 人员姓名+置信度

**性能报告**
- 文件: `camera_benchmark_report.json`
- 内容:
  ```json
  {
    "config": {...},
    "performance_metrics": {
      "fps_actual": 30.0,
      "fps_processed": 30.0,
      "fps_theoretical": 32.15,
      "latency_stages": {
        "detection": {...},
        "extraction": {...},
        "matching": {...},
        "total": {...}
      }
    },
    "recognition_metrics": {...},
    "tracking_metrics": {...}
  }
  ```

### 可视化图表 (答辩用)

运行 `python generate_camera_charts.py` 后生成：

| 序号 | 文件名 | 用途 |
|------|--------|------|
| 1 | chart_camera_01_latency_stages.png | 🎯 各阶段延迟分解 |
| 2 | chart_camera_02_realtime_performance.png | 🎬 FPS和延迟对比 |
| 3 | chart_camera_03_recognition_dashboard.png | 📊 仪表盘 (6个KPI) |
| 4 | chart_camera_04_tracking_analysis.png | 📍 轨迹分析 |
| 5 | chart_camera_05_config_summary.png | ⚙️  配置表 |

---

## 🎓 答辩展示流程

### 第1步：获取演示数据
```bash
# 选择以下任一方式
bash run_camera_benchmark.sh              # 最完整
# 或
python camera_interactive.py --enable-annotation --use-tracker
```

### 第2步：生成答辩级图表 (如果还未生成)
```bash
python generate_camera_charts.py --report camera_benchmark_report.json
```

### 第3步：答辩展示 (推荐顺序)
1. **chart_camera_03_recognition_dashboard.png** - "系统整体性能"
2. **chart_camera_02_realtime_performance.png** - "实时处理能力"
3. **chart_camera_01_latency_stages.png** - "性能分解分析"
4. **camera_demo_*.mp4** - "实时演示效果" (全屏放映)
5. **chart_camera_05_config_summary.png** - "回答规模问题"

---

## ⚙️ 高级用法

### 性能对比测试 (比较跳帧策略)

```bash
# 测试1：每帧处理
python camera_interactive.py \
    --duration 30 --skip-frames 1 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --output-report report_skip1.json

# 测试2：每3帧处理
python camera_interactive.py \
    --duration 30 --skip-frames 3 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --output-report report_skip3.json

# 生成对比图表
python generate_camera_charts.py --report report_skip1.json
python generate_camera_charts.py --report report_skip3.json
```

### 调整识别阈值 (提高准确率)

```bash
python camera_interactive.py \
    --duration 30 \
    --threshold 0.6 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --output-report report_high_threshold.json
```

### 使用不同摄像头

```bash
# 尝试摄像头ID 0, 1, 2...
python camera_interactive.py \
    --camera-id 1 \
    --duration 30 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt"
```

---

## 🐛 常见问题

### Q: 摄像头打不开
A: 尝试不同的 `--camera-id`：
```bash
python camera_interactive.py --camera-id 1 ...
```

### Q: CUDA out of memory
A: 降低检测阈值或启用跳帧：
```bash
python camera_interactive.py \
    --skip-frames 3 \
    --det-conf-threshold 0.7 \
    ...
```

### Q: 识别不准确
A: 提高识别阈值：
```bash
python camera_interactive.py \
    --threshold 0.6 \
    ...
```

### Q: 只想要报告不想标注
A: 不使用 `--enable-annotation` 参数

### Q: 想更快得到结果
A: 跳过报告和图表：
```bash
python camera_interactive.py --skip-report --camera-id 0 ...
```

---

## 📋 完整 shell 脚本示例

### minimal_test.sh (最小化)
```bash
#!/bin/bash
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --skip-report
```

### full_benchmark.sh (完整版)
```bash
#!/bin/bash
# 采样 + 标注
python camera_interactive.py \
    --duration 30 \
    --camera-id 0 \
    --db-path "benchmark/YTF_100p.db" \
    --weights "weights/model_best.pt" \
    --enable-annotation \
    --use-tracker \
    --output-report camera_report.json

# 生成图表
python generate_camera_charts.py --report camera_report.json

echo "✅ 完成！检查输出文件"
```

---

生成日期: 2026-03-22
最后更新: 2026-03-22
