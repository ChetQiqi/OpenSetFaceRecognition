# 📹 摄像头实时性能评估 - 快速开始

## ⚡ 一键运行（推荐）

```bash
bash run_camera_benchmark.sh
```

**自动执行：**
1. ✅ 30秒摄像头采样
2. ✅ 交互式标注 (可选: a/r/q)
3. ✅ 生成性能报告
4. ✅ 所有文件保存到 `camera_eval/` 文件夹

---

## 📊 输出文件说明

采样完成后，`camera_eval/` 文件夹中包含：

| 文件名 | 用途 |
|--------|------|
| `camera_demo_*.mp4` | 📹 演示视频（含识别框、FPS指标） |
| `camera_benchmark_report.json` | 📄 完整性能数据（JSON格式） |
| `performance_metrics.csv` | 📊 性能指标表（Excel/CSV格式）|

---

## 📋 性能指标表 (CSV)

`performance_metrics.csv` 包含以下指标：

### 配置部分
- 采样时长 (秒)
- 总帧数
- 处理帧数
- 人脸库规模 (人数)

### 实时性能
| 指标 | 说明 |
|------|------|
| 实际帧率 | 摄像头输入：X.XX FPS |
| 处理帧率 | 系统处理：X.XX FPS |
| 理论帧率 | 理想情况：X.XX FPS |

### 延迟分析 (ms毫秒)
| 阶段 | 描述 |
|------|------|
| 检测平均延迟 | 人脸检测(MTCNN) |
| 提取平均延迟 | 特征提取(iResNet50) |
| 匹配平均延迟 | 底库搜索(1:N) |
| 总平均延迟 | 端到端总耗时 |
| 总P95延迟 | 95%情况的延迟 |

### 识别性能
| 指标 | 说明 |
|------|------|
| 总检测数 | 检测到的人脸总数 |
| 总识别数 | 成功识别的人脸数 |
| 不同人物 | 识别到的不同个体数 |
| 检测率 (%) | 检测到人脸的帧比例 |
| 平均置信度 | 识别结果的平均置信值 (0-1) |

### 跟踪性能
| 指标 | 说明 |
|------|------|
| 轨迹总数 | 跟踪的人脸轨迹数量 |
| 平均轨迹长度 | 同一人被跟踪的帧数 |
| 最长轨迹 | 最长的连续跟踪长度 |
| 身份一致性 (%) | 同轨迹内识别结果的一致性 |

---

## 📊 JSON 报告结构

`camera_benchmark_report.json` 包含详细数据：

```json
{
  "config": {
    "duration_seconds": 30,
    "total_frames": 900,
    "processed_frames": 900,
    "db_size": 100
  },
  "performance_metrics": {
    "fps_actual": 30.0,
    "fps_processed": 30.0,
    "fps_theoretical": 35.2,
    "latency_stages": {
      "detection": {
        "mean": 65.3,
        "std": 5.2,
        "p50": 64.1,
        "p95": 72.5,
        "count": 900
      },
      "extraction": {...},
      "matching": {...},
      "total": {...}
    }
  },
  "recognition_metrics": {
    "total_detections": 245,
    "total_recognitions": 198,
    "unique_persons": 15,
    "detection_rate": 27.2,
    "confidence_stats": {
      "mean": 0.8234,
      "std": 0.1123,
      "min": 0.5001,
      "max": 0.9989
    }
  },
  "tracking_metrics": {
    "total_tracks": 18,
    "avg_track_length": 13.6,
    "max_track_length": 45,
    "identity_consistency": 94.2
  }
}
```

---

## 🎯 关键性能指标评估标准

| 指标 | 目标 | 说明 |
|------|------|------|
| **实时性** | ≥25 FPS | 达到实时处理标准 |
| **总延迟** | <50ms | 单帧平均耗时 |
| **检测率** | ≥80% | 检测到人脸的能力 |
| **识别准确率** | 越高越好 | 基于人工标注验证 |
| **置信度** | ≥0.5 | 识别结果的可信度 |
| **身份一致性** | ≥80% | 跟踪稳定性 |

---

## 🔧 自定义参数

### 快速修改脚本参数

编辑 `run_camera_benchmark.sh`，调整：

```bash
DURATION=30              # 采样时长（秒）
CAMERA_ID=0              # 摄像头ID（改为1,2...尝试不同摄像头）
SKIP_FRAMES=3            # 跳帧：1=每帧，3=每3帧（性能优化）
```

### 或直接命令行指定

```bash
# 60秒采样，每3帧处理一次
python camera_interactive.py \
    --duration 60 \
    --skip-frames 3 \
    --enable-annotation \
    --use-tracker \
    --db-path "benchmark\\YTF_100p.db" \
    --weights "weights\\adasin_best.pt" \
    --output-report "camera_eval/my_report.json"
```

---

## ⏸️ 标注模式操作说明

采样完成后会进入标注模式（如使用了 `--enable-annotation`）：

```
拍摄的演示视频将播放，按键标注：

  'a' - 接受：本帧的识别结果正确 ✅
  'r' - 拒绝：本帧的识别结果错误 ❌
  ' ' - 暂停/继续播放
  'q' - 退出标注
```

人工标注可用于验证识别准确率。

---

## 📁 文件夹结构

运行后的输出结构：

```
RecognitionSystem/
├── camera_eval/               ← 所有输出都在这里
│   ├── camera_demo_*.mp4      ← 演示视频
│   ├── camera_benchmark_report.json  ← JSON报告
│   └── performance_metrics.csv       ← 性能指标表
├── run_camera_benchmark.sh
├── camera_interactive.py
└── ... (其他文件)
```

---

## 🐛 常见问题

### Q: 摄像头打不开
A: 尝试修改 `CAMERA_ID=1` 或更高数字
```bash
sed -i 's/CAMERA_ID=0/CAMERA_ID=1/g' run_camera_benchmark.sh
bash run_camera_benchmark.sh
```

### Q: 内存不足
A: 启用跳帧优化
```bash
sed -i 's/SKIP_FRAMES=3/SKIP_FRAMES=5/g' run_camera_benchmark.sh
```

### Q: 想跳过标注快速生成报告
A: 去掉 `--enable-annotation` 参数（编辑脚本第17行）

### Q: 识别不准
A: 确保人脸库正确
```bash
# 检查数据库
ls -lh benchmark/YTF_100p.db
```

### Q: 性能指标不达标
A: 检查：
1. 是否启用了GPU (`--device cuda:0`)
2. 检测器参数是否合理 (`--det-conf-threshold 0.6`)
3. 可尝试提高识别阈值 (`--threshold 0.6`)

---

## 📊 查看结果

### 方式1：打开CSV表格（推荐）
```bash
# Windows
start camera_eval/performance_metrics.csv

# Linux/Mac
open camera_eval/performance_metrics.csv
```

### 方式2：查看JSON报告
```bash
# Windows
type camera_eval/camera_benchmark_report.json

# Linux/Mac
cat camera_eval/camera_benchmark_report.json
```

### 方式3：用Python分析
```python
import json

with open('camera_eval/camera_benchmark_report.json') as f:
    report = json.load(f)

# 或遍历查看
for key, val in report.items():
    print(f"{key}: {val}")
```

---

## 🎓 论文答辩参考

可在论文中引用：
- **性能数据**：camera_benchmark_report.json (JSON)
- **指标表格**：performance_metrics.csv (Excel)
- **演示视频**：camera_demo_*.mp4 (答辩时展示)

示例引用：
```
本文采用摄像头实时采样评估系统在真实场景下的性能。
采样时长：30秒，采样帧数：900帧，人脸库规模：100人。
结果如表1所示，系统实时处理帧率达到30.0 FPS，
平均单帧延迟35.2ms，已满足实时处理要求。
```

---

## 💡 优化建议

如果性能不达标，可尝试：

1. **启用人脸跟踪**（已启用）
   ```bash
   --use-tracker
   ```

2. **调整跳帧策略**
   ```bash
   --skip-frames 5  # 每5帧处理一次
   ```

3. **降低检测阈值**
   ```bash
   --det-conf-threshold 0.7
   ```

4. **调整识别阈值**
   ```bash
   --threshold 0.6
   ```

---

最后更新：2026-03-22
