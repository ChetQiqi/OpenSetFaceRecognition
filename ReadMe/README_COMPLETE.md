# 人脸识别系统 - 完整使用指南

基于 iResNet50 + AdaSin 的深度学习人脸识别系统，支持实时摄像头识别、图片/视频批量识别、性能基准测试等功能。

---

## 📁 项目结构

```
E:\MyRecoCode/
├── apps/recognition_system/        # 主要应用代码
│   ├── core/                       # 核心算法模块
│   │   ├── detector.py             # MTCNN 人脸检测
│   │   ├── model.py                # iResNet50 特征提取
│   │   ├── matcher.py              # 余弦相似度匹配
│   │   ├── feature_db.py           # SQLite 特征数据库
│   │   ├── operations.py           # 统一调用接口
│   │   ├── cli.py                  # 命令行工具
│   │   ├── camera_thread.py        # 摄像头识别线程
│   │   └── eval_comprehensive.py   # 准确性评估
│   ├── streamlit_app.py            # Streamlit Web 界面
│   ├── add_person_to_db.py         # 人员管理工具
│   ├── run.py                      # 统一启动脚本
│   ├── draw_architecture.py        # 绘制系统架构图
│   ├── draw_camera_flow.py         # 绘制流程图
│   ├── draw_nn_structure.py        # 绘制神经网络结构图
│   └── start_web_ui.bat/.sh        # 一键启动脚本
├── weights/
│   └── adasin_best.pt              # 预训练模型权重
├── benchmark_system.py             # 性能基准测试
├── run_complete_benchmark.py       # 完整测试流程（数据集分割+注册+测试）
├── run_benchmark_flexible.py       # 灵活测试（可选择是否重新注册）
├── merge_benchmark_results.py      # 合并结果生成 LaTeX 表格
├── split_dataset_for_eval.py       # 数据集分割工具
└── requirements.txt                # Python 依赖

生成的文件：
├── benchmark_casia/                # 基准测试输出
│   ├── train/                      # 训练集（60%）
│   ├── test/                       # 测试集（40%）
│   ├── face_features.db            # 特征数据库
│   ├── benchmark_results.json      # 基准测试结果
│   └── performance_table.tex       # LaTeX 性能表格
├── evaluation_results/             # 评估结果
│   └── metrics_detailed.json       # 详细指标
└── log/                            # 日志文件
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境
conda create -n face_recognition python=3.10
conda activate face_recognition

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动 Web 界面（最简单）

**Windows：**
```bash
双击运行: apps\recognition_system\start_web_ui.bat
```

**Linux/Mac：**
```bash
bash apps/recognition_system/start_web_ui.sh
```

然后在浏览器打开 **http://localhost:8501**

### 3. 命令行使用

```bash
# 进入系统目录
cd E:\MyRecoCode

# 查看帮助
python apps/recognition_system/run.py --help

# 启动 Web UI
python apps/recognition_system/run.py web

# 注册人员
python -m apps.recognition_system.core.cli register-dir \
  --dataset-dir 数据集路径 \
  --weights weights/adasin_best.pt \
  --db-path 数据库路径.db

# 识别图片
python -m apps.recognition_system.core.cli recognize-image \
  --weights weights/adasin_best.pt \
  --db-path 数据库路径.db \
  --image-path 图片路径.jpg

# 实时摄像头识别
python -m apps.recognition_system.core.cli recognize-camera \
  --weights weights/adasin_best.pt \
  --db-path 数据库路径.db
```

---

## 📊 性能测试（用于论文）

### 方案A：完整自动化测试（推荐）

从 CASIA 数据集自动选择 N 个人，分割训练/测试集，完成所有测试：

```bash
python run_complete_benchmark.py \
  --casia-dir "F:\Dataset\CASIA-WebFace" \
  --num-persons 20 \
  --skip-camera
```

**输出：**
- ✅ 单帧处理时间
- ✅ 识别准确率
- ✅ GPU/内存占用
- ✅ 完整 LaTeX 表格

**耗时：** 约 10-15 分钟

---

### 方案B：快速重测（使用已有数据库）

如果已经运行过方案A，想调整参数重测：

```bash
python run_benchmark_flexible.py \
  --mode test-only \
  --db-path benchmark_casia/face_features.db \
  --test-dir benchmark_casia/test \
  --skip-camera
```

**输出：** 同方案A

**耗时：** 约 2-3 分钟（跳过注册步骤）

---

### 方案C：只测实时性和资源占用

```bash
python benchmark_system.py \
  --weights weights/adasin_best.pt \
  --db-path benchmark_casia/face_features.db \
  --test-image "F:\Dataset\CASIA-WebFace\0000045\001.jpg" \
  --skip-camera
```

**输出：**
- ✅ 单帧处理时间
- ✅ GPU/内存占用
- ❌ 无准确率（需要独立测试集）

**耗时：** 约 1 分钟

---

## 📈 查看测试结果

### 1. 查看 LaTeX 表格

```bash
# 自动生成的表格
cat benchmark_casia/performance_table.tex
```

示例输出：
```latex
\begin{table}[htbp]
\centering
\caption{系统综合性能测试结果}
\label{tab:system_performance}
\begin{tabular}{llc}
\toprule
评估维度 & 测试指标 & 测试结果 \\
\midrule
\multirow{2}{*}{实时性} & 单人脸检测与识别耗时 & 78 ms \\
 & 摄像头实时处理帧率 (FPS) & N/A \\
\hline
\multirow{2}{*}{准确性} & 等错误率 (EER) & 100.00\% \\
 & 默认阈值 (0.45) 下的识别准确率 & 84.28\% \\
\hline
\multirow{2}{*}{资源占用} & 显存占用 (RTX 3060) & 1.4 GB \\
 & 内存平均占用 & 450 MB \\
\bottomrule
\end{tabular}
\end{table}
```

### 2. 查看详细 JSON 结果

```bash
# 基准测试结果
cat benchmark_casia/benchmark_results.json

# 评估详细指标
cat evaluation_results/metrics_detailed.json
```

---

## 🎨 绘制论文图表

### 系统架构图

```bash
python apps/recognition_system/draw_architecture.py
# 输出: architecture.pdf
```

### 识别流程图

```bash
python apps/recognition_system/draw_camera_flow.py
# 输出: camera_flow.pdf
```

### 神经网络结构图

```bash
python apps/recognition_system/draw_nn_structure.py
# 输出: nn_structure.pdf
```

---

## 🔧 常用命令

### 人员管理

```bash
# 查看已注册人员
python -m apps.recognition_system.core.cli list-persons \
  --db-path video_benchmark.db

# 删除某个人
python -m apps.recognition_system.core.cli remove-person \
  --db-path benchmark_casia/face_features.db \
  --person-name 张三
```

### 数据集分割

```bash
# 手动分割数据集（60%训练，40%测试）
python split_dataset_for_eval.py \
  --source datasets/my_dataset \
  --output datasets/my_dataset_test \
  --ratio 0.4
```

---

## 📊 关键性能指标说明

| 指标 | 含义 | 来源 |
|------|------|------|
| **单帧处理时间** | 检测+识别一张图片的平均耗时 | `benchmark_system.py` |
| **实时FPS** | 摄像头实时处理的帧率 | `benchmark_system.py` |
| **Rank-1 准确率** | Top-1 识别准确率（最常用） | `evaluation_results/metrics_detailed.json` 中的 `rank1_accuracy` |
| **EER** | 等错误率（FAR=FRR时的错误率） | 多人识别场景下计算 |
| **GPU显存** | 模型加载后的显存占用 | `benchmark_system.py` |
| **内存占用** | 系统运行时的内存占用 | `benchmark_system.py` |

**重要提示：**
- **Rank-1 准确率** 是论文中最常用的指标，从 `evaluation_results/metrics_detailed.json` 的 `0.45` → `rank1_accuracy` 获取
- EER 在单人或少数人测试时可能为 100%，这是正常的，多人测试时才有意义

---

## ⚠️ 常见问题

### Q1: 为什么 EER = 100%？

**A:** EER 需要大量不同人的数据才有意义。如果测试集人数较少（<50人），建议使用 **Rank-1 准确率**作为主要指标。

### Q2: 如何修改识别阈值？

**A:** 编辑 `streamlit_app.py` 或使用命令行参数 `--threshold 0.50`（默认 0.45）

### Q3: 摄像头无法打开？

**A:**
1. 检查摄像头是否被占用
2. 修改 `camera_id` 参数（默认为 0）
3. 使用 `--skip-camera` 跳过摄像头测试

### Q4: 如何只重新评估准确性（不重新注册）？

**A:** 使用方案B：
```bash
python run_benchmark_flexible.py \
  --mode test-only \
  --db-path benchmark_casia/face_features.db \
  --test-dir benchmark_casia/test \
  --skip-camera
```

### Q5: CUDA out of memory 错误？

**A:**
1. 减少 batch size（如果有）
2. 使用 CPU 模式：`--device cpu`
3. 关闭其他占用显存的程序

---

## 📝 论文写作建议

### 推荐的性能指标表格

直接使用生成的 `performance_table.tex`，关键指标：

1. **实时性**：
   - 单帧处理时间（例如：78 ms）
   - 理论 FPS（例如：12.8 fps）

2. **准确性**：
   - **Rank-1 准确率**（最重要！例如：84.28%）
   - EER（如果有意义）

3. **资源占用**：
   - GPU 显存（例如：1.4 GB）
   - 内存（例如：450 MB）

### 注意事项

- 如果 EER = 100%，在表格中可以标注为"不适用（测试集规模限制）"
- 重点强调 **Rank-1 准确率**，这是人脸识别领域最通用的指标
- 可以添加"对比实验"一节，对比不同阈值下的准确率

---

## 📞 技术支持

- **Web 界面**：`streamlit run apps/recognition_system/streamlit_app.py`
- **命令行工具**：`python apps/recognition_system/run.py --help`
- **核心 CLI**：`python -m apps.recognition_system.core.cli --help`

---

## 🎯 推荐工作流（论文实验）

```bash
# 1. 首次完整测试（15分钟）
python run_complete_benchmark.py \
  --casia-dir "F:\Dataset\CASIA-WebFace" \
  --num-persons 50 \
  --skip-camera

# 2. 查看结果
cat benchmark_casia/performance_table.tex

# 3. 如需调整参数重测（3分钟）
python run_benchmark_flexible.py \
  --mode test-only \
  --db-path benchmark_casia/face_features.db \
  --test-dir benchmark_casia/test \
  --skip-camera

# 4. 生成论文图表
python apps/recognition_system/draw_architecture.py
python apps/recognition_system/draw_camera_flow.py
python apps/recognition_system/draw_nn_structure.py
```

**完成后你将得到：**
- ✅ 完整的性能 LaTeX 表格
- ✅ 系统架构图 (PDF)
- ✅ 识别流程图 (PDF)
- ✅ 神经网络结构图 (PDF)
- ✅ 详细的 JSON 数据文件

直接用于论文撰写！📚
