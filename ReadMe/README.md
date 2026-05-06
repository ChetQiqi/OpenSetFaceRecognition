# 人脸识别系统 (Face Recognition System)

## 📋 项目描述

这是一个基于深度学习的实时人脸识别系统，具有以下核心功能：

- ✅ **实时人脸识别**：通过摄像头或视频进行实时人脸识别
- ✅ **双模式识别**：
  - 视频中找人：识别视频中出现的所有人员
  - 验证ID：验证指定人员是否出现
- ✅ **人员管理**：注册、签到、删除人员等
- ✅ **图片识别**：单张图片的人脸识别
- ✅ **性能评估**：基于YouTube Faces数据集的性能基准测试
- ✅ **Web UI**：基于Streamlit的友好Web界面

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch
- OpenCV
- Streamlit

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动应用

#### Windows
```bash
run_app.bat
```

#### Linux / Mac
```bash
bash run_app.sh
```

然后在浏览器中打开：`http://localhost:8501`

## 📁 项目结构

```
RecognitionSystem/
├── apps/                           # Streamlit Web应用
│   └── recognition_system/
│       ├── core/                   # 核心算法模块
│       │   ├── model.py           # 深度学习模型
│       │   ├── detector.py        # 人脸检测器
│       │   ├── feature_db.py      # 特征库管理
│       │   ├── matcher.py         # 特征匹配
│       │   ├── tracker.py         # 多帧时序追踪
│       │   ├── operations.py      # 业务逻辑
│       │   └── camera_thread.py   # 摄像头线程
│       ├── streamlit_app.py       # Web界面主文件
│       └── add_person_to_db.py    # 人员注册脚本
│
├── backbones/                      # 模型骨干网络
│   └── iresnet.py                 # iResNet50架构
│
├── weights/                        # 预训练模型权重
│   └── adasin_best.pt             # 最优模型（PyTorch格式）
│
├── video_benchmark.py              # 视频性能评估脚本
├── run_video_benchmark.bat         # Windows评估脚本
├── run_video_benchmark.sh          # Linux/Mac评估脚本
│
├── run_app.bat                     # Windows启动脚本
├── run_app.sh                      # Linux/Mac启动脚本
│
├── requirements.txt                # Python依赖
├── README.md                       # 本文件
└── docs/                           # 详细文档
    ├── DUAL_MODE_RECOGNITION.md    # 双模式识别说明
    ├── IMPROVEMENTS_AUTO_EXIT.md   # 自动退出功能说明
    ├── REGISTRATION_OPTIMIZATION.md # 特征库优化说明
    └── VIDEO_BENCHMARK_GUIDE.md    # 视频评估指南
```

## 💻 功能模块

### 1. Streamlit Web 应用

主页面 `apps/recognition_system/streamlit_app.py`

**功能菜单：**
- 🏠 **主页**：系统信息和使用说明
- 👥 **人员管理**：注册人员、查看特征库、删除人员
- 🖼️ **图片识别**：上传图片进行人脸识别
- 💻 **摄像头实时识别**：实时视频流识别
  - 模式1：视频中找人（识别所有人员）
  - 模式2：验证ID（验证指定人员）
- 🎬 **视频识别**：上传视频文件进行识别
  - 模式1：视频中找人
  - 模式2：验证ID

### 2. 视频性能评估系统

用于评估系统性能的基准测试脚本：

```bash
# Windows
run_video_benchmark.bat

# Linux / Mac
bash run_video_benchmark.sh
```

**输出报告：**
- JSON格式的详细指标
- Markdown格式的可读报告
- LaTeX格式的表格（直接用于论文）

## 🎯 关键特性

### 双模式识别

#### 视频中找人
- 识别视频中出现的所有人员
- 显示识别到的人员列表（5个一行，卡片式显示）
- 自动检测陌生人（出现比例>5%时提示）

#### 验证ID
- 选择需要验证的人员ID
- 摄像头：≥5次检测为通过，≥50帧无检测为失败（自动退出）
- 视频：≥10帧检测为通过

### 时序稳定性

- 多帧投票机制：连续N帧识别为同一人才显示
- IoU匹配：视频帧间的人脸追踪
- 避免单帧误识别

### 特征库优化

- 智能选择：每人只注册最具代表性的几张图片
- 快速注册：大幅加快特征库构建速度
- 高效检索：更小的特征库，更快的匹配速度

## 📊 性能指标

基于iResNet50 + AdaSin Loss模型：

- **模型大小**：~167MB
- **输入尺寸**：112×112像素
- **输出维度**：512维特征向量
- **检测速度**：20+ FPS（CPU）
- **特征提取**：5+ FPS（CPU）

实际性能取决于硬件配置和输入分辨率。

## 🛠️ 模型信息

### 使用的模型

- **名称**：adasin_best.pt
- **架构**：iResNet50
- **Loss函数**：AdaSin Loss
- **格式**：PyTorch (.pt)
- **位置**：`weights/adasin_best.pt`

### 人脸检测

- **方法**：MTCNN
- **配置**：
  - 检测置信度阈值：0.9
  - 最小检测尺寸：40×40

## 📚 文档

详细文档位于项目根目录：

- **DUAL_MODE_RECOGNITION.md** - 双模式功能详解
- **IMPROVEMENTS_AUTO_EXIT.md** - 自动退出和结果保留机制
- **REGISTRATION_OPTIMIZATION.md** - 特征库优化说明
- **VIDEO_BENCHMARK_GUIDE.md** - 视频性能评估指南
- **README_COMPLETE.md** - 完整系统文档（论文用）
- **QUICKSTART.md** - 快速入门指南

## 🔧 配置

主要配置参数位于各模块的代码中：

### streamlit_app.py
```python
# 模型配置
weights_path = "weights/adasin_best.pt"
model_name = "iresnet50"
img_size = 112
device = "auto"  # 自动检测CUDA或CPU
```

### 识别参数（Web UI中可调）
- 识别阈值：0.0 - 1.0 (默认 0.45)
- 稳定帧数：2 - 10 (默认 3)
- 跳帧数：1 - 10 (默认 5)

## 📝 使用示例

### 1. 注册新人员

1. 打开系统，进入"👥 人员管理"页面
2. 在侧边栏选择"📂 注册新人员"
3. 输入人员ID（如：Person001）
4. 上传该人的多张照片（支持批量上传）
5. 点击"提交"完成注册

### 2. 实时人脸识别

1. 进入"📷 摄像头实时识别"页面
2. 选择功能模式：
   - "● 视频中找人"：识别所有出现的人
   - "○ 验证ID"：验证指定人员
3. 调整识别参数（可选）
4. 启动摄像头
5. 对着摄像头
6. 查看识别结果

### 3. 视频文件识别

1. 进入"🎬 视频识别"页面
2. 选择功能模式
3. 上传MP4/AVI/MOV/MKV视频
4. 调整参数并开始识别
5. 等待处理完成
6. 查看结果视频和统计数据

### 4. 性能评估

使用YouTube Faces数据集进行评估：

```bash
# Windows
run_video_benchmark.bat

# Linux / Mac
bash run_video_benchmark.sh

# 自定义参数（可选）
python video_benchmark.py \
  --images-dir "path/to/aligned_images_DB" \
  --videos-dir "path/to/videos" \
  --threshold 0.45 \
  --use-tracker \
  --stable-frames 5 \
  --max-images-per-person 10
```

## 🐛 常见问题

### Q: 摄像头无法启动？
A:
1. 检查摄像头是否正常（其他应用能否使用）
2. 调整摄像头ID（默认0，多摄像头可尝试1、2等）
3. 检查权限（Linux/Mac用户可能需要sudo）

### Q: 识别不准确？
A:
1. 确保人员注册了足够多的照片（至少5张）
2. 调低识别阈值（更容易匹配）
3. 增加稳定帧数（更稳定但更严格）
4. 改善光照条件（正面、充足光照最佳）

### Q: 系统很慢？
A:
1. 增加跳帧数（3→5→10）
2. 使用GPU（安装CUDA支持的PyTorch）
3. 减少特征库大小
4. 降低视频分辨率

### Q: CUDA提示不可用？
A:
1. 检查PyTorch版本：`python -c "import torch; print(torch.cuda.is_available())"`
2. 重新安装GPU版本：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
3. 检查NVIDIA驱动：`nvidia-smi`

## 📄 许可证

此项目用于学术和研究用途。

## 👨‍💻 作者

硕士学位论文 - 人脸识别系统

---

**需要帮助？**
- 查看详细文档：README_COMPLETE.md
- 查看快速指南：QUICKSTART.md
- 阅读功能说明：docs/ 目录文件
