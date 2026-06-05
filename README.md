<p align="center">
  <h1 align="center">🔍 OpenSet Face Recognition</h1>
  <p align="center">
    <strong>实时人脸识别系统</strong> — 基于深度学习的摄像头人脸检测、识别与管理系统
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.5-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-19-61dafb.svg" alt="React">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

---

## ✨ 功能特性

### 🎯 人脸识别
- 🎥 **实时摄像头识别** — 30 秒采样窗口，逐帧检测+特征提取+身份匹配，实时画面标注
- 🖼️ **图片批量识别** — 上传单张或多张图片，一键识别所有人脸身份
- 🏷️ **人脸跟踪** — 帧间 IoU 匹配，持续追踪同一人脸，减少重复识别

### 🧠 模型管理
- 📦 **多模型支持** — 内置 iResNet50（默认）、AdaFace、ArcFace 三种骨干网络，可扩展
- 🔄 **模型热切换** — Web 界面一键切换识别模型，无需重启服务
- 📥 **模型注册** — 上传自定义权重文件，自动识别模型架构并加载
- 📊 **模型对比** — 同一测试集上并行评估多个模型，生成对比报告

### 📈 评估与导出
- ⏱️ **性能指标采集** — 实时记录每帧 FPS、检测延迟、识别延迟、特征提取耗时
- 📋 **结果导出** — 一键导出 CSV（表格数据）+ JSON（结构化报告），支持 Excel 直接打开
- 📊 **评估报告** — 自动生成检测率、置信度分布、跟踪一致性等多维度指标
- 🎬 **评估录像** — 支持录制带标注叠加的评估视频，方便回溯验证
- 🔍 **逐帧审查** — 开发者控制台可查看每一帧的详细检测和识别结果

### 🎨 AI 肖像生成
- 🖼️ **PhotoMaker V1 + SDXL** — 基于上传照片生成高质量风格化人像
- 🎭 **6 种风格** — 商务照、证件照、古风写真、动漫、赛博朋克、职业形象照
- 💾 **本地保存** — 生成结果自动归档到本地目录，支持批量导出

### 🔐 系统管理
- 👤 **JWT 用户认证** — 注册/登录/Token 刷新，密码 bcrypt 加密存储
- 🗄️ **人员 CRUD** — 注册人脸、更新信息、删除人员、按条件搜索
- ⚙️ **环境变量配置** — 所有参数支持环境变量覆盖，方便容器化部署
- 🌐 **Swagger 文档** — FastAPI 自动生成交互式 API 文档（`/docs`）

---

## 🖥️ 界面功能

| 模块 | 说明 |
|------|------|
| 📊 Dashboard | 已注册人数、今日识别次数、GPU 状态、系统运行时长 |
| 👤 人员管理 | 注册/编辑/删除人员，上传人脸照片，浏览特征库 |
| 📷 图片识别 | 拖拽上传图片，实时显示检测框和识别结果 |
| 🎬 视频分析 | 摄像头/视频流实时分析，FPS & 延迟实时曲线 |
| 🎨 AI 肖像 | 选择人员 → 选择风格 → 一键生成，支持预览和下载 |
| ⚙️ 模型管理 | 模型列表、切换默认模型、注册新模型、模型对比 |
| 🔧 开发者控制台 | API 模拟器、性能日志、逐帧结果查看、原始 JSON 导出 |

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA | 12.1+（推荐，NVIDIA GPU） |
| GPU 显存 | ≥8 GB（AI 肖像功能需要） |
| RAM | ≥8 GB |

### 安装

**Windows：**
```batch
setup.bat
```

**Linux / macOS：**
```bash
bash setup.sh
```

**手动安装：**
```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 下载模型权重
python download_weights.py
```

### 启动

```bash
# 启动完整系统（FastAPI + React）
python run.py full

# 或仅启动 API
python run.py api

# 开发模式（热重载）
python run.py dev
```

启动后访问：
- **Web UI**：http://127.0.0.1:8000
- **API 文档**：http://127.0.0.1:8000/docs

---

## 📁 项目结构

```
OpenSetFaceRecognition/
│
├── apps/recognition_system/       # 🧠 核心应用（Python）
│   ├── core/                      #    识别引擎：检测→特征提取→匹配→跟踪
│   │   ├── operations.py          #       🔧 核心编排：检测+识别完整流程
│   │   ├── detector.py            #       👁️ MTCNN 人脸检测器
│   │   ├── model.py               #       🧬 iResNet50 特征提取模型
│   │   ├── feature_db.py          #       🗄️ SQLite 人脸特征数据库
│   │   ├── matcher.py             #       🎯 余弦相似度特征匹配
│   │   ├── tracker.py             #       🏷️ 帧间人脸轨迹跟踪
│   │   ├── adaptive_threshold.py  #       📐 自适应阈值算法
│   │   ├── metrics.py             #       📊 FPS/延迟/准确率计算
│   │   └── cli.py                 #       ⌨️ 命令行识别工具
│   │
│   ├── api/                       #    🌐 FastAPI REST 后端
│   │   ├── main.py                #       🚀 应用入口、路由注册、静态文件
│   │   ├── auth_routes.py         #       🔐 登录/注册/Token 刷新
│   │   └── schemas.py             #       📋 Pydantic 请求/响应模型
│   │
│   ├── auth/                      #    🔑 JWT 认证模块
│   │   ├── security.py            #       🔒 密码哈希 + Token 生成/验证
│   │   ├── models.py              #       👤 User ORM 模型
│   │   ├── service.py             #       🛠️ 用户注册/登录业务逻辑
│   │   └── dependencies.py        #       🔗 FastAPI 依赖注入（当前用户）
│   │
│   ├── services/                  #    ⚙️ 业务逻辑层
│   │   ├── inference_service.py   #       🔮 推理服务：图片/视频人脸识别
│   │   ├── identity_service.py    #       👥 人员管理服务
│   │   ├── portrait_service.py    #       🎨 AI 肖像生成 (PhotoMaker + SDXL)
│   │   ├── eval_service.py        #       📈 性能评估服务
│   │   ├── video_service.py       #       🎬 视频分析服务
│   │   └── developer_service.py   #       🔧 开发者工具服务
│   │
│   ├── repositories/              #    🗃️ 数据访问层（DAO）
│   │   ├── identity_repository.py #       人员 CRUD 操作
│   │   ├── user_repository.py     #       用户 CRUD 操作
│   │   └── model_repository.py    #       模型管理操作
│   │
│   ├── models/                    #    🏗️ 模型注册 & 管理
│   └── config.py                  #    ⚙️ 全局配置（支持环境变量覆盖）
│
├── frontend/                      # 🖥️ React 管理后台（TypeScript）
│   ├── src/components/            #    UI 组件
│   │   ├── DashboardView.tsx      #       📊 系统仪表盘
│   │   ├── PeopleManagementView   #       👥 人员管理
│   │   ├── ImageRecognitionView   #       📷 图片识别
│   │   ├── VideoAnalysisView.tsx  #       🎬 视频分析
│   │   ├── AIPortraitView.tsx     #       🎨 AI 肖像生成
│   │   ├── DeveloperConsoleView   #       🔧 开发者控制台
│   │   ├── Sidebar.tsx            #       📱 侧边导航栏
│   │   ├── LoginPage.tsx          #       🔐 登录页
│   │   └── ...
│   ├── src/api/                   #    🌐 API 客户端（axios）
│   ├── src/context/               #    🗂️ React Context（Auth、Toast）
│   ├── src/hooks/                 #    🪝 自定义 Hooks
│   └── src/types.ts               #    📝 TypeScript 类型定义
│
├── backbones/                     # 🏗️ 模型骨干网络
│   ├── iresnet.py                 #    iResNet50/100/152
│   └── mobilefacenet.py           #    MobileFaceNet（轻量）
│
├── tools/                         # 🔧 实用工具
│   ├── check_gpu.py               #    GPU 检测与性能测试
│   ├── check_db.py                #    数据库完整性检查
│   └── register_random_persons.py #    批量随机注册工具
│
├── weights/                       # 📦 模型权重（需下载）
│   ├── model_best.pt              #    iResNet50 (167 MB) — 必需
│   └── hf_cache/                  #    HuggingFace 缓存 — AI 肖像用
│
├── benchmark/                     # 🗄️ 人脸数据库（本地数据）
├── run.py                         # 🚀 统一入口（菜单/命令行）
├── download_weights.py            # ⬇️ 权重下载工具
├── setup.bat / setup.sh           # ⚡ 一键安装脚本
├── requirements.txt               # 📋 Python 依赖
├── LICENSE                        # 📄 MIT 开源协议
└── README.md                      # 📖 本文件
```

---

## ⌨️ 命令行使用

```bash
python run.py                  # 交互式菜单
python run.py api              # 启动 API (http://localhost:8000)
python run.py full             # 构建 React + 启动 API
python run.py dev              # API + React 热重载
python run.py manager          # 人脸库管理
python run.py register <目录>  # 批量注册人脸
python run.py recognize-image <图片>  # 图片识别
python run.py recognize-camera       # 摄像头识别
```

---

## 🎨 AI 肖像生成

> ⚠️ 可选功能，需要额外安装依赖和下载模型（~10 GB）

```bash
# 安装 PhotoMaker
pip install git+https://github.com/TencentARC/PhotoMaker.git

# 安装精确版本依赖
pip install diffusers==0.29.2 transformers==4.43.0 huggingface_hub==0.36.2

# 下载模型
python download_weights.py --hf
```

支持风格：💼 商务照 · 🪪 证件照 · 🏯 古风写真 · 🎭 动漫 · 🤖 赛博朋克

---

## 🔧 配置

编辑 `apps/recognition_system/config.py`，或通过环境变量覆盖：

```bash
export RECOGNITION_DB_PATH=/data/my_faces.db
export RECOGNITION_DEVICE=cuda
export RECOGNITION_THRESHOLD=0.45
export RECOGNITION_DETECTOR=mtcnn
```

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RECOGNITION_DB_PATH` | `benchmark/YTF_100p.db` | 数据库路径 |
| `RECOGNITION_WEIGHTS_PATH` | `weights/model_best.pt` | 模型权重 |
| `RECOGNITION_DEVICE` | `auto` | cuda / cpu |
| `RECOGNITION_THRESHOLD` | `0.45` | 识别阈值 |
| `RECOGNITION_DETECTOR` | `mtcnn` | 检测器后端 |

---

## 🐛 常见问题

### 启动报错 `model_best.pt` 找不到

```bash
python download_weights.py
```
按照提示下载模型权重文件。

### MTCNN 速度慢

- 降低检测频率，每 2-3 帧处理一次
- 替换为 YOLOv8-Face 检测器（5-10 倍提速）

### CUDA Out of Memory

- 设置 `RECOGNITION_DEVICE=cpu` 使用 CPU 模式
- 关闭 AI 肖像生成功能

### NumPy 版本冲突

```bash
pip install "numpy<2.0"
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源，仅供学术研究和学习使用。

---

**Made with ❤️ by [ChetQiqi](https://github.com/ChetQiqi)**
