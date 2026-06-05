#!/usr/bin/env bash
# ============================================================
# 人脸识别系统 - Linux / macOS 一键安装脚本
# Face Recognition System - Setup (Unix)
# ============================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  人脸识别系统 - 一键安装脚本                              ║"
echo "║  Face Recognition System - Setup (Linux/macOS)           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ========== 检查 Python ==========
echo "[1/5] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] 未找到 Python3！请先安装 Python 3.10+"
    echo "        Ubuntu: sudo apt install python3 python3-venv python3-pip"
    echo "        macOS:  brew install python@3.10"
    exit 1
fi
python3 --version
echo ""

# ========== 创建虚拟环境 ==========
echo "[2/5] 创建虚拟环境..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "[OK] 虚拟环境已创建: $VENV_DIR"
else
    echo "[SKIP] 虚拟环境已存在"
fi
echo ""

# ========== 激活虚拟环境 ==========
echo "[3/5] 激活虚拟环境..."
source "$VENV_DIR/bin/activate"
echo ""

# ========== 安装依赖 ==========
echo "[4/5] 安装 Python 依赖..."
echo "[INFO] 安装核心依赖（人脸检测 + 识别）..."
pip install --upgrade pip -q
pip install -r requirements.txt
echo ""

# ========== 创建目录 ==========
echo "[5/5] 创建必要目录..."
mkdir -p weights benchmark camera_eval
echo "[OK] 目录结构已就绪"
echo ""

# ========== 完成 ==========
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ 安装完成！                                           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  下一步:                                                 ║"
echo "║    1. 下载模型权重到 weights/                            ║"
echo "║       python download_weights.py                         ║"
echo "║                                                          ║"
echo "║    2. 创建/导入人脸数据库到 benchmark/                   ║"
echo "║       python download_weights.py --db                    ║"
echo "║                                                          ║"
echo "║    3. 启动系统                                           ║"
echo "║       python run.py                                      ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

read -p "是否现在下载模型权重? (y/n): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python download_weights.py
fi
