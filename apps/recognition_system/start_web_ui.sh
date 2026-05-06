#!/bin/bash
# 人脸识别系统 Web UI 一键启动脚本
# 自动激活 conda 环境并启动 Streamlit

echo ""
echo "================================================"
echo "   人脸识别系统 Web UI - 一键启动"
echo "================================================"
echo ""

# 激活 conda 环境
echo "📦 正在激活 conda 环境..."
eval "$(conda shell.bash hook)"
conda activate face_recognition310

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 错误：无法激活 conda 环境 face_recognition310"
    echo ""
    echo "请先创建环境："
    echo "  conda env create -f environment.yml"
    echo ""
    exit 1
fi

echo "✅ conda 环境已激活"
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")"
echo "📁 当前目录: $(pwd)"
echo ""

# 检查 streamlit 是否安装
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  streamlit 未安装，正在安装..."
    pip install streamlit>=1.20.0
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ streamlit 安装失败"
        echo ""
        exit 1
    fi
    echo "✅ streamlit 安装成功"
    echo ""
fi

# 启动应用
echo "🚀 正在启动 Web UI..."
echo ""
echo "================================================"
echo " 访问地址: http://localhost:8501"
echo " 按 Ctrl+C 停止服务"
echo "================================================"
echo ""

python run.py web
