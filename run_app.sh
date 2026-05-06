#!/bin/bash
# 人脸识别系统启动脚本 (Linux/Mac)

echo "=========================================="
echo "🚀 启动人脸识别系统"
echo "=========================================="
echo ""

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装，请先安装Python 3.8+"
    exit 1
fi

# 检查streamlit是否安装
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Streamlit未安装，正在安装依赖..."
    pip install -r requirements.txt
    echo ""
fi

# 启动Streamlit应用
echo "📱 正在启动Streamlit Web界面..."
echo "🌐 应用地址: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止应用"
echo ""

cd "$(dirname "$0")"
streamlit run apps/recognition_system/streamlit_app.py
