@echo off
REM 人脸识别系统 Web UI 一键启动脚本
REM 自动激活 conda 环境并启动 Streamlit

echo.
echo ================================================
echo    人脸识别系统 Web UI - 一键启动
echo ================================================
echo.

REM 激活 conda 环境
echo 📦 正在激活 conda 环境...
call conda activate face_recognition310
if errorlevel 1 (
    echo.
    echo ❌ 错误：无法激活 conda 环境 face_recognition310
    echo.
    echo 请先创建环境：
    echo   conda env create -f environment.yml
    echo.
    pause
    exit /b 1
)

echo ✅ conda 环境已激活
echo.

REM 切换到应用目录
cd /d %~dp0
echo 📁 当前目录: %CD%
echo.

REM 检查 streamlit 是否安装
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ⚠️  streamlit 未安装，正在安装...
    pip install streamlit>=1.20.0
    if errorlevel 1 (
        echo.
        echo ❌ streamlit 安装失败
        echo.
        pause
        exit /b 1
    )
    echo ✅ streamlit 安装成功
    echo.
)

REM 启动应用
echo 🚀 正在启动 Web UI...
echo.
echo ================================================
echo  访问地址: http://localhost:8501
echo  按 Ctrl+C 停止服务
echo ================================================
echo.

python run.py web

pause
