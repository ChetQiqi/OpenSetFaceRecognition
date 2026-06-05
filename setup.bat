@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  人脸识别系统 - Windows 一键安装脚本                      ║
echo ║  Face Recognition System - Setup (Windows)               ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM ========== 检查 Python ==========
echo [1/5] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python！请先安装 Python 3.10
    echo         下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo.

REM ========== 创建虚拟环境 ==========
echo [2/5] 创建虚拟环境...
set VENV_DIR=%~dp0venv
if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
    echo [OK] 虚拟环境已创建: %VENV_DIR%
) else (
    echo [SKIP] 虚拟环境已存在
)
echo.

REM ========== 激活虚拟环境 ==========
echo [3/5] 激活虚拟环境...
call "%VENV_DIR%\Scripts\activate.bat"
echo.

REM ========== 安装依赖 ==========
echo [4/5] 安装 Python 依赖...
echo [INFO] 安装核心依赖（人脸检测 + 识别）...
pip install --upgrade pip -q
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARN] 部分依赖安装失败，请检查网络连接
)
echo.

REM ========== 创建目录 ==========
echo [5/5] 创建必要目录...
if not exist "weights" mkdir weights
if not exist "benchmark" mkdir benchmark
if not exist "camera_eval" mkdir camera_eval
echo [OK] 目录结构已就绪
echo.

REM ========== 完成 ==========
echo ╔══════════════════════════════════════════════════════════╗
echo ║  ✅ 安装完成！                                           ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║                                                          ║
echo ║  下一步:                                                 ║
echo ║    1. 下载模型权重到 weights/                            ║
echo ║       python download_weights.py                         ║
echo ║                                                          ║
echo ║    2. 创建/导入人脸数据库到 benchmark/                   ║
echo ║       python download_weights.py --db                    ║
echo ║                                                          ║
echo ║    3. 启动系统                                           ║
echo ║       python run.py                                      ║
echo ║                                                          ║
echo ║  可选: 安装 AI 肖像生成模块                              ║
echo ║       pip install git+https://github.com/TencentARC/PhotoMaker.git  ║
echo ║       pip install diffusers==0.29.2 transformers==4.43.0            ║
echo ║                                                          ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM 自动启动下载工具
echo 是否现在下载模型权重?
set /p DOWNLOAD="输入 y 继续，其他跳过: "
if /i "%DOWNLOAD%"=="y" (
    python download_weights.py
)

pause
