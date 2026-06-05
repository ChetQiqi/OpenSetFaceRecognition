#!/usr/bin/env python3
"""
OpenSet Face Recognition — 应用入口
=====================================

用法：
    python run.py                  # 交互式菜单
    python run.py api              # 启动 FastAPI + React 前端
    python run.py dev              # API + React 开发模式（热重载）
    python run.py build            # 仅构建 React 前端
    python run.py manager          # 人脸库管理
    python run.py register <目录>  # 批量注册人脸
    python run.py recognize-image <图片>  # 图片识别
    python run.py recognize-camera       # 摄像头实时识别
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_DIR = Path(__file__).resolve().parent
REACT_UI_DIR = PROJECT_DIR / "frontend"


def build_react() -> bool:
    """构建 React 前端。成功返回 True。"""
    if not (REACT_UI_DIR / "package.json").exists():
        print("[WARN] React 项目未找到，跳过构建。")
        return True  # 不是错误——可能用户只需要 API

    print("[BUILD] 构建 React 前端...")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    try:
        subprocess.run([npm, "install"], cwd=str(REACT_UI_DIR), check=True)
        subprocess.run([npm, "run", "build"], cwd=str(REACT_UI_DIR), check=True)
        print("[BUILD] React 前端构建成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] React 构建失败: {e}")
        return False
    except FileNotFoundError:
        print("[ERROR] npm 未找到。请安装 Node.js: https://nodejs.org")
        return False


def start_api():
    """启动 FastAPI（含 React 前端）。"""
    print("\n" + "=" * 60)
    print("  OpenSet Face Recognition — API Server")
    print("=" * 60)
    print(f"\n  Web UI:  http://127.0.0.1:8000")
    print(f"  API 文档: http://127.0.0.1:8000/docs")
    print(f"  按 Ctrl+C 停止\n")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "apps.recognition_system.api.main:app",
        "--host", "127.0.0.1", "--port", "8000",
    ])


def start_react_dev():
    """启动 React 开发服务器（热重载）。"""
    print("[DEV] 启动 React 开发服务器...")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    try:
        subprocess.run([npm, "run", "dev"], cwd=str(REACT_UI_DIR))
    except FileNotFoundError:
        print("[ERROR] npm 未找到。")


def show_menu():
    """显示主菜单"""
    print("""
╔══════════════════════════════════════════════════════════╗
║     OpenSet Face Recognition — 人脸识别系统              ║
╠══════════════════════════════════════════════════════════╣
║  1. 启动完整系统 (FastAPI + React)                       ║
║  2. 仅启动 API (开发用)                                  ║
║  3. API + React 开发模式 (热重载)                        ║
║  4. 构建 React 前端                                      ║
║  5. 人脸数据库管理                                       ║
║  0. 退出                                                 ║
╚══════════════════════════════════════════════════════════╝
""")


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        args = sys.argv[2:]

        if command == "api":
            start_api()
        elif command == "full":
            if build_react():
                start_api()
        elif command == "dev":
            api_proc = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "apps.recognition_system.api.main:app",
                "--host", "127.0.0.1", "--port", "8000",
            ])
            try:
                time.sleep(2)
                start_react_dev()
            finally:
                api_proc.terminate()
        elif command == "build":
            build_react()
        elif command == "react-dev":
            start_react_dev()
        elif command == "manager":
            from apps.recognition_system.add_person_to_db import main as manager_main
            manager_main()
        elif command == "register":
            from apps.recognition_system.core.cli import main as cli_main
            sys.argv = [sys.argv[0], "register-dir"] + args
            cli_main()
        elif command == "recognize-image":
            from apps.recognition_system.core.cli import main as cli_main
            sys.argv = [sys.argv[0], "recognize-image"] + args
            cli_main()
        elif command == "recognize-camera":
            from apps.recognition_system.core.cli import main as cli_main
            sys.argv = [sys.argv[0], "recognize-camera"] + args
            cli_main()
        elif command == "cli":
            from apps.recognition_system.core.cli import main as cli_main
            sys.argv = [sys.argv[0]] + args
            cli_main()
        else:
            print(f"未知命令: {command}")
            print("\n可用命令:")
            print("  api              启动 FastAPI (含 React) → http://127.0.0.1:8000")
            print("  full             构建 React + 启动 API")
            print("  dev              API + React 热重载开发模式")
            print("  build            仅构建 React 前端")
            print("  react-dev        仅启动 React 开发服务器")
            print("  manager          人脸库管理")
            print("  register <dir>   批量注册人脸")
            print("  recognize-image <img>  图片识别")
            print("  recognize-camera       摄像头实时识别")
            sys.exit(1)
    else:
        show_menu()
        try:
            choice = input("请选择 (0-5): ").strip()
            if choice == "1":
                if build_react():
                    start_api()
            elif choice == "2":
                start_api()
            elif choice == "3":
                api_proc = subprocess.Popen([
                    sys.executable, "-m", "uvicorn",
                    "apps.recognition_system.api.main:app",
                    "--host", "127.0.0.1", "--port", "8000",
                ])
                try:
                    time.sleep(2)
                    start_react_dev()
                finally:
                    api_proc.terminate()
            elif choice == "4":
                build_react()
            elif choice == "5":
                print("启动人脸库管理...")
                from apps.recognition_system.add_person_to_db import main
                main()
            elif choice == "0":
                print("退出，再见！")
                sys.exit(0)
            else:
                print("无效选择")
        except KeyboardInterrupt:
            print("\n退出")
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
