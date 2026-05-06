#!/usr/bin/env python3
"""
Face Recognition System - CLI Entry Point
快速启动人脸识别系统的各种工具
"""

import sys
import argparse
from pathlib import Path

# 确保导入路径正确
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def show_menu():
    """显示主菜单"""
    print("\n" + "=" * 50)
    print("   人脸识别系统 (Face Recognition System)")
    print("=" * 50)
    print("1. 🌐 启动 Web UI (Start Web UI)")
    print("2. 🗂️  人脸库管理 (Face Database Manager)")
    print("3. 📝 注册人脸 (Register Faces)")
    print("4. 🔍 识别图片 (Recognize Image)")
    print("5. 📹 识别摄像头 (Camera Recognition)")
    print("0. 退出 (Exit)")
    print("=" * 50)


def main():
    """主程序"""
    if len(sys.argv) > 1:
        # 如果有命令行参数，直接运行对应模块
        command = sys.argv[1]
        args = sys.argv[2:]

        if command == "web" or command == "streamlit" or command == "gui":
            import subprocess
            streamlit_path = "apps\\recognition_system\\streamlit_app.py"
            print("🚀 启动 Web UI...")
            print(f"📁 应用路径: {streamlit_path}")
            print("\n访问地址: http://localhost:8501")
            print("按 Ctrl+C 停止服务\n")
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_path)])
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
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  web, gui, streamlit - Start Web UI")
            print("  manager        - Face database manager")
            print("  register       - Register faces from directory")
            print("  recognize-image - Recognize faces in image")
            print("  recognize-camera - Real-time camera recognition")
            print("  cli            - Command-line interface")
            sys.exit(1)
    else:
        # 否则显示菜单（仅支持脚本直接运行时）
        show_menu()
        choice = input("\n请选择 (Select): ").strip()

        try:
            if choice == "1":
                print("🚀 启动 Web UI...")
                import subprocess
                streamlit_path = Path(__file__).parent / "streamlit_app.py"
                print(f"📁 应用路径: {streamlit_path}")
                print("\n访问地址: http://localhost:8501")
                print("按 Ctrl+C 停止服务\n")
                subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_path)])
            elif choice == "2":
                print("启动人脸库管理...")
                from apps.recognition_system.add_person_to_db import main
                main()
            elif choice == "3":
                print("启动注册工具...")
                from apps.recognition_system.core.cli import main
                sys.argv = ["python", "register-dir", "--help"]
                main()
            elif choice == "0":
                print("退出系统")
                sys.exit(0)
            else:
                print("无效选择 (Invalid choice)")
        except Exception as e:
            print(f"错误 (Error): {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
