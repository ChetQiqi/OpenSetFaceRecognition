#!/usr/bin/env python3
"""
模型权重 & 数据库下载工具
========================

用法：
    python download_weights.py              # 交互式菜单
    python download_weights.py --all        # 下载全部
    python download_weights.py --model      # 仅下载识别模型
    python download_weights.py --db         # 仅下载示例数据库
    python download_weights.py --url <URL>  # 从指定 URL 下载模型

═══════════════════════════════════════════════════════════
  ⚠️ 分发者配置指南（How to host model weights）
═══════════════════════════════════════════════════════════

方式 1：百度网盘（国内推荐）
  1. 上传 model_best.pt 到百度网盘
  2. 右键 → 分享 → 创建链接
  3. 把链接和提取码填到下面 PAN_URL / PAN_CODE
  4. 用户运行此脚本会自动打开浏览器

方式 2：GitHub Releases（国际推荐，免费 2GB）
  git tag -a v1.0 -m "Release model weights"
  git push --tags
  然后在 GitHub 网页上：
  Releases → Create Release → 上传 model_best.pt
  把下载 URL 填到下面 GITHUB_RELEASE_URL

方式 3：任何直链 URL
  python download_weights.py --url https://你的服务器/model_best.pt

═══════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import hashlib
import webbrowser
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# ╔══════════════════════════════════════════════════════════╗
# ║  ★ 分发者：在这里填写你的下载链接                        ║
# ╚══════════════════════════════════════════════════════════╝

# 百度网盘（修改为你的链接）
PAN_URL = ""        # 例如: "https://pan.baidu.com/s/xxxxx"
PAN_CODE = ""       # 提取码

# GitHub Releases 直链（创建 Release 后自动生成）
GITHUB_RELEASE_URL = "https://github.com/ChetQiqi/OpenSetFaceRecognition/releases/download/v1.0/model_best.pt"

# 其他直链
DIRECT_URL = ""     # 例如: "https://你的服务器/model_best.pt"

# 文件 SHA256（可选，用于校验下载完整性）
MODEL_SHA256 = ""   # 运行 sha256sum model_best.pt 获取

# ╔══════════════════════════════════════════════════════════╗
# ║  代码 — 无需修改以下内容                                 ║
# ╚══════════════════════════════════════════════════════════╝

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
MODEL_FILENAME = "model_best.pt"


def check_existing() -> dict:
    """检查已有文件"""
    return {
        "model": (WEIGHTS_DIR / MODEL_FILENAME).exists(),
        "hf_cache": (WEIGHTS_DIR / "hf_cache").exists(),
        "db": (BENCHMARK_DIR / "YTF_100p.db").exists(),
    }


def print_status(status: dict):
    """打印状态"""
    print("\n" + "=" * 60)
    print("  当前状态 (Current Status)")
    print("=" * 60)
    items = [
        (f"识别模型 ({MODEL_FILENAME})", status["model"]),
        ("HuggingFace 缓存 (AI肖像用)", status["hf_cache"]),
        ("示例数据库 (YTF_100p.db)", status["db"]),
    ]
    for name, ok in items:
        print(f"  {'✅' if ok else '❌'}  {name}")
    print("=" * 60)


def sha256_file(path: Path) -> str:
    """计算文件 SHA256"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_sha256: str = "") -> bool:
    """
    从 URL 下载文件，支持断点续传和进度条。
    返回 True 表示下载成功。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在
    if dest.exists():
        if expected_sha256 and sha256_file(dest) == expected_sha256:
            print(f"[OK] 文件已存在且校验通过: {dest.name}")
            return True
        else:
            print(f"[INFO] 文件已存在，将覆盖: {dest.name}")

    print(f"[DOWNLOAD] {url}")
    print(f"[INFO] 保存到: {dest}")
    print(f"[INFO] 开始下载... (文件较大，请耐心等待)")

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urlopen(req)
        total_size = int(response.headers.get("Content-Length", 0))

        downloaded = 0
        chunk_size = 8192

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    pct = downloaded * 100 / total_size
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    bar_len = 30
                    filled = int(bar_len * downloaded / total_size)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  [{bar}] {pct:.0f}%  {mb_down:.1f}/{mb_total:.1f} MB", end="")
                else:
                    mb_down = downloaded / (1024 * 1024)
                    print(f"\r  已下载: {mb_down:.1f} MB", end="")

        print()  # 换行

        # 校验
        if expected_sha256:
            actual = sha256_file(dest)
            if actual != expected_sha256:
                print(f"[ERROR] SHA256 校验失败！")
                print(f"  期望: {expected_sha256}")
                print(f"  实际: {actual}")
                print(f"  请重新下载")
                dest.unlink(missing_ok=True)
                return False
            print("[OK] SHA256 校验通过 ✓")

        print(f"[OK] 下载完成: {dest.name}")
        return True

    except URLError as e:
        print(f"\n[ERROR] 下载失败: {e}")
        print("[HINT] 请检查网络连接，或尝试手动下载")
        return False
    except KeyboardInterrupt:
        print(f"\n[INFO] 下载中断，已保存部分文件: {dest.name}")
        print(f"       下次运行将重新下载")
        return False


def download_model(auto_url: str = None):
    """下载识别模型"""
    print(f"\n[STEP] 下载识别模型 ({MODEL_FILENAME}, ~167 MB)")

    # 决定使用哪个 URL
    url = auto_url or DIRECT_URL or GITHUB_RELEASE_URL

    if url:
        dest = WEIGHTS_DIR / MODEL_FILENAME
        success = download_file(url, dest, MODEL_SHA256)
        if success:
            print("\n[OK] 识别模型就绪！可以运行 python run.py 启动系统")
        else:
            print("\n[INFO] 自动下载失败，请尝试手动下载（见下方说明）")
            print_baidu_pan_help()
        return

    # 没有配置 URL，显示手动下载说明
    if PAN_URL:
        print(f"\n[INFO] 百度网盘下载:")
        print(f"  链接: {PAN_URL}")
        print(f"  提取码: {PAN_CODE}")
        print(f"  下载后把 {MODEL_FILENAME} 放到: {WEIGHTS_DIR}")
        print(f"\n  是否现在打开浏览器？")
        try:
            choice = input("  输入 y 打开链接: ").strip().lower()
            if choice == "y":
                webbrowser.open(PAN_URL)
                print(f"  [OK] 已打开浏览器")
                print(f"  下载完成后，把 {MODEL_FILENAME} 放到:")
                print(f"    {WEIGHTS_DIR}")
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print_baidu_pan_help()


def print_baidu_pan_help():
    """打印手动获取说明"""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       模型权重获取指南                                    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📦 {MODEL_FILENAME} (iResNet50, ~167 MB)              ║
║     请从以下方式获取后放到:                               ║
║     {WEIGHTS_DIR}                   ║
║                                                          ║
║  方式 1: 项目分发者的网盘链接                             ║
║     联系分发者获取百度网盘 / Google Drive 链接            ║
║                                                          ║
║  方式 2: 自己训练模型                                    ║
║     iResNet50 + MS1MV2 数据集训练                        ║
║                                                          ║
║  方式 3: 使用其他 ArcFace/AdaFace 权重                   ║
║     将 .pt 文件放到 weights/ 目录即可                     ║
║     在 config.py 中修改 weights_path                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


def download_from_huggingface():
    """从 HuggingFace 下载 PhotoMaker 模型"""
    print("\n[STEP] 下载 AI 肖像生成模型 (PhotoMaker + SDXL, ~10 GB)")

    os.environ["HF_HUB_CACHE"] = str(WEIGHTS_DIR / "hf_cache")

    try:
        from huggingface_hub import snapshot_download
        print("[INFO] 正在从 HuggingFace 下载 TencentARC/PhotoMaker...")
        snapshot_download(
            "TencentARC/PhotoMaker",
            cache_dir=str(WEIGHTS_DIR / "hf_cache"),
            resume_download=True,
        )
        print("[OK] PhotoMaker 模型下载完成！")
        print("[INFO] SDXL 模型将在首次运行时自动下载")
    except ImportError:
        print("[SKIP] huggingface_hub 未安装")
        print("       安装: pip install huggingface_hub")
    except Exception as e:
        print(f"[WARN] 下载失败: {e}")
        print("       可以手动下载后放到 weights/hf_cache/")


def create_demo_db():
    """创建空的示例数据库"""
    print("\n[STEP] 创建示例数据库...")
    try:
        from apps.recognition_system.core.feature_db import FeatureDatabase

        db_path = BENCHMARK_DIR / "YTF_100p.db"
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

        db = FeatureDatabase(str(db_path))
        db.create_tables()
        db.close()

        print(f"[OK] 空数据库已创建: {db_path}")
        print("   使用 'python run.py manager' 添加人脸")
    except Exception as e:
        print(f"[WARN] 创建数据库失败: {e}")
        print("   这不影响已有数据库的使用")


def main():
    parser = argparse.ArgumentParser(
        description="模型权重 & 数据库下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_weights.py                        # 交互式菜单
  python download_weights.py --status               # 查看当前状态
  python download_weights.py --model                # 下载识别模型
  python download_weights.py --url https://xxx/pt   # 从 URL 下载
  python download_weights.py --db                   # 创建空数据库
  python download_weights.py --all                  # 下载全部
        """,
    )
    parser.add_argument("--all", action="store_true", help="下载所有可自动下载的内容")
    parser.add_argument("--model", action="store_true", help="下载识别模型（167 MB）")
    parser.add_argument("--hf", action="store_true", help="下载 HuggingFace 模型（~10 GB）")
    parser.add_argument("--db", action="store_true", help="创建空的示例数据库")
    parser.add_argument("--status", action="store_true", help="仅查看当前状态")
    parser.add_argument("--url", type=str, default=None, help="从指定 URL 下载模型权重")
    args = parser.parse_args()

    # 确保目录存在
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    status = check_existing()
    print_status(status)

    if args.status:
        return

    # --url: 直接下载
    if args.url:
        dest = WEIGHTS_DIR / MODEL_FILENAME
        download_file(args.url, dest, MODEL_SHA256)
        return

    # --all / --model
    if args.all or args.model:
        download_model()

    # --all / --hf
    if args.all or args.hf:
        download_from_huggingface()

    # --all / --db
    if args.all or args.db:
        create_demo_db()

    # 无参数: 交互式菜单
    if not any([args.all, args.model, args.hf, args.db]):
        print("\n请选择操作:")
        print(f"  1. 下载识别模型 ({MODEL_FILENAME}, ~167 MB)")
        print("  2. 下载 AI 肖像模型 (PhotoMaker/SDXL, ~10 GB)")
        print("  3. 创建空数据库")
        print("  4. 查看手动下载说明")
        print("  0. 退出")

        try:
            choice = input("\n选择 (0-4): ").strip()
            if choice == "1":
                download_model()
            elif choice == "2":
                download_from_huggingface()
            elif choice == "3":
                create_demo_db()
            elif choice == "4":
                print_baidu_pan_help()
            else:
                print("退出")
        except (EOFError, KeyboardInterrupt):
            print("\n退出")


if __name__ == "__main__":
    main()
