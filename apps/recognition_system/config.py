import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 确保关键目录存在
(Path(__file__).resolve().parents[2] / "weights").mkdir(parents=True, exist_ok=True)
(Path(__file__).resolve().parents[2] / "benchmark").mkdir(parents=True, exist_ok=True)


def _env_str(key: str, default: str) -> str:
    """从环境变量读取字符串配置"""
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    """从环境变量读取浮点数配置"""
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _env_int(key: str, default: int) -> int:
    """从环境变量读取整数配置"""
    val = os.environ.get(key)
    return int(val) if val is not None else default


@dataclass(frozen=True)
class AppConfig:
    # ---- 路径配置（支持环境变量覆盖）----
    db_path: str = _env_str(
        "RECOGNITION_DB_PATH",
        str((PROJECT_ROOT / "benchmark" / "YTF_100p.db").resolve()),
    )
    auth_db_path: str = _env_str(
        "RECOGNITION_AUTH_DB_PATH",
        str((PROJECT_ROOT / "benchmark" / "auth.db").resolve()),
    )
    weights_path: str = _env_str(
        "RECOGNITION_WEIGHTS_PATH",
        str((PROJECT_ROOT / "weights" / "model_best.pt").resolve()),
    )

    # ---- 模型配置 ----
    model_name: str = _env_str("RECOGNITION_MODEL", "iresnet50")
    img_size: int = _env_int("RECOGNITION_IMG_SIZE", 112)
    device: str = _env_str("RECOGNITION_DEVICE", "auto")

    # ---- 检测器配置 ----
    detector_backend: str = _env_str("RECOGNITION_DETECTOR", "mtcnn")
    detector_conf_threshold: float = _env_float("RECOGNITION_DET_CONF", 0.60)
    detector_min_size: int = _env_int("RECOGNITION_DET_MIN_SIZE", 40)

    # ---- 识别配置 ----
    recognition_threshold: float = _env_float("RECOGNITION_THRESHOLD", 0.45)
    gallery_mode: str = _env_str("RECOGNITION_GALLERY_MODE", "mean")
    match_reduce: str = _env_str("RECOGNITION_MATCH_REDUCE", "topk_mean")
    topk: int = _env_int("RECOGNITION_TOPK", 3)

    # ---- JWT 配置 ----
    jwt_secret_key: str = _env_str("RECOGNITION_JWT_SECRET", "change-this-in-production")
    jwt_algorithm: str = _env_str("RECOGNITION_JWT_ALGORITHM", "HS256")
    jwt_access_token_expire_minutes: int = _env_int("RECOGNITION_JWT_EXPIRE", 720)


def get_config() -> AppConfig:
    """获取应用配置（单例模式）"""
    return AppConfig()
