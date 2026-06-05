"""
AI 肖像生成服务 (AI Portrait Generation Service)

基于 PhotoMaker V2 + Stable Diffusion XL 实现。
使用 HuggingFace Diffusers 框架，支持多种风格的人物肖像生成。

依赖:
    - diffusers >= 0.27.0
    - photomaker (pip install git+https://github.com/TencentARC/PhotoMaker.git)
    - torch >= 2.0
    - transformers

使用方式:
    from apps.recognition_system.services.portrait_service import PortraitService

    service = PortraitService()
    result_path = service.generate(
        reference_image_path="path/to/photo.jpg",
        style="business",
        output_dir="output/portraits",
        seed=42,
    )
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 风格定义 — 提示词映射
# ---------------------------------------------------------------------------
# PhotoMaker 要求 prompt 格式: "a [类别词] img [风格描述]"
# 触发词 "img" 必须紧跟在类别词 (man/woman/person) 之后

STYLE_PROMPTS: Dict[str, Dict[str, str]] = {
    "business": {
        "label": "💼 商务照",
        "prompt": "a person img wearing a formal business suit, professional headshot, clean white background, corporate portrait photography, soft studio lighting, photorealistic, high quality, sharp focus",
        "negative_prompt": "nsfw, worst quality, low quality, blurry, distorted face, bad anatomy, cartoon, painting, 3d render",
    },
    "id_photo": {
        "label": "🪪 证件照",
        "prompt": "a person img in a formal ID photo, front-facing portrait, neutral expression, plain white background, passport photo style, even lighting, photorealistic, high quality",
        "negative_prompt": "nsfw, worst quality, low quality, blurry, smile, open mouth, glasses, hat, cartoon, painting, 3d render",
    },
    "ancient_chinese": {
        "label": "🏮 古风写真",
        "prompt": "a person img in traditional Chinese hanfu clothing, ancient Chinese style portrait, elegant pose, classical Chinese beauty aesthetic, ink wash painting atmosphere, soft warm lighting, artistic, high quality",
        "negative_prompt": "nsfw, worst quality, low quality, blurry, modern clothing, western style, cartoon, 3d render, distorted face",
    },
    "anime": {
        "label": "🎨 动漫风格",
        "prompt": "anime style portrait of a person img, manga illustration, vibrant colors, studio ghibli inspired, Japanese animation aesthetic, clean lineart, beautiful shading, high quality",
        "negative_prompt": "nsfw, worst quality, low quality, photorealistic, 3d render, realistic, blurry, distorted face",
    },
    "cyberpunk": {
        "label": "🤖 赛博朋克",
        "prompt": "a person img in cyberpunk style, neon lights background, futuristic city at night, high-tech atmosphere, cybernetic aesthetic, blade runner inspired, dramatic lighting, high quality, detailed",
        "negative_prompt": "nsfw, worst quality, low quality, blurry, natural daylight, rural, cartoon, painting, distorted face",
    },
    "professional": {
        "label": "👔 职业形象照",
        "prompt": "a person img in business casual attire, professional corporate portrait, modern office background, confident friendly expression, natural window lighting, photorealistic, high quality, sharp focus",
        "negative_prompt": "nsfw, worst quality, low quality, blurry, casual t-shirt, distorted face, cartoon, painting, 3d render",
    },
}


@dataclass
class PortraitResult:
    """生成结果"""

    style: str
    style_label: str
    reference_image_path: str
    output_path: str
    prompt_used: str
    seed: int
    generation_time_seconds: float
    width: int
    height: int


@dataclass
class PortraitServiceConfig:
    """PortraitService 配置"""

    # PhotoMaker 模型路径 (HuggingFace 模型 ID 或本地路径)
    base_model_id: str = "SG161222/RealVisXL_V4.0"
    photomaker_model_dir: str = ""  # PhotoMaker 权重目录，为空则自动下载

    # 生成参数
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    start_merge_step: int = 10
    height: int = 1024
    width: int = 1024
    trigger_word: str = "img"

    # 设备
    device: str = "cuda"  # cuda / cpu

    # 输出目录 (相对于项目根目录)
    output_subdir: str = "ai_portraits"

    # 是否使用 CPU offload (节省显存，8GB 显卡建议开启)
    enable_cpu_offload: bool = True

    # HuggingFace 镜像 (国内用户建议设置为 "https://hf-mirror.com")
    hf_endpoint: str = ""

    # HuggingFace 缓存目录 (None=默认 ~/.cache/huggingface)
    hf_cache_dir: str = "weights\\hf_cache"


class PortraitService:
    """
    AI 肖像生成服务。

    封装 PhotoMaker V2 + Stable Diffusion XL 管线，
    提供一键式人物肖像风格转换功能。

    使用示例:
        config = PortraitServiceConfig(device="cuda")
        service = PortraitService(config)
        result = service.generate("path/to/photo.jpg", "business")
        print(f"生成完成: {result.output_path}")
    """

    # 风格列表
    STYLES = list(STYLE_PROMPTS.keys())

    def __init__(self, config: Optional[PortraitServiceConfig] = None):
        self.config = config or PortraitServiceConfig()
        self._pipe = None
        self._loaded = False

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    @classmethod
    def get_styles(cls) -> List[Dict[str, str]]:
        """返回可用风格列表（用于 UI 展示）。"""
        return [
            {"key": key, "label": info["label"]}
            for key, info in STYLE_PROMPTS.items()
        ]

    @classmethod
    def get_style_info(cls, style: str) -> Optional[Dict[str, str]]:
        """获取指定风格的详细信息。"""
        return STYLE_PROMPTS.get(style)

    def is_loaded(self) -> bool:
        """管线是否已加载。"""
        return self._loaded and self._pipe is not None

    def generate(
        self,
        reference_image: str | Image.Image | np.ndarray,
        style: str,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        start_merge_step: Optional[int] = None,
    ) -> PortraitResult:
        """
        生成 AI 肖像。

        参数:
            reference_image: 参考人物照片 (路径/PIL Image/numpy 数组)
            style: 风格 (可选: business, id_photo, ancient_chinese, anime, cyberpunk, professional)
            output_dir: 输出目录 (默认使用配置中的 output_subdir)
            seed: 随机种子 (None 则随机)
            num_inference_steps: 推理步数 (覆盖配置)
            guidance_scale: 引导强度 (覆盖配置)
            start_merge_step: ID 注入起始步 (覆盖配置)

        返回:
            PortraitResult 包含生成结果的元信息

        异常:
            ValueError: 无效的风格名称
            RuntimeError: 模型未加载或生成失败
        """
        style_info = self.get_style_info(style)
        if style_info is None:
            raise ValueError(
                f"无效的风格: {style!r}。可用风格: {', '.join(self.STYLES)}"
            )

        # 解析输出目录
        if output_dir is None:
            from apps.recognition_system.config import PROJECT_ROOT

            output_dir = str(PROJECT_ROOT / self.config.output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        # 加载参考图像
        input_image = self._load_image(reference_image)

        # 确保管线已加载
        self._ensure_pipeline()

        # 生成参数
        gen_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        n_steps = num_inference_steps or self.config.num_inference_steps
        g_scale = guidance_scale or self.config.guidance_scale
        merge_step = start_merge_step or self.config.start_merge_step

        prompt = style_info["prompt"]
        negative_prompt = style_info["negative_prompt"]

        logger.info(
            "开始生成肖像: style=%s, seed=%d, steps=%d, guidance=%.1f, merge_step=%d",
            style, gen_seed, n_steps, g_scale, merge_step,
        )

        t_start = time.perf_counter()

        try:
            import torch

            # 使用 CPU offload 时 generator 应在 CPU 上创建
            gen_device = "cpu" if self.config.enable_cpu_offload else self.config.device
            generator = torch.Generator(device=gen_device).manual_seed(gen_seed)

            # 生成
            result_images = self._pipe(
                prompt=prompt,
                input_id_images=[input_image],
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                num_inference_steps=n_steps,
                start_merge_step=merge_step,
                guidance_scale=g_scale,
                generator=generator,
                height=self.config.height,
                width=self.config.width,
            ).images

            generated_image = result_images[0]

        except Exception as exc:
            logger.error("肖像生成失败: %s", exc)
            raise RuntimeError(f"PhotoMaker 生成失败: {exc}") from exc

        elapsed = time.perf_counter() - t_start

        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"portrait_{style}_{timestamp}_seed{gen_seed}.png"
        output_path = os.path.join(output_dir, filename)
        generated_image.save(output_path)

        logger.info("肖像已保存: %s (耗时 %.1fs)", output_path, elapsed)

        return PortraitResult(
            style=style,
            style_label=style_info["label"],
            reference_image_path=(
                reference_image if isinstance(reference_image, str) else "<memory>"
            ),
            output_path=output_path,
            prompt_used=prompt,
            seed=gen_seed,
            generation_time_seconds=round(elapsed, 2),
            width=generated_image.width,
            height=generated_image.height,
        )

    def unload(self):
        """卸载管线以释放 GPU 显存。"""
        if self._pipe is not None:
            logger.info("正在卸载 PhotoMaker 管线...")
            del self._pipe
            self._pipe = None
            self._loaded = False
            _cleanup_gpu()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _ensure_pipeline(self):
        """懒加载管线（首次调用时加载）。"""
        if self._loaded and self._pipe is not None:
            return
        self._load_pipeline()

    def _load_pipeline(self):
        """加载 PhotoMaker + SDXL 管线（针对 8GB 显存和低 RAM 优化）。"""
        import torch
        import gc
        from diffusers import EulerDiscreteScheduler

        # 配置 HuggingFace 缓存目录和镜像
        from apps.recognition_system.config import PROJECT_ROOT

        hf_cache = str((PROJECT_ROOT / self.config.hf_cache_dir).resolve()) if self.config.hf_cache_dir else str((PROJECT_ROOT / "weights" / "hf_cache").resolve())
        # 重要：只设 HF_HUB_CACHE，不设 HF_HOME
        # HF_HOME 会让它去 hf_cache/hub/ 找，但我们的文件直接在 hf_cache/ 下
        os.environ["HF_HUB_CACHE"] = hf_cache
        logger.info("  HF 缓存: %s", hf_cache)
        if self.config.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.config.hf_endpoint
            logger.info("  HF 镜像: %s", self.config.hf_endpoint)

        # 确定数据类型和设备
        is_cuda = self.config.device == "cuda" and torch.cuda.is_available()
        if is_cuda:
            # 使用 float16（比 bfloat16 更兼容，尤其配合 CPU offload）
            dtype = torch.float16
        else:
            dtype = torch.float32

        # ── 内存预检 ──
        self._check_memory(is_cuda)

        logger.info("正在加载 PhotoMaker 管线...")
        logger.info("  基础模型: %s", self.config.base_model_id)
        logger.info("  设备: %s", self.config.device)
        logger.info("  数据类型: %s", dtype)

        # 加载前清理内存
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

        try:
            # 导入 PhotoMaker 管线
            from photomaker import PhotoMakerStableDiffusionXLPipeline

            gpu_full_load = is_cuda and torch.cuda.get_device_properties(0).total_memory / 1e9 >= 7
            # ★ 加载策略：显存 >= 7 GB → 直接上 GPU；否则 CPU offload
            # SDXL fp16 ≈ 6.5 GB，8GB 显卡足够直接加载
            if gpu_full_load:
                logger.info("  大显存模式：直接加载到 GPU...")
                pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                    self.config.base_model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16" if dtype != torch.float32 else None,
                    local_files_only=False,
                    device=self.config.device,
                    cache_dir=hf_cache,
                )
            else:
                logger.info("  低显存模式：加载到 CPU + sequential offload...")
                pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                    self.config.base_model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16" if dtype != torch.float32 else None,
                    local_files_only=False,
                    cache_dir=hf_cache,
                    # 不传 device，默认加载到 CPU
                )

            # 加载 PhotoMaker 适配器 (CPU 上操作)
            photomaker_dir = self.config.photomaker_model_dir
            if not photomaker_dir:
                from huggingface_hub import snapshot_download

                photomaker_dir = snapshot_download("TencentARC/PhotoMaker", cache_dir=hf_cache)

            pipe.load_photomaker_adapter(
                photomaker_dir,
                subfolder="",
                weight_name="photomaker-v1.bin",
                trigger_word=self.config.trigger_word,
                pm_version="v1",
            )

            # 设置调度器
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config
            )

            # 融合 LoRA 权重 (CPU 上操作)
            pipe.fuse_lora()

        except ImportError as exc:
            raise ImportError(
                "PhotoMaker 库未安装。请运行:\n"
                "  pip install git+https://github.com/TencentARC/PhotoMaker.git\n"
                f"原始错误: {exc}"
            ) from exc
        except MemoryError as exc:
            raise RuntimeError(
                "内存不足，无法加载 SDXL 模型（需要约 8 GB 可用 RAM）。\n"
                "请尝试以下操作：\n"
                "  1. 关闭其他占用内存的程序（浏览器、IDE 等）\n"
                "  2. 扩大 Windows 页面文件（虚拟内存）到至少 16 GB\n"
                "     → 设置 → 系统 → 关于 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存 → 更改\n"
                f"原始错误: {exc}"
            ) from exc
        except OSError as exc:
            # Windows 页面文件太小 (OSError 1455)
            if "1455" in str(exc) or "页面文件" in str(exc):
                raise RuntimeError(
                    "Windows 页面文件太小，无法内存映射模型文件。\n"
                    "请扩大虚拟内存：\n"
                    "  设置 → 系统 → 关于 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存 → 更改\n"
                    "  建议：自定义大小，初始 16384 MB，最大 32768 MB\n"
                    "  然后重启电脑。\n"
                    f"原始错误: {exc}"
                ) from exc
            raise

        # ── 显存优化 ──
        if is_cuda and not gpu_full_load:
            # CPU 加载模式 → 启用 sequential offload
            logger.info("  显存不足 7GB，启用 sequential CPU offload...")
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
        elif not is_cuda:
            logger.info("  使用 CPU 推理 (速度较慢)")

        self._pipe = pipe
        self._loaded = True
        logger.info("PhotoMaker 管线加载完成 ✓")

    @staticmethod
    def _check_memory(is_cuda: bool):
        """检查系统内存是否足够加载模型。"""
        import psutil

        avail_ram_gb = psutil.virtual_memory().available / 1e9
        min_ram_gb = 8.0  # SDXL fp16 模型约需 6.5 GB + 缓冲

        if avail_ram_gb < min_ram_gb:
            logger.warning(
                "⚠ 可用 RAM: %.1f GB (建议 ≥ %.0f GB)。加载大型模型可能导致内存不足。",
                avail_ram_gb, min_ram_gb,
            )
            logger.warning(
                "如果加载失败，请关闭其他程序释放内存，或扩大 Windows 页面文件。"
            )

        if is_cuda:
            import torch
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(
                "  系统 RAM 可用: %.1f GB | GPU 显存: %.1f GB",
                avail_ram_gb, total_vram_gb,
            )

    @staticmethod
    def _load_image(source: str | Image.Image | np.ndarray) -> Image.Image:
        """统一加载图像（路径 / PIL Image / numpy 数组）。"""
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source[..., ::-1]).convert("RGB")  # BGR → RGB
        if isinstance(source, (str, os.PathLike)):
            path = str(source)
            if not os.path.exists(path):
                raise FileNotFoundError(f"参考图像不存在: {path}")
            return Image.open(path).convert("RGB")
        raise TypeError(f"不支持的图像类型: {type(source)}")


# ---------------------------------------------------------------------------
# GPU 显存清理
# ---------------------------------------------------------------------------

def _cleanup_gpu():
    """清理 GPU 显存。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
