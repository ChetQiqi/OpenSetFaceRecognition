from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from apps.recognition_system.api.auth_routes import router as auth_router
from apps.recognition_system.api.schemas import (
    AddIdentityResponse,
    BenchmarkSummaryResponse,
    CameraActionResponse,
    CameraStartRequest,
    CameraStatusResponse,
    DeleteIdentityResponse,
    EvalJobListResponse,
    EvalJobStatus,
    EvalJobSubmitResponse,
    HealthResponse,
    IdentityListResponse,
    IJBEvalRequest,
    ImageRecognitionResponse,
    LFWEvalRequest,
    ModelEvalResponse,
    ModelRuntimeInfoResponse,
    ModelSwitchRequest,
    ModelActivateResponse,
    ModelDeleteResponse,
    ModelListResponse,
    ModelWeightsResponse,
    PersonDetailResponse,
    PortraitGenerateRequest,
    PortraitGenerateResponse,
    PortraitPersonImageItem,
    PortraitStyleItem,
    PortraitStyleListResponse,
    RenameIdentityRequest,
    RenameIdentityResponse,
    StatsResponse,
    ThresholdSweepRequest,
    UpdateIdentityRequest,
    UpdateIdentityResponse,
    VideoRecognitionResponse,
)
from apps.recognition_system.auth.db import init_auth_db
from apps.recognition_system.auth.dependencies import require_roles
from apps.recognition_system.auth.models import User, UserRole
from apps.recognition_system.config import PROJECT_ROOT, get_config
from apps.recognition_system.models import ModelRegistry, ModelService
from apps.recognition_system.repositories import IdentityRepository, ModelRepository
from apps.recognition_system.services import (
    DeveloperService,
    EvalService,
    IdentityService,
    InferenceService,
    ModelManagementService,
    PortraitService,
    VideoService,
)

from apps.recognition_system.services.portrait_service import PortraitServiceConfig


def _has_cuda() -> bool:
    """检测是否有可用的 CUDA 设备。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


config = get_config()
repository = IdentityRepository(config.db_path)
model_repository = ModelRepository(config.db_path)
model_service = ModelService(config, repository)
model_registry = ModelRegistry()
model_registry.configure(model_service=model_service, model_repository=model_repository)
identity_service = IdentityService(repository, model_service)
inference_service = InferenceService(config, model_service)
video_service = VideoService(inference_service)
developer_service = DeveloperService(repository, model_service)
eval_service = EvalService(model_service, repository)
model_management_service = ModelManagementService(
    model_repository=model_repository,
    model_registry=model_registry,
    project_root=PROJECT_ROOT,
)
portrait_config = PortraitServiceConfig(
    device=config.device if config.device != "auto" else ("cuda" if _has_cuda() else "cpu"),
    enable_cpu_offload=True,  # 8GB 显卡建议开启 CPU offload
    hf_endpoint="https://hf-mirror.com",  # 国内 HuggingFace 镜像加速下载
    hf_cache_dir=str(PROJECT_ROOT / "weights" / "hf_cache"),  # 模型存项目内，不占 C 盘
)
portrait_service = PortraitService(portrait_config)
app = FastAPI(title="Recognition System API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)


@app.on_event("startup")
def startup_event():
    init_auth_db()
    model_registry.initialize_from_db()


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": model_service.is_loaded()}


@app.post("/model/load", response_model=HealthResponse)
def load_model(_: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value))):
    try:
        active = model_management_service.get_active_model()
        if active is not None:
            model_registry.activate(active, persist=False)
        else:
            model_service.load()
        return {"status": "ok", "model_loaded": model_service.is_loaded()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats", response_model=StatsResponse)
def stats(_: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value))):
    return identity_service.stats()


@app.get("/identity", response_model=IdentityListResponse)
def list_identities(
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value))
):
    return {"persons": identity_service.list_identities()}


@app.post("/identity/add", response_model=AddIdentityResponse)
async def add_identity(
    person_id: str = Form(...),
    files: List[UploadFile] = File(...),
    gender: Optional[str] = Form("unspecified"),
    birth_date: Optional[str] = Form(""),
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value)),
):
    try:
        images = [(file.filename or "uploaded_image", await file.read()) for file in files]
        return identity_service.add_identity(person_id, images,
                                             gender=gender or "unspecified",
                                             birth_date=birth_date or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/identity/{person_name}", response_model=DeleteIdentityResponse)
def delete_identity(person_name: str, _: User = Depends(require_roles(UserRole.admin.value))):
    return identity_service.delete_identity(person_name)


@app.get("/identity/{person_name}/detail", response_model=PersonDetailResponse)
def get_identity_detail(
    person_name: str,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    detail = identity_service.get_identity(person_name)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_name}' not found")
    return detail


@app.put("/identity/{person_name}", response_model=UpdateIdentityResponse)
def update_identity_metadata(
    person_name: str,
    payload: UpdateIdentityRequest,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    try:
        return identity_service.update_identity(
            old_name=person_name,
            new_name=payload.new_name or "",
            gender=payload.gender or "",
            birth_date=payload.birth_date or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/identity/rename", response_model=RenameIdentityResponse)
def rename_identity(payload: RenameIdentityRequest, _: User = Depends(require_roles(UserRole.admin.value))):
    try:
        return identity_service.rename_identity(payload.old_name, payload.new_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/recognize", response_model=ImageRecognitionResponse)
async def recognize_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.45),
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value)),
):
    try:
        return inference_service.recognize_image(await file.read(), threshold=threshold, draw=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/recognize/video", response_model=VideoRecognitionResponse)
async def recognize_video(
    file: UploadFile = File(...),
    skip_frames: int = Form(5),
    threshold: float = Form(0.45),
    stable_frames: int = Form(3),
    mode: str = Form("视频中找人"),
    verify_target: Optional[str] = Form(None),
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value)),
):
    try:
        return video_service.recognize_video(
            video_bytes=await file.read(),
            filename=file.filename or "uploaded.mp4",
            skip_frames=skip_frames,
            threshold=threshold,
            stable_frames=stable_frames,
            mode=mode,
            verify_target=verify_target,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/camera/start", response_model=CameraActionResponse)
def start_camera(
    payload: CameraStartRequest,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value)),
):
    try:
        return video_service.start_camera(**payload.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/camera/stop", response_model=CameraActionResponse)
def stop_camera(_: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value))):
    return video_service.stop_camera()


@app.post("/camera/clear", response_model=CameraActionResponse)
def clear_camera(_: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value))):
    return video_service.clear_camera()


@app.get("/camera/status", response_model=CameraStatusResponse)
def camera_status(
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value))
):
    return video_service.camera_status()


@app.post("/model/switch", response_model=ModelRuntimeInfoResponse)
def switch_model(
    payload: ModelSwitchRequest,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    try:
        model_service.switch_model(
            model_name=payload.model_name,
            weights_path=payload.weights_path,
            img_size=payload.img_size,
            device=payload.device,
        )
        return model_service.get_runtime_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/model/weights", response_model=ModelWeightsResponse)
def list_model_weights(_: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value))):
    weights_dir = Path(config.weights_path).resolve().parent
    if not weights_dir.exists():
        return {"files": []}

    exts = {".pt", ".pth", ".ckpt", ".bin", ".onnx"}
    files = []
    for file_path in sorted(weights_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in exts:
            files.append({"name": file_path.name, "path": str(file_path.resolve())})
    managed_dir = (weights_dir / "managed").resolve()
    if managed_dir.exists():
        for file_path in sorted(managed_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in exts:
                files.append({"name": f"managed/{file_path.name}", "path": str(file_path.resolve())})
    return {"files": files}


@app.get("/model/runtime", response_model=ModelRuntimeInfoResponse)
def model_runtime(
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value))
):
    return model_service.get_runtime_info()


@app.get("/developer/benchmark", response_model=BenchmarkSummaryResponse)
def developer_benchmark(_: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value))):
    return developer_service.benchmark_summary()


@app.post("/developer/model-eval", response_model=ModelEvalResponse)
def developer_model_eval(
    topn: int = 10,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    return developer_service.model_eval(topn=topn)


@app.post("/models/upload", response_model=ModelActivateResponse)
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    backbone: str = Form("iresnet50"),
    embedding_size: int = Form(512),
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    try:
        model = model_management_service.upload_model(
            file_name=file.filename or "uploaded_model.pt",
            content=await file.read(),
            name=name,
            backbone=backbone,
            embedding_size=embedding_size,
        )
        return model
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/models/list", response_model=ModelListResponse)
def list_models(_: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value))):
    return {"models": model_management_service.list_models()}


@app.post("/models/{model_id}/activate", response_model=ModelActivateResponse)
def activate_model(model_id: int, _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value))):
    try:
        return model_management_service.activate_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/models/{model_id}", response_model=ModelDeleteResponse)
def delete_model(model_id: int, _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value))):
    try:
        return model_management_service.delete_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 模型评估端点
# ---------------------------------------------------------------------------

@app.post("/developer/eval/lfw", response_model=EvalJobSubmitResponse)
def submit_lfw_eval(
    payload: LFWEvalRequest,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    """提交 LFW 风格验证任务（后台异步执行）。"""
    job_id = eval_service.submit_lfw_eval(
        weights_path=payload.weights_path,
        backbone=payload.backbone,
        data_root=payload.data_root,
        datasets=payload.datasets,
        batch_size=payload.batch_size,
    )
    return {"job_id": job_id, "message": f"LFW 评估任务已提交，job_id={job_id}"}


@app.post("/developer/eval/ijb", response_model=EvalJobSubmitResponse)
def submit_ijb_eval(
    payload: IJBEvalRequest,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    """提交 IJB-B/C 评估任务（后台异步执行）。"""
    job_id = eval_service.submit_ijb_eval(
        weights_path=payload.weights_path,
        backbone=payload.backbone,
        image_path=payload.image_path,
        target=payload.target,
        batch_size=payload.batch_size,
        use_norm_score=payload.use_norm_score,
        use_detector_score=payload.use_detector_score,
        use_flip_test=payload.use_flip_test,
        result_dir=payload.result_dir,
    )
    return {"job_id": job_id, "message": f"IJB {payload.target} 评估任务已提交，job_id={job_id}"}


@app.post("/developer/eval/threshold-sweep", response_model=EvalJobSubmitResponse)
def submit_threshold_sweep(
    payload: ThresholdSweepRequest,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    """提交阈值扫描评估任务（后台异步执行）。"""
    job_id = eval_service.submit_threshold_sweep(
        weights_path=payload.weights_path,
        backbone=payload.backbone,
        image_dir=payload.image_dir,
        db_path=payload.db_path,
        thresholds=payload.thresholds,
        device=payload.device,
    )
    return {"job_id": job_id, "message": f"阈值扫描任务已提交，job_id={job_id}"}


@app.get("/developer/eval/status/{job_id}", response_model=EvalJobStatus)
def eval_job_status(
    job_id: str,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    """查询评估任务状态（用于前端轮询）。"""
    job = eval_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"评估任务不存在: {job_id}")
    return job.to_dict()


@app.get("/developer/eval/jobs", response_model=EvalJobListResponse)
def list_eval_jobs(
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    """列出所有评估任务历史。"""
    return {"jobs": [j.to_dict() for j in eval_service.list_jobs()]}


@app.post("/developer/eval/cancel/{job_id}")
def cancel_eval_job(
    job_id: str,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    """取消正在运行的评估任务。"""
    ok = eval_service.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="任务不存在或已结束，无法取消")
    return {"message": f"任务 {job_id} 已停止"}


# ── AI 肖像生成端点 ──

@app.get("/portrait/styles", response_model=PortraitStyleListResponse)
def get_portrait_styles(
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value)),
):
    """获取可用的肖像生成风格列表。"""
    return {"styles": PortraitService.get_styles()}


@app.get("/portrait/persons")
def get_portrait_persons(
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value)),
):
    """获取已注册人员及其原始照片信息。"""
    persons_data = identity_service.list_identities()
    result = []
    for person in persons_data:
        name = person["name"]
        image_paths = repository.list_person_image_paths(name)
        result.append({
            "name": name,
            "image_paths": image_paths,
        })
    return {"persons": result}


@app.get("/portrait/person/{person_name}/image")
def get_portrait_person_image(
    person_name: str,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value)),
):
    """获取指定人员的原始照片内容 (base64)。"""
    import base64 as _b64

    image_path = repository.get_person_latest_image_path(person_name)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"人员 '{person_name}' 的照片不存在")

    path = Path(image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"照片文件不存在: {image_path}")

    content = path.read_bytes()
    ext = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp"}
    mime = mime_map.get(ext, "image/jpeg")
    return {"name": person_name, "image_path": str(path), "image_base64": _b64.b64encode(content).decode(), "mime": mime}


@app.post("/portrait/generate", response_model=PortraitGenerateResponse)
def generate_portrait(
    payload: PortraitGenerateRequest,
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    """生成 AI 肖像。"""
    import base64 as _b64

    # 验证图像路径
    image_path = payload.image_path
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=400, detail=f"参考图像不存在: {image_path}")

    try:
        result = portrait_service.generate(
            reference_image=image_path,
            style=payload.style,
            seed=payload.seed,
            num_inference_steps=payload.num_inference_steps,
            guidance_scale=payload.guidance_scale,
            start_merge_step=payload.start_merge_step,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"肖像生成失败: {exc}") from exc

    # 读取生成结果并 base64 编码
    output_content = Path(result.output_path).read_bytes()
    return {
        "style": result.style,
        "style_label": result.style_label,
        "reference_image_path": result.reference_image_path,
        "result_image_base64": _b64.b64encode(output_content).decode(),
        "output_path": result.output_path,
        "prompt_used": result.prompt_used,
        "seed": result.seed,
        "generation_time_seconds": result.generation_time_seconds,
        "width": result.width,
        "height": result.height,
        "message": f"生成完成，耗时 {result.generation_time_seconds:.1f} 秒",
    }


@app.post("/portrait/unload")
def unload_portrait_model(
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value)),
):
    """卸载肖像生成模型以释放 GPU 显存。"""
    portrait_service.unload()
    return {"message": "PhotoMaker 管线已卸载"}


@app.get("/portrait/status")
def portrait_status(
    _: User = Depends(require_roles(UserRole.admin.value, UserRole.developer.value, UserRole.viewer.value)),
):
    """查询肖像生成服务状态。"""
    return {
        "loaded": portrait_service.is_loaded(),
        "device": portrait_service.config.device,
        "available_styles": PortraitService.get_styles(),
    }


# ── Serve React SPA (frontend) ──
REACT_BUILD_DIR = PROJECT_ROOT / "frontend" / "dist"
if REACT_BUILD_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(REACT_BUILD_DIR / "assets")), name="react_assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(REACT_BUILD_DIR / "index.html")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        file_path = REACT_BUILD_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(REACT_BUILD_DIR / "index.html")
