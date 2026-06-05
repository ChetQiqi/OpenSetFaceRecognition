from typing import Dict, List, Optional

from pydantic import BaseModel


class IdentityItem(BaseModel):
    name: str
    embedding_count: int
    gender: str = "unspecified"
    birth_date: str = ""


class IdentityListResponse(BaseModel):
    persons: List[IdentityItem]


class StatsResponse(BaseModel):
    person_count: int
    embedding_count: int


class AddIdentityResponse(BaseModel):
    person_name: str
    success_count: int
    fail_count: int
    success_files: List[str]
    failed_files: List[Dict[str, str]]


class DeleteIdentityResponse(BaseModel):
    deleted: bool
    person_name: str


class RenameIdentityRequest(BaseModel):
    old_name: str
    new_name: str


class RenameIdentityResponse(BaseModel):
    renamed: bool
    old_name: str
    new_name: str


class UpdateIdentityRequest(BaseModel):
    gender: Optional[str] = None
    birth_date: Optional[str] = None
    new_name: Optional[str] = None


class UpdateIdentityResponse(BaseModel):
    updated: bool
    old_name: str
    new_name: str


class PersonDetailResponse(BaseModel):
    name: str
    embedding_count: int
    gender: str
    birth_date: str


class RecognitionResult(BaseModel):
    box: List[int]
    name: str
    display_name: str
    score: float
    raw_score: float
    accepted: bool
    label: str
    support: int = 0


class PersonCard(BaseModel):
    name: str
    gender: str = "unspecified"
    birth_date: str = ""
    embedding_count: int = 0
    gallery_face_base64: Optional[str] = None
    score: float = 0.0
    accepted: bool = False


class ImageRecognitionResponse(BaseModel):
    results: List[RecognitionResult]
    annotated_image_base64: Optional[str] = None
    matched_pairs: List[Dict[str, str]] = []
    person_cards: List[PersonCard] = []


class VideoRecognitionResponse(BaseModel):
    video_base64: str
    output_filename: str
    total_frames: int
    processed_frames: int
    written_frames: int
    fps: int
    width: int
    height: int
    person_counts: Dict[str, int]
    stranger_count: int
    verify_result: Optional[Dict[str, object]] = None
    person_cards: List[PersonCard] = []


class CameraStartRequest(BaseModel):
    camera_id: int = 0
    skip_frames: int = 3
    threshold: float = 0.45
    stable_frames: int = 3
    mode: str = "视频中找人"
    verify_target: Optional[str] = None


class CameraActionResponse(BaseModel):
    running: bool
    message: str


class CameraStatusResponse(BaseModel):
    running: bool
    results: List[RecognitionResult]
    stats: Dict[str, int]
    total_frames: int
    processed_frames: int
    fps: float
    verify_status: Optional[str] = None
    verify_count: int = 0
    person_cards: List[PersonCard] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelSwitchRequest(BaseModel):
    model_name: str
    weights_path: Optional[str] = None
    img_size: Optional[int] = None
    device: Optional[str] = None


class ModelRuntimeInfoResponse(BaseModel):
    model_name: str
    weights_path: str
    img_size: int
    device: str
    loaded: bool
    gallery_size: int


class ModelWeightItem(BaseModel):
    name: str
    path: str


class ModelWeightsResponse(BaseModel):
    files: List[ModelWeightItem]


class ModelItem(BaseModel):
    id: int
    name: str
    path: str
    backbone: str
    embedding_size: int
    framework: str
    created_at: str
    is_active: bool


class ModelListResponse(BaseModel):
    models: List[ModelItem]


class ModelActivateResponse(BaseModel):
    id: int
    name: str
    path: str
    backbone: str
    embedding_size: int
    framework: str
    created_at: str
    is_active: bool


class ModelDeleteResponse(BaseModel):
    deleted: bool
    id: int
    reason: Optional[str] = None


class BenchmarkPersonItem(BaseModel):
    name: str
    embedding_count: int


class BenchmarkSummaryResponse(BaseModel):
    database_path: str
    person_count: int
    embedding_count: int
    top_persons: List[BenchmarkPersonItem]


class ModelEvalResponse(BaseModel):
    runtime: ModelRuntimeInfoResponse
    gallery_size: int
    class_balance_top: List[BenchmarkPersonItem]
    class_balance_bottom: List[BenchmarkPersonItem]
    recommendation: str


# ---------------------------------------------------------------------------
# 模型评估任务（EvalService）
# ---------------------------------------------------------------------------

class LFWEvalRequest(BaseModel):
    """LFW 风格验证请求。需要 bcolz 格式的标准人脸验证数据集目录。"""
    weights_path: str
    backbone: str = "iresnet50"
    data_root: str
    datasets: List[str] = ["lfw", "calfw", "cplfw", "agedb_30", "cfp_fp", "vgg2_fp"]
    batch_size: int = 512


class IJBEvalRequest(BaseModel):
    """IJB-B / IJB-C 评估请求。"""
    weights_path: str
    backbone: str = "iresnet50"
    image_path: str          # 包含 loose_crop/ 和 meta/ 的目录
    target: str = "IJBB"     # "IJBB" 或 "IJBC"
    batch_size: int = 512
    use_norm_score: bool = True
    use_detector_score: bool = True
    use_flip_test: bool = True
    result_dir: str = ""


class ThresholdSweepRequest(BaseModel):
    """阈值扫描评估请求。使用按人名分文件夹的自定义测试图片目录。"""
    weights_path: str
    backbone: str = "iresnet50"
    image_dir: str
    db_path: str
    thresholds: str = "0.30,0.35,0.40,0.45,0.50,0.55,0.60"
    device: str = "auto"


class EvalJobStatus(BaseModel):
    """评估任务状态（轮询用）。"""
    job_id: str
    job_type: str
    status: str
    model_name: str
    weights_path: str
    params: Dict[str, object]
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0
    progress_msg: str = ""
    progress_data: Dict[str, object] = {}
    result: Optional[Dict[str, object]] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class EvalJobSubmitResponse(BaseModel):
    job_id: str
    message: str


class EvalJobListResponse(BaseModel):
    jobs: List[EvalJobStatus]


# ---------------------------------------------------------------------------
# AI 肖像生成 (Portrait Generation)
# ---------------------------------------------------------------------------

class PortraitStyleItem(BaseModel):
    """单个风格信息。"""
    key: str
    label: str


class PortraitStyleListResponse(BaseModel):
    """可用风格列表。"""
    styles: List[PortraitStyleItem]


class PortraitPersonImageItem(BaseModel):
    """已注册人员的图像信息。"""
    name: str
    image_paths: List[str]
    latest_image_base64: Optional[str] = None


class PortraitGenerateRequest(BaseModel):
    """肖像生成请求。"""
    person_name: str          # 人员名称
    image_path: str           # 参考图像路径（服务器端路径）
    style: str                # 风格 key
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    start_merge_step: Optional[int] = None


class PortraitGenerateResponse(BaseModel):
    """肖像生成响应。"""
    style: str
    style_label: str
    reference_image_path: str
    result_image_base64: str
    output_path: str
    prompt_used: str
    seed: int
    generation_time_seconds: float
    width: int
    height: int
    message: str = ""
