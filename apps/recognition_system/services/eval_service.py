"""
模型评估服务 —— 后台异步任务，调用 tools/ 下的 runner 子脚本。

支持两类评估：
  1. LFW-style verification（lfw/calfw/cplfw/agedb_30/cfp_fp/vgg2_fp）
     由 tools/eval_lfw_runner.py 执行，逻辑与 FaceRec_plus/verification.py 完全一致。

  2. IJB evaluation（IJBB / IJBC）
     由 tools/eval_ijb_runner.py 执行，逻辑与 FaceRec_plus/evaluate_IJB.py 完全一致。

两种 runner 均使用项目自身的 backbones.iresnet（与 FaceRec_plus 一致），
无需依赖外部 FaceRec_plus 目录。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# runner 脚本所在目录（项目根 / tools/）
_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tools"


# ---------------------------------------------------------------------------
# Job 数据结构
# ---------------------------------------------------------------------------

@dataclass
class EvalJob:
    job_id: str
    job_type: str           # "lfw_verification" | "ijb_evaluation" | "threshold_sweep"
    status: str             # "pending" | "running" | "done" | "error"
    model_name: str
    weights_path: str
    params: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0
    progress_msg: str = "等待中"
    progress_data: Dict[str, Any] = field(default_factory=dict)   # 细粒度进度（数据集/阶段）
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - (self.started_at or end)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "model_name": self.model_name,
            "weights_path": self.weights_path,
            "params": self.params,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "progress_msg": self.progress_msg,
            "progress_data": self.progress_data,
            "result": self.result,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed(), 1),
        }


# ---------------------------------------------------------------------------
# EvalService
# ---------------------------------------------------------------------------

class EvalService:
    MAX_JOBS = 50

    def __init__(self, model_service, repository):
        self.model_service = model_service
        self.repository = repository
        self._jobs: Dict[str, EvalJob] = {}
        self._procs: Dict[str, Any] = {}   # job_id -> subprocess.Popen
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公共提交接口
    # ------------------------------------------------------------------

    def submit_lfw_eval(
        self,
        weights_path: str,
        backbone: str,
        data_root: str,
        datasets: Optional[List[str]] = None,
        batch_size: int = 512,
    ) -> str:
        if datasets is None:
            datasets = ["lfw", "calfw", "cplfw", "agedb_30", "cfp_fp", "vgg2_fp"]
        job = self._new_job(
            "lfw_verification",
            model_name=backbone,
            weights_path=weights_path,
            params={
                "data_root": data_root,
                "datasets": datasets,
                "batch_size": batch_size,
            },
        )
        threading.Thread(target=self._run_runner, args=(job, "lfw"), daemon=True).start()
        return job.job_id

    def submit_ijb_eval(
        self,
        weights_path: str,
        backbone: str,
        image_path: str,
        target: str = "IJBB",
        batch_size: int = 512,
        use_norm_score: bool = True,
        use_detector_score: bool = True,
        use_flip_test: bool = True,
        result_dir: str = "",
    ) -> str:
        job = self._new_job(
            "ijb_evaluation",
            model_name=backbone,
            weights_path=weights_path,
            params={
                "image_path": image_path,
                "target": target,
                "batch_size": batch_size,
                "use_norm_score": use_norm_score,
                "use_detector_score": use_detector_score,
                "use_flip_test": use_flip_test,
                "result_dir": result_dir or str(Path(image_path).parent),
            },
        )
        threading.Thread(target=self._run_runner, args=(job, "ijb"), daemon=True).start()
        return job.job_id

    def submit_threshold_sweep(
        self,
        weights_path: str,
        backbone: str,
        image_dir: str,
        db_path: str,
        thresholds: str = "0.30,0.35,0.40,0.45,0.50,0.55,0.60",
        device: str = "auto",
    ) -> str:
        job = self._new_job(
            "threshold_sweep",
            model_name=backbone,
            weights_path=weights_path,
            params={
                "image_dir": image_dir,
                "db_path": db_path,
                "thresholds": thresholds,
                "device": device,
            },
        )
        threading.Thread(target=self._run_threshold_sweep, args=(job,), daemon=True).start()
        return job.job_id

    def get_job(self, job_id: str) -> Optional[EvalJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[EvalJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        """取消正在运行的任务，返回是否成功取消。"""
        job = self._jobs.get(job_id)
        if job is None or job.status not in ("running", "pending"):
            return False
        proc = self._procs.get(job_id)
        if proc is not None:
            try:
                import signal, os
                # Windows 用 terminate()，Linux 用 SIGTERM
                proc.terminate()
            except Exception:
                pass
        job.status = "error"
        job.error = "用户手动停止"
        job.progress_msg = "已停止"
        job.finished_at = time.time()
        return True

    # ------------------------------------------------------------------
    # 内部 —— 任务管理
    # ------------------------------------------------------------------

    def _new_job(self, job_type: str, model_name: str, weights_path: str, params: Dict) -> EvalJob:
        job_id = str(uuid.uuid4())[:8]
        job = EvalJob(
            job_id=job_id,
            job_type=job_type,
            status="pending",
            model_name=model_name,
            weights_path=weights_path,
            params=params,
        )
        with self._lock:
            self._jobs[job_id] = job
            done_jobs = [j for j in self._jobs.values() if j.status in ("done", "error")]
            if len(self._jobs) > self.MAX_JOBS and done_jobs:
                oldest = min(done_jobs, key=lambda j: j.created_at)
                del self._jobs[oldest.job_id]
        return job

    # ------------------------------------------------------------------
    # LFW / IJB runner（子进程）
    # ------------------------------------------------------------------

    def _run_runner(self, job: EvalJob, runner_type: str) -> None:
        """通用 runner 执行器：写 config.json → 调子进程 → 读 result.json"""
        job.status = "running"
        job.started_at = time.time()

        runner_map = {
            "lfw": "eval_lfw_runner.py",
            "ijb": "eval_ijb_runner.py",
        }
        runner_script = _TOOLS_DIR / runner_map[runner_type]
        if not runner_script.exists():
            job.status = "error"
            job.error = f"Runner 脚本不存在: {runner_script}"
            job.finished_at = time.time()
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            result_path = Path(tmpdir) / "result.json"

            # 构造 config，project_root 供 runner 将项目加入 sys.path
            project_root = str(Path(__file__).resolve().parents[3])
            cfg = {
                **job.params,
                "project_root": project_root,
                "weights_path": job.weights_path,
                "backbone": job.model_name,
                "output_json": str(result_path),
            }

            config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

            job.progress_msg = f"正在运行 {runner_type.upper()} 评估（这可能需要几分钟）..."
            job.progress = 5.0

            try:
                proc = subprocess.Popen(
                    [sys.executable, str(runner_script), str(config_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self._procs[job.job_id] = proc

                stdout_lines: List[str] = []
                for line in proc.stdout:  # type: ignore[union-attr]
                    line = line.rstrip()
                    stdout_lines.append(line)
                    if line.startswith("__PROGRESS__"):
                        # 解析结构化进度行
                        try:
                            payload = json.loads(line[len("__PROGRESS__"):].strip())
                            job.progress = float(payload.get("percent", job.progress))
                            # LFW 格式
                            if "done" in payload and "total" in payload:
                                done = payload["done"]
                                total = payload["total"]
                                current = payload.get("current", "")
                                partial = payload.get("partial", {})
                                job.progress_msg = (
                                    f"正在评估 {current} ({done}/{total})" if current
                                    else f"评估完成 ({done}/{total})"
                                )
                                job.progress_data = {
                                    "type": "lfw",
                                    "done": done,
                                    "total": total,
                                    "current": current,
                                    "datasets": partial,
                                }
                            # IJB 格式
                            elif "phase" in payload:
                                phase = payload["phase"]
                                msg = payload.get("msg", phase)
                                job.progress_msg = msg
                                if phase not in job.progress_data.get("phases_done", []):
                                    phases_done = job.progress_data.get("phases_done", [])
                                    if phase != job.progress_data.get("current_phase"):
                                        phases_done = list(phases_done) + \
                                            ([job.progress_data.get("current_phase")]
                                             if job.progress_data.get("current_phase") else [])
                                    job.progress_data = {
                                        "type": "ijb",
                                        "phases_done": phases_done,
                                        "current_phase": phase,
                                        "current_msg": msg,
                                    }
                        except Exception:
                            pass
                    elif line:
                        # 普通日志行，只更新 msg，不覆盖 progress_data
                        job.progress_msg = line[:120]

                proc.wait(timeout=3600)

                if proc.returncode != 0:
                    job.status = "error"
                    job.error = "\n".join(stdout_lines[-40:])
                    job.progress_msg = "评估失败"
                    return

                if not result_path.exists():
                    job.status = "error"
                    job.error = "Runner 未生成结果文件\n\n" + "\n".join(stdout_lines[-20:])
                    return

                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["stdout_tail"] = stdout_lines[-20:]
                job.result = result
                job.status = "done"
                job.progress = 100.0
                job.progress_msg = "评估完成"

            except subprocess.TimeoutExpired:
                proc.kill()
                job.status = "error"
                job.error = "评估超时（>60分钟）"
            except Exception:
                job.status = "error"
                job.error = traceback.format_exc()
                job.progress_msg = "评估出错"
            finally:
                self._procs.pop(job.job_id, None)
                job.finished_at = time.time()

    # ------------------------------------------------------------------
    # 阈值扫描（调用 core/eval_comprehensive.py 子进程）
    # ------------------------------------------------------------------

    def _run_threshold_sweep(self, job: EvalJob) -> None:
        import csv

        job.status = "running"
        job.started_at = time.time()
        job.progress_msg = "准备评估环境..."

        try:
            params = job.params
            image_dir = params["image_dir"]
            db_path = params["db_path"]
            thresholds_str = params["thresholds"]
            device = params["device"]

            if not Path(image_dir).exists():
                raise FileNotFoundError(f"测试图片目录不存在: {image_dir}")
            if not Path(db_path).exists():
                raise FileNotFoundError(f"特征数据库不存在: {db_path}")

            with tempfile.TemporaryDirectory() as tmpdir:
                job.progress_msg = "正在运行阈值扫描（这可能需要几分钟）..."
                job.progress = 10.0

                cmd = [
                    sys.executable, "-m",
                    "apps.recognition_system.core.eval_comprehensive",
                    "--weights", job.weights_path,
                    "--model-name", job.model_name,
                    "--db-path", db_path,
                    "--input-dir", image_dir,
                    "--thresholds", thresholds_str,
                    "--device", device,
                    "--out-dir", tmpdir,
                    "--skip-inference-timing",
                ]

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=900,
                )

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"子进程失败 (code={proc.returncode}):\n"
                        + (proc.stderr or proc.stdout or "未知错误").strip()
                    )

                job.progress = 80.0
                job.progress_msg = "读取评估结果..."

                summary_csv = Path(tmpdir) / "metrics_summary.csv"
                summary_rows: List[Dict] = []
                if summary_csv.exists():
                    with summary_csv.open("r", encoding="utf-8", newline="") as f:
                        summary_rows = list(csv.DictReader(f))

                detailed_json = Path(tmpdir) / "metrics_detailed.json"
                detailed: Dict = {}
                if detailed_json.exists():
                    detailed = json.loads(detailed_json.read_text(encoding="utf-8"))

                job.result = {
                    "type": "threshold_sweep",
                    "model": job.model_name,
                    "weights": Path(job.weights_path).name,
                    "thresholds": thresholds_str,
                    "summary": summary_rows,
                    "detailed": detailed,
                    "chart_data": _extract_chart_data(summary_rows),
                }
                job.status = "done"
                job.progress = 100.0
                job.progress_msg = "阈值扫描完成"

        except subprocess.TimeoutExpired:
            job.status = "error"
            job.error = "评估超时（>15分钟）"
        except Exception:
            job.status = "error"
            job.error = traceback.format_exc()
            job.progress_msg = "评估出错"
        finally:
            job.finished_at = time.time()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _extract_chart_data(summary_rows: List[Dict]) -> Dict[str, List]:
    chart: Dict[str, List] = {
        "thresholds": [], "rank1_accuracy": [], "far": [],
        "frr": [], "tar_at_far_1e3": [], "f1_score": [], "auc_score": [],
    }
    for row in summary_rows:
        try:
            chart["thresholds"].append(float(row.get("threshold", 0)))
            chart["rank1_accuracy"].append(float(row.get("rank1_accuracy", 0)))
            chart["far"].append(float(row.get("far", 0)))
            chart["frr"].append(float(row.get("frr", 0)))
            chart["tar_at_far_1e3"].append(float(row.get("tar_at_far_1e3", 0)))
            chart["f1_score"].append(float(row.get("f1_score", 0)))
            chart["auc_score"].append(float(row.get("auc_score", 0)))
        except (ValueError, TypeError):
            continue
    return chart
