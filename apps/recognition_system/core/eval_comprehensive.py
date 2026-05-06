#!/usr/bin/env python3
"""
Enhanced threshold evaluation with comprehensive metrics
Integrates with existing eval_thresholds.py
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import json

from .metrics import (
    FaceRecognitionMetrics,
    ModelMetrics,
    MetricsVisualizer,
    MetricsResult,
    save_metrics_to_json,
    save_metrics_to_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced threshold sweep with comprehensive metrics for face_system recognize"
    )
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--model-name", default="iresnet50", help="Model architecture name")
    parser.add_argument("--db-path", required=True, help="Path to face database")
    parser.add_argument("--input-dir", required=True, help="Directory with test images")
    parser.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
                       help="Comma-separated threshold values")
    parser.add_argument("--match-reduce", default="topk_mean",
                       choices=["best", "mean", "topk_mean"], help="Score reduction method")
    parser.add_argument("--topk", type=int, default=3, help="Top-k for topk_mean")
    parser.add_argument("--detector-backend", default="mtcnn",
                       choices=["mtcnn"], help="Face detector backend (MTCNN only)")
    parser.add_argument("--det-conf-threshold", type=float, default=0.60,
                       help="Detector confidence threshold")
    parser.add_argument("--det-min-size", type=int, default=40, help="Minimum detection size")
    parser.add_argument("--out-dir", default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="Number of runs for inference timing measurement")
    parser.add_argument("--skip-inference-timing", action="store_true",
                       help="Skip inference timing measurements")

    return parser.parse_args()


def parse_thresholds(raw: str) -> List[float]:
    """Parse comma-separated threshold values"""
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No valid thresholds provided")
    return sorted(set(values))


def evaluate_threshold_metrics(csv_path: Path, threshold: float) -> MetricsResult:
    """Evaluate metrics from recognition results CSV"""
    metrics_calc = FaceRecognitionMetrics(threshold=threshold)

    total = 0
    correct_predictions = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            pred = row.get("name", "unknown")
            is_accepted = int(row.get("accepted", 0)) == 1
            rel = row.get("image_path", "")

            # Extract ground truth from image path
            gt = rel.split("/")[0] if "/" in rel else rel
            gt = gt.split("\\")[0] if "\\" in gt else gt

            # Get similarity score if available
            score = float(row.get("score", 0.0)) if "score" in row else (1.0 if is_accepted else 0.0)

            # Prepare labels
            pred_id = int(row.get("pred_id", 0)) if "pred_id" in row else (1 if is_accepted else 0)
            gt_id = int(row.get("gt_id", 0)) if "gt_id" in row else 0

            # Add to metrics
            is_correct = (pred == gt) if is_accepted else False
            if is_correct:
                correct_predictions += 1

            metrics_calc.add_sample(
                score=score,
                pred_label=pred_id,
                gt_label=gt_id,
                is_correct=is_correct
            )

            # Add inference time if available
            if "inference_time" in row:
                try:
                    inference_time = float(row["inference_time"])
                    metrics_calc.add_inference_time(inference_time)
                except ValueError:
                    pass

    return metrics_calc.compute_all_metrics()


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run recognition at each threshold
    print("\n" + "=" * 70)
    print("Face Recognition Evaluation with Comprehensive Metrics")
    print("=" * 70)

    summary_rows: List[Dict] = []
    all_results: Dict[float, MetricsResult] = {}

    for th in thresholds:
        report_csv = out_dir / f"report_th_{th:.2f}.csv"

        print(f"\n[Processing] threshold={th:.2f}")
        print("-" * 70)

        # Run recognition command
        cmd = [
            sys.executable,
            "-m",
            "apps.recognition_system.core.cli",
            "recognize-dir",
            "--input-dir",
            args.input_dir,
            "--weights",
            args.weights,
            "--model-name",
            args.model_name,
            "--db-path",
            args.db_path,
            "--device",
            args.device,
            "--threshold",
            f"{th:.2f}",
            "--match-reduce",
            args.match_reduce,
            "--topk",
            str(args.topk),
            "--detector-backend",
            args.detector_backend,
            "--det-conf-threshold",
            str(args.det_conf_threshold),
            "--det-min-size",
            str(args.det_min_size),
            "--report-csv",
            str(report_csv),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error running recognition: {e.stderr.decode()}")
            continue

        # Evaluate metrics
        print(f"  📊 Computing metrics...")
        metrics = evaluate_threshold_metrics(report_csv, th)
        all_results[th] = metrics

        # Create row for summary
        row = {
            "threshold": f"{th:.2f}",
            # Accuracy metrics
            "rank1_accuracy": f"{metrics.rank1_accuracy:.4f}",
            # FAR/FRR metrics
            "far": f"{metrics.far:.6f}",
            "frr": f"{metrics.frr:.6f}",
            "tar_at_far_1e3": f"{metrics.tar_at_far_1e3:.4f}",
            "tar_at_far_1e2": f"{metrics.tar_at_far_1e2:.4f}",
            "tar_at_far_1e1": f"{metrics.tar_at_far_1e1:.4f}",
            # ROC metrics
            "eer": f"{metrics.eer:.6f}",
            "auc_score": f"{metrics.auc_score:.4f}",
            # Classification metrics
            "f1_score": f"{metrics.f1:.4f}",
            "precision": f"{metrics.precision:.4f}",
            "recall": f"{metrics.recall:.4f}",
            # Performance metrics
            "inference_time_mean_ms": f"{metrics.inference_time_mean:.2f}",
            "inference_time_std_ms": f"{metrics.inference_time_std:.2f}",
            "inference_time_total_ms": f"{metrics.inference_time_total:.2f}",
            # Metadata
            "num_samples": metrics.num_samples,
            "num_identities": metrics.num_identities,
        }
        summary_rows.append(row)

        # Print key metrics
        print(f"\n  📈 Key Metrics:")
        print(f"    • Rank-1 Accuracy:     {metrics.rank1_accuracy:.4f}")
        print(f"    • FAR:                 {metrics.far:.6f}")
        print(f"    • TAR @ FAR=1e-3:      {metrics.tar_at_far_1e3:.4f}")
        print(f"    • F1-Score:            {metrics.f1:.4f}")
        print(f"    • Precision/Recall:    {metrics.precision:.4f} / {metrics.recall:.4f}")
        if metrics.inference_time_mean > 0:
            print(f"    • Inference Time:      {metrics.inference_time_mean:.2f} ± {metrics.inference_time_std:.2f} ms")

    # Save summary to CSV
    summary_csv = out_dir / "metrics_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\n✅ Summary saved to {summary_csv}")

    # Save detailed results as JSON
    results_json = out_dir / "metrics_detailed.json"
    with results_json.open("w", encoding="utf-8") as f:
        results_dict = {}
        for th, metrics in all_results.items():
            results_dict[f"{th:.2f}"] = {
                "rank1_accuracy": metrics.rank1_accuracy,
                "far": metrics.far,
                "frr": metrics.frr,
                "tar_at_far_1e3": metrics.tar_at_far_1e3,
                "tar_at_far_1e2": metrics.tar_at_far_1e2,
                "tar_at_far_1e1": metrics.tar_at_far_1e1,
                "eer": metrics.eer,
                "auc_score": metrics.auc_score,
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "inference_time_mean": metrics.inference_time_mean,
                "inference_time_std": metrics.inference_time_std,
                "num_samples": metrics.num_samples,
                "num_identities": metrics.num_identities,
            }
        json.dump(results_dict, f, indent=2)
    print(f"✅ Detailed results saved to {results_json}")

    # Find and print best threshold based on different metrics
    if all_results:
        print("\n" + "=" * 70)
        print("Best Performance by Metric:")
        print("=" * 70)

        best_rank1 = max(all_results.items(), key=lambda x: x[1].rank1_accuracy)
        best_tar = max(all_results.items(), key=lambda x: x[1].tar_at_far_1e3)
        best_f1 = max(all_results.items(), key=lambda x: x[1].f1)

        print(f"\n🏆 Best Rank-1 Accuracy:  {best_rank1[0]:.2f} ({best_rank1[1].rank1_accuracy:.4f})")
        print(f"🏆 Best TAR @ FAR=1e-3:   {best_tar[0]:.2f} ({best_tar[1].tar_at_far_1e3:.4f})")
        print(f"🏆 Best F1-Score:         {best_f1[0]:.2f} ({best_f1[1].f1:.4f})")

    print("\n" + "=" * 70)
    print("✅ Evaluation completed successfully!")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
