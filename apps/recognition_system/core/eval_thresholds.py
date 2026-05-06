#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sweep for face_system recognize-dir")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-name", default="iresnet50")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--match-reduce", default="topk_mean", choices=["best", "mean", "topk_mean"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--detector-backend", default="mtcnn", choices=["mtcnn"])
    parser.add_argument("--det-conf-threshold", type=float, default=0.60)
    parser.add_argument("--det-min-size", type=int, default=40)
    parser.add_argument("--out-dir", default="/root/FaceRec/face_runtime/reports/threshold_sweep")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_thresholds(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No valid thresholds provided")
    return values



def evaluate_one(csv_path: Path) -> Dict[str, float]:
    total = 0
    accepted = 0
    correct = 0
    false_accept = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            pred = row["name"]
            is_accepted = int(row["accepted"]) == 1
            rel = row["image_path"]
            gt = rel.split("/")[0] if "/" in rel else rel

            if is_accepted:
                accepted += 1
                if pred == gt:
                    correct += 1
                else:
                    false_accept += 1

    rejected = total - accepted
    acc = (correct / total) if total else 0.0
    accept_rate = (accepted / total) if total else 0.0
    precision = (correct / accepted) if accepted else 0.0
    far = (false_accept / total) if total else 0.0

    return {
        "total_faces": total,
        "accepted_faces": accepted,
        "rejected_faces": rejected,
        "correct_accepted": correct,
        "false_accept": false_accept,
        "closed_set_acc": acc,
        "accept_rate": accept_rate,
        "precision_on_accepted": precision,
        "false_accept_rate": far,
    }


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    summary_rows: List[Dict[str, float]] = []

    for th in thresholds:
        report_csv = out_dir / f"report_th_{th:.2f}.csv"

        cmd = [
            sys.executable,
            "-m",
            "face_system.cli",
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

        print(f"[run] threshold={th:.2f}")
        subprocess.run(cmd, check=True)

        metrics = evaluate_one(report_csv)
        row = {
            "threshold": f"{th:.2f}",
            **metrics,
        }
        summary_rows.append(row)

    fieldnames = [
        "threshold",
        "total_faces",
        "accepted_faces",
        "rejected_faces",
        "correct_accepted",
        "false_accept",
        "closed_set_acc",
        "accept_rate",
        "precision_on_accepted",
        "false_accept_rate",
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[done] summary={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
