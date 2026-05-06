"""
使用准备好的数据进行Fixed vs Adaptive对比评估

输入数据来源：prepare_open_set_data.py生成的数据集
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到Python路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_prepared_dataset(data_dir: Path) -> Dict:
    """
    加载prepare_open_set_data.py准备的数据集。

    Returns:
        {
            "gallery": [(name, emb), ...],
            "test_known_embeddings": np.ndarray,
            "test_known_labels": List[str],
            "test_unknown_embeddings": np.ndarray,
            "test_unknown_labels": List[str],
            "known_labels": List[str],
            "config": Dict
        }
    """
    # Load gallery
    gallery_embeddings = np.load(data_dir / "gallery.npy")
    with open(data_dir / "gallery_labels.txt", "r", encoding="utf-8") as f:
        gallery_labels = [line.strip() for line in f]

    gallery = list(zip(gallery_labels, gallery_embeddings))

    # Load test known
    test_known_embeddings = np.load(data_dir / "test_known.npy")
    with open(data_dir / "test_known_labels.txt", "r", encoding="utf-8") as f:
        test_known_labels = [line.strip() for line in f]

    # Load test unknown
    test_unknown_embeddings = np.load(data_dir / "test_unknown.npy")
    with open(data_dir / "test_unknown_labels.txt", "r", encoding="utf-8") as f:
        test_unknown_labels = [line.strip() for line in f]

    # Load config
    with open(data_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "gallery": gallery,
        "test_known_embeddings": test_known_embeddings,
        "test_known_labels": test_known_labels,
        "test_unknown_embeddings": test_unknown_embeddings,
        "test_unknown_labels": test_unknown_labels,
        "known_labels": config["known_labels"],
        "config": config,
    }


def compute_adaptive_stats_from_gallery(gallery: List[Tuple[str, np.ndarray]], k: float = 2.0):
    """
    从gallery计算per-identity adaptive statistics (使用校准分数)。

    Args:
        gallery: [(name, emb), ...]
        k: Standard deviation multiplier

    Returns:
        Dict[str, IdentityStatistics]
    """
    from collections import defaultdict
    from apps.recognition_system.core.matcher import cosine_similarity
    from apps.recognition_system.core.adaptive_threshold import IdentityStatistics
    from apps.recognition_system.core.operations import similarity_con
    import time

    # Group by identity
    embeddings_by_id = defaultdict(list)
    for name, emb in gallery:
        embeddings_by_id[name].append(emb)

    identity_stats = {}

    for person_name, embeddings in tqdm(embeddings_by_id.items(), desc="计算Per-Identity统计", unit="person"):
        if len(embeddings) < 2:
            # Not enough embeddings, use fallback for calibrated space
            identity_stats[person_name] = IdentityStatistics(
                identity_name=person_name,
                person_id=-1,
                mean_genuine_score=0.85,  # Default for calibrated space
                std_genuine_score=0.05,
                adaptive_threshold=0.75,
                sample_count=0,
            )
            continue

        # Compute genuine pairs (using calibrated scores)
        genuine_scores = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                raw_score = cosine_similarity(embeddings[i], embeddings[j])
                # ✨ Apply calibration to be consistent with recognition
                calibrated = similarity_con(raw_score)
                genuine_scores.append(calibrated)

        # Compute statistics
        mean_score = float(np.mean(genuine_scores))
        std_score = float(np.std(genuine_scores))
        min_score = float(np.min(genuine_scores))
        max_score = float(np.max(genuine_scores))

        # Compute adaptive threshold
        threshold = mean_score - k * std_score

        # Clamp to safe range for calibrated similarity:
        # Calibrated scores typically range 0.5-1.0 for genuine pairs
        threshold = max(threshold, 0.5)
        threshold = min(threshold, mean_score)

        identity_stats[person_name] = IdentityStatistics(
            identity_name=person_name,
            person_id=-1,
            mean_genuine_score=mean_score,
            std_genuine_score=std_score,
            min_genuine_score=min_score,
            max_genuine_score=max_score,
            sample_count=len(genuine_scores),
            adaptive_threshold=threshold,
            last_updated=time.time(),
        )

        print(
            f"[Adaptive] {person_name:30s}: μ={mean_score:.3f}, σ={std_score:.3f}, "
            f"threshold={threshold:.3f} (n={len(genuine_scores)} pairs)"
        )

    return identity_stats


def recognize_with_fixed_threshold(
    query_embedding: np.ndarray,
    gallery: List,
    threshold: float,
    match_reduce: str = "topk_mean",
    topk: int = 3,
) -> Dict:
    """Fixed threshold recognition."""
    from apps.recognition_system.core.matcher import find_best_match
    from apps.recognition_system.core.operations import similarity_con

    match = find_best_match(query_embedding, gallery, reduce=match_reduce, topk=topk)
    raw_score = float(match.score)
    calibrated_score = similarity_con(raw_score)
    accepted = calibrated_score >= threshold and match.name != "Unknown"

    return {
        "name": match.name if accepted else "Unknown",
        "score": calibrated_score,
        "raw_score": raw_score,
        "accepted": accepted,
    }


def recognize_with_adaptive_threshold(
    query_embedding: np.ndarray,
    gallery: List,
    identity_stats: Dict,
    global_fallback: float = 0.45,
    match_reduce: str = "topk_mean",
    topk: int = 3,
) -> Dict:
    """Adaptive threshold recognition."""
    from apps.recognition_system.core.adaptive_threshold import (
        adaptive_recognize_face,
        get_all_scores_for_query,
    )
    from apps.recognition_system.core.matcher import find_best_match
    from apps.recognition_system.core.operations import similarity_con

    match = find_best_match(query_embedding, gallery, reduce=match_reduce, topk=topk)
    raw_score = float(match.score)
    calibrated_score = similarity_con(raw_score)

    _, all_scores = get_all_scores_for_query(
        query_embedding, gallery, reduce=match_reduce, topk=topk
    )

    # ✨ Calibrate all_scores for consistency with identity statistics
    all_scores_calibrated = [similarity_con(s) for s in all_scores]

    decision = adaptive_recognize_face(
        query_embedding=query_embedding,
        gallery=gallery,
        identity_stats=identity_stats,
        best_match_name=match.name,
        best_match_score=calibrated_score,  # ✨ Use calibrated score for z-score computation
        all_scores=all_scores_calibrated,     # ✨ Use calibrated scores for distance ratio
        calibrated_score=calibrated_score,
        global_fallback=global_fallback,
    )

    return {
        "name": decision.identity_name if decision.accepted else "Unknown",
        "score": calibrated_score,
        "raw_score": raw_score,
        "accepted": decision.accepted,
        "decision_reason": decision.decision_reason,
        "adaptive_threshold": decision.adaptive_threshold,
        "z_score": decision.z_score,
        "distance_ratio": decision.distance_ratio,
    }


def run_evaluation(
    test_embeddings: np.ndarray,
    test_labels: List[str],
    gallery: List,
    known_labels: List[str],
    use_adaptive: bool = False,
    identity_stats: Dict = None,
    fixed_threshold: float = 0.5,
) -> Tuple[List[Dict], List[Dict], float]:
    """Run evaluation."""
    predictions = []
    ground_truth = []
    latencies = []

    known_set = set(known_labels)

    for embedding, label in tqdm(
        zip(test_embeddings, test_labels),
        total=len(test_embeddings),
        desc=f"Evaluating ({'Adaptive' if use_adaptive else 'Fixed'})",
    ):
        is_unknown = label not in known_set
        ground_truth.append({"name": label, "is_unknown": is_unknown})

        t0 = time.time()
        if use_adaptive:
            result = recognize_with_adaptive_threshold(
                embedding, gallery, identity_stats, global_fallback=fixed_threshold
            )
        else:
            result = recognize_with_fixed_threshold(
                embedding, gallery, threshold=fixed_threshold
            )
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)

        predictions.append(result)

    avg_latency = np.mean(latencies)
    return predictions, ground_truth, avg_latency


def plot_comparison_chart(
    fixed_metrics: Dict, adaptive_metrics: Dict, output_path: str
) -> None:
    """Plot comparison chart."""
    metrics_names = ["OSR", "KCA", "UDR", "F1 (Unknown)"]
    fixed_values = [
        fixed_metrics["osr"] * 100,
        fixed_metrics["kca"] * 100,
        fixed_metrics["udr"] * 100,
        fixed_metrics["f1_unknown"] * 100,
    ]
    adaptive_values = [
        adaptive_metrics["osr"] * 100,
        adaptive_metrics["kca"] * 100,
        adaptive_metrics["udr"] * 100,
        adaptive_metrics["f1_unknown"] * 100,
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, fixed_values, width, label="Fixed Threshold", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, adaptive_values, width, label="Adaptive Threshold", color="#ff7f0e")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Fixed vs Adaptive Threshold - Open-Set Recognition", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用准备好的数据进行Fixed vs Adaptive对比评估"
    )
    parser.add_argument(
        "--data-dir",
        default="benchmark/open_set_data",
        help="数据目录（prepare_open_set_data.py的输出）",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.45,
        help="Fixed threshold值（默认0.45）",
    )
    parser.add_argument(
        "--output-dir",
        default="thesis_eval",
        help="结果输出目录",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("Fixed vs Adaptive Threshold - Open-Set Evaluation")
    print("=" * 100)
    print(f"数据目录: {args.data_dir}")
    print(f"Fixed threshold: {args.fixed_threshold}")
    print(f"输出目录: {args.output_dir}\n")

    # Load dataset
    print("1. 加载数据集...")
    data_dir = Path(args.data_dir)
    dataset = load_prepared_dataset(data_dir)

    print(f"   ✓ Gallery: {len(dataset['gallery'])} embeddings from {dataset['config']['num_known']} persons")
    print(f"   ✓ Test Known: {len(dataset['test_known_embeddings'])} samples")
    print(f"   ✓ Test Unknown: {len(dataset['test_unknown_embeddings'])} samples")

    # Compute adaptive statistics
    print("\n2. 计算Adaptive Thresholds...")
    identity_stats = compute_adaptive_stats_from_gallery(dataset["gallery"])

    # Combine test sets
    print("\n3. 合并测试集...")
    test_embeddings = np.vstack([
        dataset["test_known_embeddings"],
        dataset["test_unknown_embeddings"]
    ])
    test_labels = dataset["test_known_labels"] + dataset["test_unknown_labels"]
    print(f"   ✓ Total test samples: {len(test_embeddings)}")

    # Run evaluations
    print("\n4️⃣  评估Fixed Threshold...")
    pred_fixed, gt_fixed, latency_fixed = run_evaluation(
        test_embeddings,
        test_labels,
        dataset["gallery"],
        dataset["known_labels"],
        use_adaptive=False,
        fixed_threshold=args.fixed_threshold,
    )

    print("\n5️⃣  评估Adaptive Threshold...")
    pred_adaptive, gt_adaptive, latency_adaptive = run_evaluation(
        test_embeddings,
        test_labels,
        dataset["gallery"],
        dataset["known_labels"],
        use_adaptive=True,
        identity_stats=identity_stats,
        fixed_threshold=args.fixed_threshold,
    )

    # Compute metrics
    print("\n6️⃣  计算Open-Set指标...")
    from apps.recognition_system.core.open_set_metrics import (
        compute_open_set_metrics,
        metrics_to_dict,
    )

    metrics_fixed = compute_open_set_metrics(pred_fixed, gt_fixed, dataset["known_labels"])
    metrics_adaptive = compute_open_set_metrics(pred_adaptive, gt_adaptive, dataset["known_labels"])

    # Print results
    print("\n" + "=" * 100)
    print("📊 结果对比")
    print("=" * 100)

    print("\n{:30s} | {:>15s} | {:>15s} | {:>15s}".format(
        "Metric", "Fixed", "Adaptive", "Improvement"
    ))
    print("-" * 100)

    results_table = []

    metrics_comparison = {
        "OSR (Open-Set Recognition)": ("osr", True),
        "KCA (Known Class Accuracy)": ("kca", True),
        "UDR (Unknown Detection Rate)": ("udr", True),
        "Precision (Unknown)": ("precision_unknown", True),
        "F1-Score (Unknown)": ("f1_unknown", True),
        "Avg Latency (ms)": (None, False),
    }

    for metric_name, (metric_key, is_percentage) in metrics_comparison.items():
        if metric_key:
            val_fixed = metrics_to_dict(metrics_fixed)[metric_key]
            val_adaptive = metrics_to_dict(metrics_adaptive)[metric_key]
        else:
            val_fixed = latency_fixed
            val_adaptive = latency_adaptive

        if is_percentage:
            val_fixed_str = f"{val_fixed * 100:.2f}%"
            val_adaptive_str = f"{val_adaptive * 100:.2f}%"
            improvement = (val_adaptive - val_fixed) * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            val_fixed_str = f"{val_fixed:.2f}"
            val_adaptive_str = f"{val_adaptive:.2f}"
            improvement = val_adaptive - val_fixed
            improvement_str = f"{improvement:+.2f}"

        print(f"{metric_name:30s} | {val_fixed_str:>15s} | {val_adaptive_str:>15s} | {improvement_str:>15s}")

        results_table.append({
            "Metric": metric_name,
            "Fixed": val_fixed_str,
            "Adaptive": val_adaptive_str,
            "Improvement": improvement_str,
        })

    print("=" * 100)

    # Save results
    print(f"\n7️⃣  保存结果...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    print("  💾 保存CSV...")
    csv_path = output_dir / "fixed_vs_adaptive_results.csv"
    df = pd.DataFrame(results_table)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"     ✓ {csv_path}")

    # JSON
    print("  💾 保存JSON...")
    json_path = output_dir / "fixed_vs_adaptive_results.json"
    results_json = {
        "dataset": {
            "num_known": dataset["config"]["num_known"],
            "num_unknown": dataset["config"]["num_unknown"],
            "gallery_size": dataset["config"]["gallery_size"],
            "test_known_size": dataset["config"]["test_known_size"],
            "test_unknown_size": dataset["config"]["test_unknown_size"],
        },
        "fixed_threshold": {
            "metrics": metrics_to_dict(metrics_fixed),
            "avg_latency_ms": latency_fixed,
        },
        "adaptive_threshold": {
            "metrics": metrics_to_dict(metrics_adaptive),
            "avg_latency_ms": latency_adaptive,
        },
        "improvements": {
            "osr": (metrics_adaptive.osr - metrics_fixed.osr) * 100,
            "kca": (metrics_adaptive.kca - metrics_fixed.kca) * 100,
            "udr": (metrics_adaptive.udr - metrics_fixed.udr) * 100,
            "f1_unknown": (metrics_adaptive.f1_unknown - metrics_fixed.f1_unknown) * 100,
            "latency_overhead": latency_adaptive - latency_fixed,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"   ✓ JSON: {json_path}")

    # Plot
    print("  📊 生成可视化图表...")
    plot_path = output_dir / "fixed_vs_adaptive_comparison.png"
    plot_comparison_chart(
        metrics_to_dict(metrics_fixed),
        metrics_to_dict(metrics_adaptive),
        str(plot_path),
    )

    print("\n" + "=" * 100)
    print("✅✅✅ 评估完成！")
    print("=" * 100)
    print(f"\n📁 结果保存目录: {output_dir}/")
    print(f"\n📊 生成的文件:")
    print(f"   📄 {csv_path.name}")
    print(f"   📄 {json_path.name}")
    print(f"   📄 {plot_path.name}")
    print(f"\n🎉 所有结果已准备好，可以写入论文了！")
    print("=" * 100)


if __name__ == "__main__":
    main()
