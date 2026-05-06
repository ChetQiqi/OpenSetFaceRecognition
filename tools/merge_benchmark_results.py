#!/usr/bin/env python3
"""
合并基准测试和评估结果，生成完整的性能表格
运行: python merge_benchmark_results.py
"""

import json
from pathlib import Path
import argparse


def load_latest_evaluation():
    """从log或evaluation_results目录加载最新的评估结果"""
    # 优先查找 evaluation_results/metrics_detailed.json（新格式）
    simple_path = Path("evaluation_results/metrics_detailed.json")
    if simple_path.exists():
        with open(simple_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 兼容旧格式：log/evaluation_results_*/
    log_dir = Path("log")
    if log_dir.exists():
        eval_dirs = sorted(log_dir.glob("evaluation_results_*"), reverse=True)
        if eval_dirs:
            metrics_file = eval_dirs[0] / "metrics_detailed.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

    return None


def generate_complete_latex(benchmark_results, eval_results=None, output_path="complete_table.tex"):
    """生成包含所有指标的完整LaTeX表格"""

    latex = r"""\begin{table}[htbp]
\centering
\caption{系统综合性能测试结果}
\label{tab:system_performance}
\begin{tabular}{llc}
\toprule
评估维度 & 测试指标 & 测试结果 \\
\midrule
"""

    # ---- 实时性 ----
    single_frame = benchmark_results.get('single_frame', {})
    realtime_fps = benchmark_results.get('realtime_fps', {})

    latex += r"\multirow{2}{*}{实时性} & 单人脸检测与识别耗时 & "
    if 'mean_ms' in single_frame:
        latex += f"{single_frame['mean_ms']:.0f} ms \\\\\n"
    else:
        latex += "N/A \\\\\n"

    latex += r" & 摄像头实时处理帧率 (FPS) & "
    if 'avg_fps' in realtime_fps:
        latex += f"{realtime_fps['avg_fps']:.1f} fps \\\\\n"
    else:
        latex += "N/A \\\\\n"

    latex += r"\hline" + "\n"

    # ---- 准确性 ----
    latex += r"\multirow{2}{*}{准确性}"

    if eval_results:
        # 处理新格式：按阈值分组的结果
        result_045 = eval_results.get("0.45", {})

        # 计算 EER
        if "eer" in result_045:
            eer_percent = result_045["eer"] * 100
            latex += f" & 等错误率 (EER) & {eer_percent:.2f}\\% \\\\\n"
        else:
            latex += r" & 等错误率 (EER) & N/A \\" + "\n"

        # 获取 0.45 阈值下的识别准确率
        if "rank1_accuracy" in result_045:
            acc_045 = result_045["rank1_accuracy"] * 100
            latex += f" & 默认阈值 (0.45) 下的识别准确率 & {acc_045:.2f}\\% \\\\\n"
        else:
            latex += r" & 默认阈值 (0.45) 下的识别准确率 & N/A \\" + "\n"
    else:
        latex += r" & 等错误率 (EER) & 待测试 \\" + "\n"
        latex += r" & 默认阈值 (0.45) 下的识别准确率 & 待测试 \\" + "\n"

    latex += r"\hline" + "\n"

    # ---- 资源占用 ----
    gpu_info = benchmark_results.get('resources', {}).get('gpu', {})
    mem_info = benchmark_results.get('resources', {}).get('memory', {})

    if gpu_info.get('gpu_available'):
        gpu_name_short = gpu_info.get('gpu_name', 'GPU').replace('NVIDIA GeForce ', '')
        latex += r"\multirow{2}{*}{资源占用} & "
        latex += f"显存占用 ({gpu_name_short}) & "
        latex += f"{gpu_info['gpu_memory_reserved_gb']:.1f} GB \\\\\n"
    else:
        latex += r"\multirow{2}{*}{资源占用} & "
        latex += "显存占用 & CPU 模式 \\\\\n"

    latex += r" & 内存平均占用 & "
    if 'cpu_memory_rss_mb' in mem_info:
        latex += f"{mem_info['cpu_memory_rss_mb']:.0f} MB \\\\\n"
    else:
        latex += "N/A \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    Path(output_path).write_text(latex, encoding='utf-8')
    print(f"✅ 完整LaTeX表格已保存: {output_path}")
    return latex


def main():
    parser = argparse.ArgumentParser("Merge benchmark and evaluation results")
    parser.add_argument("--benchmark-json", default="benchmark_results.json",
                        help="Benchmark results JSON file")
    parser.add_argument("--eval-json", default="",
                        help="Evaluation results JSON file (optional, auto-detect if not specified)")
    parser.add_argument("--output", default="complete_table.tex",
                        help="Output LaTeX file")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("合并基准测试与评估结果")
    print("="*60 + "\n")

    # 1. 加载基准测试结果
    benchmark_path = Path(args.benchmark_json)
    if not benchmark_path.exists():
        print(f"❌ 找不到基准测试结果: {args.benchmark_json}")
        print("   请先运行: python benchmark_system.py ...\n")
        return

    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark_results = json.load(f)
    print(f"✅ 已加载基准测试结果: {args.benchmark_json}")

    # 2. 加载评估结果
    eval_results = None
    if args.eval_json:
        eval_path = Path(args.eval_json)
        if eval_path.exists():
            with open(eval_path, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)
            print(f"✅ 已加载评估结果: {args.eval_json}")
        else:
            print(f"⚠️  找不到评估结果: {args.eval_json}")
    else:
        # 自动检测
        eval_results = load_latest_evaluation()
        if eval_results:
            print("✅ 自动检测到最新评估结果")
        else:
            print("⚠️  未找到评估结果，准确性指标将标记为'待测试'")
            print("   提示: 运行 python apps/recognition_system/core/eval_comprehensive.py ...\n")

    # 3. 生成完整表格
    print()
    latex = generate_complete_latex(benchmark_results, eval_results, args.output)
    print()
    print("="*60)
    print("LaTeX 表格预览:")
    print("="*60)
    print(latex)


if __name__ == "__main__":
    main()
