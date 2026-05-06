"""
摄像头性能分析可视化
从camera_benchmark_report.json生成性能分析图表
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_report(report_path: str) -> dict:
    """加载报告数据"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_chart_01_latency_stages(report: dict):
    """图表1：延迟分解 - 各阶段耗时柱状图"""
    print("生成 chart_camera_01_latency_stages.png...")

    latency = report['performance_metrics']['latency_stages']

    stages = ['检测', '提取', '匹配', '总延迟']
    means = [
        latency['detection']['mean'],
        latency['extraction']['mean'],
        latency['matching']['mean'],
        latency['total']['mean']
    ]
    p95s = [
        latency['detection']['p95'],
        latency['extraction']['p95'],
        latency['matching']['p95'],
        latency['total']['p95']
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax.bar(x - width/2, means, width, label='平均延迟', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, p95s, width, label='P95延迟', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('处理阶段', fontsize=12, fontweight='bold')
    ax.set_ylabel('延迟 (毫秒)', fontsize=12, fontweight='bold')
    ax.set_title('🎯 摄像头实时性能：各阶段延迟分解', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('chart_camera_01_latency_stages.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_chart_02_realtime_performance(report: dict):
    """图表2：实时性能指标对比"""
    print("生成 chart_camera_02_realtime_performance.png...")

    perf = report['performance_metrics']

    fps_values = [
        perf['fps_actual'],
        perf['fps_processed'],
        perf['fps_theoretical']
    ]
    fps_labels = ['实际帧率\n(capture)', '处理帧率\n(process)', '理论帧率\n(compute)']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：FPS对比
    colors = ['#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax1.bar(fps_labels, fps_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.axhline(y=25, color='red', linestyle='--', linewidth=2, label='实时标准 (25 FPS)')
    ax1.axhline(y=30, color='green', linestyle='--', linewidth=2, label='高清标准 (30 FPS)')

    ax1.set_ylabel('帧率 (FPS)', fontsize=12, fontweight='bold')
    ax1.set_title('帧率对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, fps_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 右图：延迟分布（箱线图）
    latency = report['performance_metrics']['latency_stages']['total']
    box_data = [
        [latency['min'], latency['p50'], latency['mean'], latency['p95'], latency['max']]
    ]

    bp = ax2.boxplot(box_data, labels=['总延迟'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#96CEB4')
    bp['boxes'][0].set_alpha(0.7)

    ax2.set_ylabel('延迟 (毫秒)', fontsize=12, fontweight='bold')
    ax2.set_title('延迟分布 (P50/P95)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('chart_camera_02_realtime_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_chart_03_recognition_dashboard(report: dict):
    """图表3：识别性能仪表盘"""
    print("生成 chart_camera_03_recognition_dashboard.png...")

    config = report['config']
    recog = report['recognition_metrics']
    perf = report['performance_metrics']

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # 大标题
    fig.suptitle('📊 摄像头实时识别系统性能仪表盘', fontsize=16, fontweight='bold', y=0.98)

    # KPI卡片布局
    kpis = [
        ('实时处理\nFPS', f"{perf['fps_processed']:.1f}", '#45B7D1'),
        ('人脸检测率', f"{recog['detection_rate']:.1f}%", '#FF6B6B'),
        ('识别人物数', f"{recog['unique_persons']}", '#4ECDC4'),
        ('平均延迟', f"{perf['latency_stages']['total']['mean']:.1f}ms", '#96CEB4'),
        ('P95延迟', f"{perf['latency_stages']['total']['p95']:.1f}ms", '#FFEAA7'),
        ('置信度', f"{recog['confidence_stats']['mean']:.3f}", '#DDA15E'),
    ]

    positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
    ]

    for (kpi_name, kpi_value, color), (row, col) in zip(kpis, positions):
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')

        # 绘制卡片背景
        rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                                      linewidth=2, edgecolor=color, facecolor=color, alpha=0.1,
                                      transform=ax.transAxes)
        ax.add_patch(rect)

        # 添加文本
        ax.text(0.5, 0.65, kpi_name, ha='center', va='center', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.5, 0.35, kpi_value, ha='center', va='center', fontsize=18, fontweight='bold',
               color=color, transform=ax.transAxes)

    # 下方详细信息
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')

    info_text = f"""
    采样配置: {config['duration_seconds']}秒采样 | 跳帧策略: 每{config['skip_frames']}帧 | 人脸库规模: {config['db_size']}人
    检测结果: {recog['total_detections']}次检测 | {recog['total_recognitions']}次识别 | 平均置信度: {recog['confidence_stats']['mean']:.4f}
    """

    ax_info.text(0.05, 0.5, info_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                transform=ax_info.transAxes)

    plt.savefig('chart_camera_03_recognition_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_chart_04_tracking_analysis(report: dict):
    """图表4：跟踪性能分析"""
    print("生成 chart_camera_04_tracking_analysis.png...")

    tracking = report['tracking_metrics']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：轨迹数量
    ax1.bar(['轨迹数量'], [tracking['total_tracks']], color='#45B7D1', width=0.5, alpha=0.8)
    ax1.set_ylabel('数量', fontsize=11, fontweight='bold')
    ax1.set_title('总轨迹数', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(tracking['total_tracks'] * 1.2, 1))
    for i, v in enumerate([tracking['total_tracks']]):
        ax1.text(i, v + 1, str(int(v)), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 右上：轨迹长度分布
    ax2.bar(['平均', '最长'],
           [tracking['avg_track_length'], tracking['max_track_length']],
           color=['#96CEB4', '#FFEAA7'], alpha=0.8)
    ax2.set_ylabel('帧数', fontsize=11, fontweight='bold')
    ax2.set_title('轨迹长度分析', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 左下：身份一致性
    consistency = tracking['identity_consistency']
    ax3.bar(['身份\n一致性'], [consistency], color='#4ECDC4' if consistency > 80 else '#FF6B6B', width=0.5, alpha=0.8)
    ax3.set_ylabel('百分比 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('识别身份一致性', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='良好标准')
    ax3.text(0, consistency + 3, f'{consistency:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

    # 右下：总结表
    ax4.axis('off')
    summary_data = [
        ['指标', '数值'],
        ['轨迹总数', f"{tracking['total_tracks']}"],
        ['平均轨迹长度', f"{tracking['avg_track_length']:.1f} 帧"],
        ['最长轨迹', f"{tracking['max_track_length']} 帧"],
        ['身份一致性', f"{tracking['identity_consistency']:.1f}%"],
    ]

    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 头行着色
    for i in range(2):
        table[(0, i)].set_facecolor('#45B7D1')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 奇偶行着色
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.suptitle('📍 摄像头跟踪性能分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_camera_04_tracking_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_chart_05_config_summary(report: dict):
    """图表5：配置和性能总结表"""
    print("生成 chart_camera_05_config_summary.png...")

    config = report['config']
    perf = report['performance_metrics']
    recog = report['recognition_metrics']
    tracking = report['tracking_metrics']

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    # 标题
    ax.text(0.5, 0.97, '⚙️ 摄像头测试配置与性能汇总', fontsize=16, fontweight='bold',
           ha='center', transform=ax.transAxes)

    summary_sections = [
        ('📋 测试配置', [
            ['采样时长', f"{config['duration_seconds']}秒"],
            ['摄像头ID', f"{config['camera_id'] if 'camera_id' in config else 'N/A'}"],
            ['总帧数', f"{config['total_frames']:,}"],
            ['处理帧数', f"{config['processed_frames']:,}"],
            ['人脸库规模', f"{config['db_size']:,}人"],
        ]),
        ('🎬 实时性能', [
            ['实际帧率', f"{perf['fps_actual']:.2f} FPS"],
            ['处理帧率', f"{perf['fps_processed']:.2f} FPS"],
            ['理论帧率', f"{perf['fps_theoretical']:.2f} FPS"],
            ['平均延迟', f"{perf['latency_stages']['total']['mean']:.2f}ms"],
            ['P95延迟', f"{perf['latency_stages']['total']['p95']:.2f}ms"],
        ]),
        ('🎯 识别性能', [
            ['总检测数', f"{recog['total_detections']}"],
            ['总识别数', f"{recog['total_recognitions']}"],
            ['不同人物', f"{recog['unique_persons']}"],
            ['检测率', f"{recog['detection_rate']:.2f}%"],
            ['平均置信度', f"{recog['confidence_stats']['mean']:.4f}"],
        ]),
        ('📍 跟踪性能', [
            ['轨迹数量', f"{tracking['total_tracks']}"],
            ['平均轨迹长度', f"{tracking['avg_track_length']:.1f}帧"],
            ['最长轨迹', f"{tracking['max_track_length']}帧"],
            ['身份一致性', f"{tracking['identity_consistency']:.1f}%"],
            ['---', '---'],
        ])
    ]

    y_pos = 0.88
    colors = ['#E8F4F8', '#F8E8E8', '#E8F8E8', '#F8F8E8']

    for section_idx, (title, data) in enumerate(summary_sections):
        # 章节标题
        ax.text(0.05 + (section_idx % 2) * 0.48, y_pos, title, fontsize=12, fontweight='bold',
               transform=ax.transAxes)

        # 数据行
        row_y = y_pos - 0.04
        for row_idx, (key, value) in enumerate(data):
            if key == '---':
                continue

            # 背景
            rect = mpatches.Rectangle((0.05 + (section_idx % 2) * 0.48, row_y - 0.032),
                                      0.42, 0.03, transform=ax.transAxes,
                                      facecolor=colors[section_idx], alpha=0.5, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)

            # 文本
            ax.text(0.07 + (section_idx % 2) * 0.48, row_y - 0.005, key, fontsize=9, fontweight='bold',
                   transform=ax.transAxes, va='center')
            ax.text(0.40 + (section_idx % 2) * 0.48, row_y - 0.005, value, fontsize=9,
                   transform=ax.transAxes, va='center', ha='right', family='monospace')

            row_y -= 0.035

        # 每两个章节换行
        if (section_idx + 1) % 2 == 0:
            y_pos = row_y - 0.02

    plt.savefig('chart_camera_05_config_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="摄像头性能分析可视化")
    parser.add_argument("--report", default="camera_benchmark_report.json", help="报告文件路径")
    args = parser.parse_args()

    # 检查报告文件
    if not Path(args.report).exists():
        print(f"❌ 报告文件不存在: {args.report}")
        print("请先运行 camera_interactive.py 生成报告")
        return

    print("\n" + "="*80)
    print("📊 正在生成摄像头性能分析图表...")
    print("="*80 + "\n")

    # 加载报告
    report = load_report(args.report)

    # 生成5个图表
    generate_chart_01_latency_stages(report)
    generate_chart_02_realtime_performance(report)
    generate_chart_03_recognition_dashboard(report)
    generate_chart_04_tracking_analysis(report)
    generate_chart_05_config_summary(report)

    print("\n" + "="*80)
    print("✅ 所有图表已生成完成！")
    print("="*80)
    print("\n📊 生成的文件列表（答辩用）:")
    print("  ✅ chart_camera_01_latency_stages.png - 延迟分解")
    print("  ✅ chart_camera_02_realtime_performance.png - 实时性能")
    print("  ✅ chart_camera_03_recognition_dashboard.png - 识别仪表盘")
    print("  ✅ chart_camera_04_tracking_analysis.png - 跟踪分析")
    print("  ✅ chart_camera_05_config_summary.png - 配置汇总")
    print("\n📌 推荐展示顺序:")
    print("  1. chart_camera_03_recognition_dashboard.png（快速概览）")
    print("  2. chart_camera_02_realtime_performance.png（实时性能）")
    print("  3. chart_camera_01_latency_stages.png（详细延迟分析）")
    print("  4. chart_camera_04_tracking_analysis.png（跟踪性能）")
    print("  5. chart_camera_05_config_summary.png（完整数据汇总）")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
