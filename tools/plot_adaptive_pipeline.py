"""
绘制自适应开放集人脸识别流程图

生成高质量的流程图，展示Fixed vs Adaptive两种方法的完整流程
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案 - 高对比度配色
COLOR_OFFLINE = '#B3D9FF'      # 中蓝色 - 离线阶段
COLOR_ONLINE = '#FFD699'       # 中橙色 - 在线阶段
COLOR_FIXED = '#FFB3B3'        # 中红色 - 固定阈值
COLOR_ADAPTIVE = '#B3E6B3'     # 中绿色 - 自适应阈值
COLOR_DECISION = '#E6CCFF'     # 中紫色 - 决策模块
COLOR_ARROW = '#555555'        # 深灰色 - 箭头
COLOR_TEXT = '#1a1a1a'         # 更深灰色 - 文字
COLOR_REJECT = '#FF6B6B'       # 深红色 - Unknown
COLOR_ACCEPT = '#51CF66'       # 深绿色 - Known


def draw_box(ax, x, y, width, height, text, color, fontsize=14, bold=False):
    """绘制圆角矩形框"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_TEXT,
        facecolor=color,
        linewidth=2.5,
        zorder=2
    )
    ax.add_patch(box)

    # 添加文字
    weight = 'heavy' if bold else 'bold'
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize,
        color=COLOR_TEXT,
        weight=weight,
        zorder=3
    )
    return box


def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """绘制箭头"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=COLOR_ARROW,
        linewidth=3.0,
        mutation_scale=30,
        zorder=1
    )
    ax.add_patch(arrow)

    # 添加标签
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x, mid_y, label,
            ha='center', va='bottom',
            fontsize=12,
            color=COLOR_ARROW,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none', alpha=0.8),
            zorder=3
        )


def draw_diamond(ax, x, y, width, height, text, color):
    """绘制菱形决策框"""
    vertices = [
        (x + width/2, y + height),      # 上
        (x + width, y + height/2),      # 右
        (x + width/2, y),                # 下
        (x, y + height/2),               # 左
    ]
    diamond = mpatches.Polygon(
        vertices,
        closed=True,
        edgecolor=COLOR_TEXT,
        facecolor=color,
        linewidth=2.5,
        zorder=2
    )
    ax.add_patch(diamond)

    # 添加文字（多行）
    lines = text.split('\n')
    for i, line in enumerate(lines):
        offset = (len(lines) - 1) * 0.2 / 2  # 居中偏移
        ax.text(
            x + width/2, y + height/2 + (offset - i*0.2),
            line,
            ha='center', va='center',
            fontsize=15,
            color=COLOR_TEXT,
            weight='heavy',
            zorder=3
        )


def plot_adaptive_pipeline():
    """绘制简洁的自适应开放集识别流程图"""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # ========== 标题 ==========
    ax.text(7.5, 9.5, 'Adaptive Open-Set Face Recognition',
            ha='center', va='center', fontsize=24, weight='heavy', color=COLOR_TEXT)

    # ========== 左上：离线阶段 ==========
    ax.text(2.9, 8.7, 'Offline Stage', ha='center', fontsize=16, weight='heavy', color=COLOR_TEXT)

    draw_box(ax, 1.3, 7.8, 3.2, 0.8, 'Face Database', COLOR_OFFLINE, fontsize=16)
    draw_arrow(ax, 2.9, 7.8, 2.9, 7.2)

    draw_box(ax, 1.3, 6.0, 3.2, 1.0, 'Gallery', COLOR_OFFLINE, fontsize=16)
    draw_arrow(ax, 2.9, 6.0, 2.9, 5.2)

    draw_box(ax, 1.1, 4.0, 3.6, 1.0, 'Compute Stats\nμ, σ', COLOR_ADAPTIVE, fontsize=16, bold=True)

    # ========== 右上：在线阶段 ==========
    ax.text(12.1, 8.7, 'Online Recognition', ha='center', fontsize=16, weight='heavy', color=COLOR_TEXT)

    draw_box(ax, 10.5, 7.8, 3.2, 0.8, 'Query Face', COLOR_ONLINE, fontsize=16)
    draw_arrow(ax, 12.1, 7.8, 12.1, 7.2)

    draw_box(ax, 10.5, 6.6, 3.2, 0.6, 'Extract Feature', COLOR_ONLINE, fontsize=15)
    draw_arrow(ax, 12.1, 6.6, 12.1, 5.9)

    draw_box(ax, 10.5, 5.3, 3.2, 0.6, 'Best Match', COLOR_ONLINE, fontsize=15)
    draw_arrow(ax, 12.1, 5.3, 12.1, 4.6)

    draw_box(ax, 10.5, 4.0, 3.2, 0.6, 'Similarity', COLOR_ONLINE, fontsize=15)
    draw_arrow(ax, 12.1, 4.0, 12.1, 3.5)

    # ========== 中间分支点 ==========
    ax.add_patch(Circle((12.1, 3.3), 0.12, facecolor=COLOR_TEXT, zorder=3))
    draw_arrow(ax, 12.1, 3.3, 4.0, 3.3, style='-')
    draw_arrow(ax, 12.1, 3.3, 11.0, 3.3, style='-')

    # ========== 左下：Fixed Threshold ==========
    ax.text(3.5, 2.7, 'Fixed', ha='center', fontsize=16, weight='heavy', color=COLOR_TEXT)

    draw_diamond(ax, 2.2, 1.3, 3.6, 1.4, 'Score ≥ T?', COLOR_FIXED)
    draw_arrow(ax, 4.0, 3.3, 4.0, 2.7)

    # Fixed结果 - 先横后直的拐弯方式
    # Unknown: 从菱形左顶点出发，先水平向左，再垂直下降到框顶
    ax.plot([2.2, 0.7, 0.7], [2.0, 2.0, 0.9],
            color=COLOR_ARROW, linewidth=3.0, zorder=1)
    ax.arrow(0.7, 0.9, 0, -0.05, head_width=0.15, head_length=0.08,
             fc=COLOR_ARROW, ec=COLOR_ARROW, linewidth=2.5, zorder=1)
    draw_box(ax, 0.1, 0.15, 1.2, 0.7, 'Unknown', COLOR_REJECT, fontsize=14)
    ax.text(2.0, 2.1, 'No', fontsize=15, color=COLOR_ARROW, weight='heavy')

    # Known: 从菱形右顶点出发，先水平向右，再垂直下降到框顶
    ax.plot([5.8, 6.9, 6.9], [2.0, 2.0, 0.9],
            color=COLOR_ARROW, linewidth=3.0, zorder=1)
    ax.arrow(6.9, 0.9, 0, -0.05, head_width=0.15, head_length=0.08,
             fc=COLOR_ARROW, ec=COLOR_ARROW, linewidth=2.5, zorder=1)
    draw_box(ax, 6.3, 0.15, 1.2, 0.7, 'Known', COLOR_ACCEPT, fontsize=14)
    ax.text(5.7, 2.1, 'Yes', fontsize=15, color=COLOR_ARROW, weight='heavy')

    # ========== 右下：Adaptive Threshold ==========
    ax.text(11.7, 2.7, 'Adaptive', ha='center', fontsize=16, weight='heavy', color=COLOR_TEXT)

    # 从Offline Stats引线 - 连接到菱形上顶点
    draw_arrow(ax, 4.7, 4.5, 11.0, 2.7, style='->')
    ax.text(7.7, 3.8, 'T_i = μ - 2σ', fontsize=15, color=COLOR_ARROW, style='italic', weight='heavy',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none', alpha=0.95))

    draw_diamond(ax, 9.2, 1.3, 3.6, 1.4, 'Score ≥ T_i?', COLOR_ADAPTIVE)
    draw_arrow(ax, 11.0, 3.3, 11.0, 2.7)

    # Adaptive结果 - 先横后直的拐弯方式
    # Unknown: 从菱形左顶点出发，先水平向左，再垂直下降到框顶
    ax.plot([9.2, 8.3, 8.3], [2.0, 2.0, 0.9],
            color=COLOR_ARROW, linewidth=3.0, zorder=1)
    ax.arrow(8.3, 0.9, 0, -0.05, head_width=0.15, head_length=0.08,
             fc=COLOR_ARROW, ec=COLOR_ARROW, linewidth=2.5, zorder=1)
    draw_box(ax, 7.7, 0.15, 1.2, 0.7, 'Unknown', COLOR_REJECT, fontsize=14)
    ax.text(9.0, 2.1, 'No', fontsize=15, color=COLOR_ARROW, weight='heavy')

    # Known: 从菱形右顶点出发，先水平向右，再垂直下降到框顶
    ax.plot([12.8, 14.0, 14.0], [2.0, 2.0, 0.9],
            color=COLOR_ARROW, linewidth=3.0, zorder=1)
    ax.arrow(14.0, 0.9, 0, -0.05, head_width=0.15, head_length=0.08,
             fc=COLOR_ARROW, ec=COLOR_ARROW, linewidth=2.5, zorder=1)
    draw_box(ax, 13.4, 0.15, 1.2, 0.7, 'Known', COLOR_ACCEPT, fontsize=14)
    ax.text(12.7, 2.1, 'Yes', fontsize=15, color=COLOR_ARROW, weight='heavy')

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_OFFLINE, edgecolor=COLOR_TEXT, label='Offline'),
        mpatches.Patch(facecolor=COLOR_ONLINE, edgecolor=COLOR_TEXT, label='Online'),
        mpatches.Patch(facecolor=COLOR_FIXED, edgecolor=COLOR_TEXT, label='Fixed'),
        mpatches.Patch(facecolor=COLOR_ADAPTIVE, edgecolor=COLOR_TEXT, label='Adaptive'),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=13, frameon=True)

    plt.tight_layout()
    return fig


def main():
    """主函数"""
    print("=" * 80)
    print("🎨 绘制自适应开放集人脸识别流程图")
    print("=" * 80)

    # 生成流程图
    fig = plot_adaptive_pipeline()

    # 保存
    output_dir = Path(__file__).parent.parent / "thesis_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "adaptive_pipeline_flowchart.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 流程图已保存: {output_path}")

    # 同时保存PDF版本（适合论文）
    output_pdf = output_dir / "adaptive_pipeline_flowchart.pdf"
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF版本已保存: {output_pdf}")

    # 显示
    plt.show()

    print("\n" + "=" * 80)
    print("🎉 完成！流程图已生成")
    print("=" * 80)
    print(f"\n📁 输出文件:")
    print(f"   • PNG (高分辨率): {output_path.name}")
    print(f"   • PDF (矢量图): {output_pdf.name}")
    print(f"\n💡 提示: PDF版本适合直接插入论文")


if __name__ == "__main__":
    from pathlib import Path
    main()
