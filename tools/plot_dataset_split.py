"""
绘制数据集划分示意图
用于论文说明Gallery和Test的划分策略
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_dataset_split():
    """绘制数据集划分示意图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(7, 7.5, 'Open-Set Dataset Partitioning Strategy',
            ha='center', fontsize=18, weight='bold')
    ax.text(7, 7.1, 'CASIA-WebFace 200个身份的划分方式',
            ha='center', fontsize=14, color='#555')

    # ========== 第一层：200个身份 ==========
    total_box = mpatches.FancyBboxPatch(
        (0.5, 5.5), 13, 1.2,
        boxstyle="round,pad=0.05",
        edgecolor='#333', facecolor='#E8E8E8',
        linewidth=2.5
    )
    ax.add_patch(total_box)
    ax.text(7, 6.1, '200 Identities (CASIA-WebFace subset)',
            ha='center', fontsize=14, weight='bold')

    # 箭头向下
    ax.annotate('', xy=(4, 5.5), xytext=(4, 5.0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))
    ax.annotate('', xy=(10, 5.5), xytext=(10, 5.0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))

    # ========== 第二层：Known vs Unknown ==========
    # Known box
    known_box = mpatches.FancyBboxPatch(
        (0.5, 3.8), 6, 1.0,
        boxstyle="round,pad=0.05",
        edgecolor='#2E7D32', facecolor='#C8E6C9',
        linewidth=2.5
    )
    ax.add_patch(known_box)
    ax.text(3.5, 4.5, 'Known Persons', ha='center', fontsize=13, weight='bold', color='#1B5E20')
    ax.text(3.5, 4.1, '89 identities (45%)', ha='center', fontsize=11, color='#2E7D32')

    # Unknown box
    unknown_box = mpatches.FancyBboxPatch(
        (7.5, 3.8), 6, 1.0,
        boxstyle="round,pad=0.05",
        edgecolor='#C62828', facecolor='#FFCDD2',
        linewidth=2.5
    )
    ax.add_patch(unknown_box)
    ax.text(10.5, 4.5, 'Unknown Persons', ha='center', fontsize=13, weight='bold', color='#B71C1C')
    ax.text(10.5, 4.1, '100 identities (55%)', ha='center', fontsize=11, color='#C62828')

    # 箭头向下
    ax.annotate('', xy=(3.5, 3.8), xytext=(3.5, 3.3),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#2E7D32'))
    ax.annotate('', xy=(10.5, 3.8), xytext=(10.5, 3.3),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#C62828'))

    # ========== 第三层：Known的详细划分 ==========
    # 时间序列示意
    ax.text(3.5, 3.0, 'Each Known Person (20 embeddings)',
            ha='center', fontsize=12, weight='bold', color='#1B5E20')

    # 绘制embedding序列
    emb_y = 2.2
    emb_width = 0.45
    emb_height = 0.5

    # Gallery部分 (1-10)
    for i in range(10):
        box = mpatches.Rectangle(
            (0.8 + i * emb_width, emb_y),
            emb_width * 0.9, emb_height,
            facecolor='#81C784', edgecolor='#2E7D32', linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(0.8 + i * emb_width + emb_width * 0.45, emb_y + emb_height/2,
                f'{i+1}', ha='center', va='center', fontsize=8, weight='bold')

    # Test Known部分 (11-20)
    for i in range(10, 20):
        box = mpatches.Rectangle(
            (0.8 + i * emb_width, emb_y),
            emb_width * 0.9, emb_height,
            facecolor='#4CAF50', edgecolor='#1B5E20', linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(0.8 + i * emb_width + emb_width * 0.45, emb_y + emb_height/2,
                f'{i+1}', ha='center', va='center', fontsize=8, weight='bold', color='white')

    # 标注
    ax.text(3.1, 1.8, 'Gallery (10)', ha='center', fontsize=11, weight='bold', color='#2E7D32')
    ax.text(3.1, 1.5, '89×10 = 890', ha='center', fontsize=9, color='#2E7D32')

    ax.text(5.6, 1.8, 'Test Known (10)', ha='center', fontsize=11, weight='bold', color='#1B5E20')
    ax.text(5.6, 1.5, '89×10 = 890', ha='center', fontsize=9, color='#1B5E20')

    # 大括号
    ax.plot([0.8, 0.8, 5.3, 5.3], [emb_y - 0.15, emb_y - 0.25, emb_y - 0.25, emb_y - 0.15],
            'k-', linewidth=2)
    ax.plot([5.3, 5.3, 9.8, 9.8], [emb_y - 0.15, emb_y - 0.25, emb_y - 0.25, emb_y - 0.15],
            'k-', linewidth=2)

    # ========== 第三层：Unknown的说明 ==========
    ax.text(10.5, 3.0, 'Each Unknown Person',
            ha='center', fontsize=12, weight='bold', color='#B71C1C')

    unknown_box_detail = mpatches.FancyBboxPatch(
        (8.5, 1.8), 4, 1.0,
        boxstyle="round,pad=0.05",
        edgecolor='#C62828', facecolor='#EF9A9A',
        linewidth=2
    )
    ax.add_patch(unknown_box_detail)
    ax.text(10.5, 2.5, 'Test Unknown (All)', ha='center', fontsize=11, weight='bold', color='#B71C1C')
    ax.text(10.5, 2.2, '100 persons × ~123 avg', ha='center', fontsize=9, color='#C62828')
    ax.text(10.5, 1.95, '= 12,345 total', ha='center', fontsize=9, color='#C62828')

    # ========== 底部统计说明 ==========
    stats_box = mpatches.FancyBboxPatch(
        (0.5, 0.2), 13, 0.9,
        boxstyle="round,pad=0.05",
        edgecolor='#666', facecolor='#FFF9C4',
        linewidth=2
    )
    ax.add_patch(stats_box)

    stats_text = (
        'Dataset Summary:\n'
        '• Gallery: 890 samples (10/person) → Compute adaptive thresholds (μ-2σ)\n'
        '• Test Known: 890 samples (10/person) → Evaluate KCA\n'
        '• Test Unknown: 12,345 samples (all) → Evaluate UDR'
    )
    ax.text(7, 0.65, stats_text, ha='center', va='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0))

    plt.tight_layout()
    return fig


def main():
    """主函数"""
    print("=" * 80)
    print("绘制数据集划分示意图")
    print("=" * 80)

    # 生成图表
    fig = plot_dataset_split()

    # 保存
    output_dir = Path("thesis_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_png = output_dir / "dataset_split_diagram.png"
    output_pdf = output_dir / "dataset_split_diagram.pdf"

    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')

    print(f"\n✅ 图表已保存:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")

    plt.show()

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
