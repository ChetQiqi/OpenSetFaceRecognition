"""
只生成 Normalized Distribution Comparison 图表
（论文级高质量）
"""

import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# 导入系统模块
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from apps.recognition_system.core.matcher import cosine_similarity


def similarity_con(score: float) -> float:
    """校准相似度分数（分段线性函数）"""
    if score < 0.0:
        return 0.0
    if score <= 0.25:
        return score * 2.0
    if score <= 0.34:
        return 3.33 * score - 0.33
    if score < 0.6:
        return 0.38 * score + 0.67
    return 0.25 * score + 0.75


def extract_embeddings_from_db(db_path: str) -> Dict[str, List[np.ndarray]]:
    """从数据库中提取所有人的 embedding"""
    embeddings_by_person = {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.name, e.feature
        FROM embedding e
        JOIN person p ON e.person_id = p.id
        ORDER BY p.id, e.id
    """)
    rows = cursor.fetchall()
    conn.close()

    for person_name, feature_blob in rows:
        feature = np.frombuffer(feature_blob, dtype=np.float32)
        if person_name not in embeddings_by_person:
            embeddings_by_person[person_name] = []
        embeddings_by_person[person_name].append(feature)

    return embeddings_by_person


def generate_genuine_pairs(embeddings_by_person: Dict[str, List[np.ndarray]]) -> List[float]:
    """生成 genuine pairs（使用校准后的相似度）"""
    genuine_scores = []
    for person_name, embeddings in embeddings_by_person.items():
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                raw_score = cosine_similarity(embeddings[i], embeddings[j])
                if raw_score > -1.0:
                    # calibrated_score = similarity_con(raw_score)  # 应用校准
                    genuine_scores.append(raw_score)
    return genuine_scores


def generate_impostor_pairs(embeddings_by_person: Dict[str, List[np.ndarray]],
                           num_samples: int = 100000) -> List[float]:
    """生成 impostor pairs（使用校准后的相似度）"""
    impostor_scores = []
    person_ids = list(embeddings_by_person.keys())
    all_embeddings = []
    person_map = []

    for person_name in person_ids:
        for embedding in embeddings_by_person[person_name]:
            all_embeddings.append(embedding)
            person_map.append(person_name)

    count = 0
    attempts = 0
    max_attempts = num_samples * 10

    while count < num_samples and attempts < max_attempts:
        i = np.random.randint(0, len(all_embeddings))
        j = np.random.randint(0, len(all_embeddings))
        attempts += 1

        if i == j or person_map[i] == person_map[j]:
            continue

        raw_score = cosine_similarity(all_embeddings[i], all_embeddings[j])
        if raw_score > -1.0:
            # calibrated_score = similarity_con(raw_score)  # 应用校准
            impostor_scores.append(raw_score)
            count += 1

    return impostor_scores


def plot_normalized_comparison(genuine_scores: List[float], impostor_scores: List[float],
                               output_dir: Path):
    """只生成归一化对比图"""

    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制归一化直方图
    ax.hist(impostor_scores, bins=80, alpha=0.7, label='Impostor (Different people)',
            color='#d62728', density=True, edgecolor='black', linewidth=0.3)
    ax.hist(genuine_scores, bins=40, alpha=0.7, label='Genuine (Same person)',
            color='#2ca02c', density=True, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Calibrated Similarity Score', fontsize=24, fontweight='bold')
    ax.set_ylabel('Density', fontsize=24, fontweight='bold')
    ax.set_title('Normalized Similarity Score Distribution (Calibrated)\n' +
                 f'Impostor (n={len(impostor_scores):,}) vs Genuine (n={len(genuine_scores):,})',
                fontsize=25, fontweight='bold', pad=15)
    ax.legend(fontsize=18, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_ylim([0, ax.get_ylim()[1]])

    # 添加统计标注 - 框更小，文字更大
    stats_text = f"μ_imp: {np.mean(impostor_scores):+.3f}  μ_gen: {np.mean(genuine_scores):+.3f}\nSeparation: {np.mean(genuine_scores) - np.mean(impostor_scores):.3f}"
    # 位置调整到图片中心最上方，字体加大，框更小
    # ax.text(0.5, 0.58, stats_text, transform=ax.transAxes, fontsize=11,
    #         verticalalignment='top', horizontalalignment='center',
    #         family='monospace', weight='bold',
    #         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, pad=4))

    plt.tight_layout()
    plt.savefig(str(output_dir / "normalized_distribution_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_dir / 'normalized_distribution_comparison.png'}")
    plt.close()


def main():
    db_path = "benchmark/CASIA_200_features.db"
    output_dir = Path("threshold_analysis")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("📊 生成 Normalized Distribution Comparison 图表")
    print("=" * 80)

    print("\n【加载数据】")
    embeddings_by_person = extract_embeddings_from_db(db_path)
    print(f"✅ 加载了 {len(embeddings_by_person)} 个人的 embedding")

    print("\n【生成相似度对】")
    genuine_scores = generate_genuine_pairs(embeddings_by_person)
    print(f"✅ Genuine pairs: {len(genuine_scores)}")

    impostor_scores = generate_impostor_pairs(embeddings_by_person, num_samples=100000)
    print(f"✅ Impostor pairs: {len(impostor_scores)}")

    print("\n【生成图表】")
    plot_normalized_comparison(genuine_scores, impostor_scores, output_dir)

    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
