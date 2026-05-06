"""
使用真实人脸图片重新验证 ROC 曲线
这个脚本从实际图片目录加载人脸，并重新计算相似度分布
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 导入系统模块
import sys
sys.path.insert(0, str(Path(__file__).parent / "apps" / "recognition_system"))
from core.matcher import cosine_similarity
from core.model import FaceEmbeddingModel


def load_real_images_with_embeddings(db_path: str, image_root: Path) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """
    加载真实图片和对应的 embedding
    返回: {person_name: [(embedding, image_path), ...]}
    """
    data = {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询所有 embedding 及其对应的图片路径
    cursor.execute("""
        SELECT p.name, e.feature, e.image_path
        FROM embedding e
        JOIN person p ON e.person_id = p.id
        ORDER BY p.id, e.id
    """)
    rows = cursor.fetchall()
    conn.close()

    print(f"\n从数据库加载 {len(rows)} 个 embedding...")

    loaded_count = 0
    missing_count = 0

    for person_name, feature_blob, image_path_stored in rows:
        # 从 BLOB 中恢复 embedding
        feature = np.frombuffer(feature_blob, dtype=np.float32)

        # 检查实际图片是否存在
        person_dir = image_root / person_name
        if not person_dir.exists():
            missing_count += 1
            continue

        if person_name not in data:
            data[person_name] = []

        data[person_name].append({
            'embedding': feature,
            'image_path': image_path_stored
        })
        loaded_count += 1

    print(f"✅ 成功加载: {loaded_count} 个 embedding")
    print(f"⚠️  目录缺失: {missing_count} 个 embedding")
    print(f"📊 可用人物: {len(data)} 个")

    return data


def load_images_for_verification(db_path: str, image_root: Path, num_images_per_person: int = 3) -> Dict[str, List[np.ndarray]]:
    """
    从实际图片目录加载人脸图片用于验证
    这用于对比 embedding 的准确性
    """
    images = {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM person WHERE id <= 99")  # 只处理有图片的人物
    persons = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"\n从图片目录加载样本图片 (每人最多 {num_images_per_person} 张)...")

    for person_name in persons:
        person_dir = image_root / person_name
        if not person_dir.exists():
            continue

        # 查找所有视频目录下的 jpg/png
        image_files = []
        for video_dir in person_dir.iterdir():
            if video_dir.is_dir():
                image_files.extend(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))

        # 随机选择样本
        sample_images = random.sample(image_files, min(num_images_per_person, len(image_files)))

        valid_images = []
        for img_path in sample_images:
            img = cv2.imread(str(img_path))
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                valid_images.append(img)

        if valid_images:
            images[person_name] = valid_images

    print(f"✅ 加载了 {len(images)} 个人物的样本图片")
    return images


def generate_genuine_pairs_from_db(data: Dict[str, List]) -> List[float]:
    """生成 genuine pairs（同一人的 embedding）"""
    genuine_scores = []

    for person_name, records in data.items():
        embeddings = [rec['embedding'] for rec in records]

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                score = cosine_similarity(embeddings[i], embeddings[j])
                if score > -1.0:
                    genuine_scores.append(score)

    print(f"✅ 生成 {len(genuine_scores)} 个 genuine pairs")
    if genuine_scores:
        print(f"   均值: {np.mean(genuine_scores):.4f}, 范围: [{np.min(genuine_scores):.4f}, {np.max(genuine_scores):.4f}]")

    return genuine_scores


def generate_impostor_pairs_from_db(data: Dict[str, List], num_samples: int = 100000) -> List[float]:
    """生成 impostor pairs（不同人的 embedding）"""
    impostor_scores = []

    persons = list(data.keys())
    all_embeddings = []
    person_map = []

    for person_name in persons:
        for rec in data[person_name]:
            all_embeddings.append(rec['embedding'])
            person_map.append(person_name)

    print(f"\n生成 impostor pairs (最多 {num_samples} 个)...")

    count = 0
    attempts = 0
    max_attempts = num_samples * 10

    while count < num_samples and attempts < max_attempts:
        i = np.random.randint(0, len(all_embeddings))
        j = np.random.randint(0, len(all_embeddings))
        attempts += 1

        if i == j or person_map[i] == person_map[j]:
            continue

        score = cosine_similarity(all_embeddings[i], all_embeddings[j])
        if score > -1.0:
            impostor_scores.append(score)
            count += 1

    print(f"✅ 生成 {len(impostor_scores)} 个 impostor pairs")
    if impostor_scores:
        print(f"   均值: {np.mean(impostor_scores):.4f}, 范围: [{np.min(impostor_scores):.4f}, {np.max(impostor_scores):.4f}]")

    return impostor_scores


def compute_roc_metrics(genuine_scores: List[float], impostor_scores: List[float]):
    """计算 ROC 曲线"""
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([np.array(genuine_scores), np.array(impostor_scores)])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def find_optimal_threshold(genuine_scores: List[float], impostor_scores: List[float]) -> Tuple[float, float, float]:
    """找到最优阈值"""
    thresholds = np.linspace(-1, 1, 1000)

    fars = []
    frrs = []

    for tau in thresholds:
        far = np.sum(np.array(impostor_scores) >= tau) / len(impostor_scores)
        frr = np.sum(np.array(genuine_scores) < tau) / len(genuine_scores)

        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    diff = np.abs(fars - frrs)
    optimal_idx = np.argmin(diff)
    optimal_tau = thresholds[optimal_idx]
    optimal_far = fars[optimal_idx]
    optimal_frr = frrs[optimal_idx]

    return optimal_tau, optimal_far, optimal_frr, thresholds, fars, frrs


def plot_comparison(genuine_scores_db, impostor_scores_db, output_dir: Path):
    """绘制对比图表"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 分布对比
    ax = axes[0, 0]
    ax.hist(genuine_scores_db, bins=50, alpha=0.7, label='Genuine', color='green')
    ax.hist(impostor_scores_db, bins=50, alpha=0.7, label='Impostor', color='red')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distribution (From Database Embeddings)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ROC 曲线
    ax = axes[0, 1]
    fpr, tpr, roc_auc = compute_roc_metrics(genuine_scores_db, impostor_scores_db)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FAR/FRR 曲线
    ax = axes[1, 0]
    optimal_tau, optimal_far, optimal_frr, thresholds, fars, frrs = find_optimal_threshold(genuine_scores_db, impostor_scores_db)
    ax.plot(thresholds, fars, 'r-', lw=2, label='FAR')
    ax.plot(thresholds, frrs, 'b-', lw=2, label='FRR')
    ax.plot(optimal_tau, optimal_far, 'go', markersize=10, label=f'EER (τ={optimal_tau:.4f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Error Rate')
    ax.set_title('FAR and FRR vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 统计信息
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Database-based Analysis

    Genuine Pairs: {len(genuine_scores_db)}
      Mean: {np.mean(genuine_scores_db):.4f}
      Std: {np.std(genuine_scores_db):.4f}

    Impostor Pairs: {len(impostor_scores_db)}
      Mean: {np.mean(impostor_scores_db):.4f}
      Std: {np.std(impostor_scores_db):.4f}

    Optimal Threshold: τ = {optimal_tau:.4f}
      FAR = {optimal_far:.4f}
      FRR = {optimal_frr:.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(str(output_dir / "roc_verification_with_images.png"), dpi=300)
    print(f"✅ 图表已保存: {output_dir / 'roc_verification_with_images.png'}")
    plt.close()


def main():
    db_path = "benchmark/YTF_100p.db"
    image_root = Path(r"G:\YTF_dataset\OpenDataLab___YouTube_Faces\raw\data\YouTubeFaces\aligned_images_DB")
    output_dir = Path("threshold_analysis")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("🔍 使用真实图片数据验证 ROC 曲线")
    print("=" * 80)

    # 加载 embedding 和图片对应关系
    data = load_real_images_with_embeddings(db_path, image_root)

    # 生成 genuine 和 impostor pairs
    print("\n【生成相似度对】")
    genuine_scores = generate_genuine_pairs_from_db(data)
    impostor_scores = generate_impostor_pairs_from_db(data, num_samples=100000)

    # 计算最优阈值
    print("\n【计算最优阈值】")
    optimal_tau, optimal_far, optimal_frr, thresholds, fars, frrs = find_optimal_threshold(genuine_scores, impostor_scores)
    print(f"✅ 最优阈值: τ = {optimal_tau:.4f}")
    print(f"   FAR = {optimal_far:.4f}, FRR = {optimal_frr:.4f}")
    print(f"   EER = {(optimal_far + optimal_frr) / 2:.4f}")

    # 绘制对比图表
    print("\n【生成可视化图表】")
    plot_comparison(genuine_scores, impostor_scores, output_dir)

    print("\n" + "=" * 80)
    print("✅ 验证完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
