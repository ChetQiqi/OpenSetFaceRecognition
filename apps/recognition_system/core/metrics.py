#!/usr/bin/env python3
"""
Comprehensive metrics module for face recognition evaluation.
Implements: Rank-1 Accuracy, TAR@FAR, FAR, ROC curve, F1-score, Precision, Recall, etc.
"""

import os
import csv
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.nn as nn

try:
    from sklearn.metrics import roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class MetricsResult:
    """Container for all evaluation metrics"""
    # Accuracy metrics
    rank1_accuracy: float

    # FAR/FRR metrics
    far: float
    frr: float
    tar_at_far_1e3: float  # TAR @ FAR = 1e-3
    tar_at_far_1e2: float  # TAR @ FAR = 1e-2
    tar_at_far_1e1: float  # TAR @ FAR = 1e-1

    # ROC metrics
    eer: float  # Equal Error Rate
    auc_score: float

    # Classification metrics
    f1: float
    precision: float
    recall: float

    # Performance metrics
    inference_time_mean: float
    inference_time_std: float
    inference_time_total: float

    # Model metrics
    model_size_mb: float
    total_parameters: int
    trainable_parameters: int

    # Metadata
    num_samples: int
    num_identities: int
    threshold: float = 0.5


class FaceRecognitionMetrics:
    """Calculate comprehensive face recognition metrics"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.inference_times: List[float] = []
        self.scores: List[float] = []
        self.predictions: List[int] = []
        self.ground_truth: List[int] = []
        self.rank1_correct: int = 0
        self.rank1_total: int = 0

    def add_inference_time(self, time_ms: float):
        """Record inference time in milliseconds"""
        self.inference_times.append(time_ms)

    def add_sample(self, score: float, pred_label: int, gt_label: int, is_correct: bool = None):
        """
        Add a single sample result

        Args:
            score: Similarity score between 0 and 1
            pred_label: Predicted person ID
            gt_label: Ground truth person ID
            is_correct: Whether top-1 prediction matches ground truth
        """
        self.scores.append(score)
        self.predictions.append(pred_label)
        self.ground_truth.append(gt_label)

        if is_correct is not None:
            if is_correct:
                self.rank1_correct += 1
            self.rank1_total += 1

    def compute_rank1_accuracy(self) -> float:
        """Compute Rank-1 accuracy"""
        if self.rank1_total == 0:
            return 0.0
        return self.rank1_correct / self.rank1_total

    def compute_far_frr(self) -> Tuple[float, float]:
        """
        Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR)

        For genuine pairs (same person):
            - FRR: percentage of genuine pairs rejected (score < threshold)

        For imposter pairs (different persons):
            - FAR: percentage of imposter pairs accepted (score >= threshold)
        """
        genuine_scores = []
        imposter_scores = []

        for score, gt_label, pred_label in zip(self.scores, self.ground_truth, self.predictions):
            if gt_label == pred_label:  # Genuine pair
                genuine_scores.append(score)
            else:  # Imposter pair
                imposter_scores.append(score)

        if len(genuine_scores) == 0 or len(imposter_scores) == 0:
            return 0.0, 0.0

        genuine_scores = np.array(genuine_scores)
        imposter_scores = np.array(imposter_scores)

        frr = np.sum(genuine_scores < self.threshold) / len(genuine_scores)
        far = np.sum(imposter_scores >= self.threshold) / len(imposter_scores)

        return far, frr

    def compute_tar_at_far(self, target_far: float = 1e-3) -> float:
        """
        Compute True Acceptance Rate (TAR) at specific False Acceptance Rate (FAR)

        Args:
            target_far: Target FAR level (default 1e-3)

        Returns:
            TAR at the specified FAR
        """
        genuine_scores = []
        imposter_scores = []

        for score, gt_label, pred_label in zip(self.scores, self.ground_truth, self.predictions):
            if gt_label == pred_label:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)

        if len(genuine_scores) == 0 or len(imposter_scores) == 0:
            return 0.0

        genuine_scores = np.array(genuine_scores)
        imposter_scores = np.array(imposter_scores)

        # Find threshold that gives us the target FAR
        # Sort imposter scores in descending order
        sorted_imposter = np.sort(imposter_scores)[::-1]
        num_imposters = len(imposter_scores)

        # Number of imposters that should be rejected
        num_to_reject = int(np.ceil(num_imposters * (1 - target_far)))

        if num_to_reject >= len(sorted_imposter):
            threshold = sorted_imposter[-1]
        else:
            threshold = sorted_imposter[num_to_reject]

        # Calculate TAR at this threshold
        tar = np.sum(genuine_scores >= threshold) / len(genuine_scores)
        return tar

    def compute_roc_metrics(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute ROC curve and related metrics

        Returns:
            fpr: False positive rates
            tpr: True positive rates
            eer: Equal Error Rate
            auc: Area Under Curve
        """
        binary_labels = np.array([1 if gt == pred else 0 for gt, pred in zip(self.ground_truth, self.predictions)])
        scores = np.array(self.scores)

        if len(set(binary_labels)) < 2:  # Need both classes
            return np.array([]), np.array([]), 0.0, 0.0

        # Use sklearn if available
        if SKLEARN_AVAILABLE:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, thresholds = roc_curve(binary_labels, scores)
            roc_auc = auc(fpr, tpr)
        else:
            # Simple fallback: compute metrics without ROC curve
            # Use correlation metric instead
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            roc_auc = 0.5

        # Compute EER (Equal Error Rate) - approximation
        if len(fpr) > 0 and len(tpr) > 0:
            fnr = 1 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = float(fpr[eer_idx])
        else:
            eer = 0.0

        return fpr, tpr, eer, roc_auc

    def compute_f1_precision_recall(self) -> Tuple[float, float, float]:
        """
        Compute F1-score, Precision, and Recall

        Returns:
            f1: F1-score
            precision: Precision
            recall: Recall
        """
        if len(self.scores) == 0:
            return 0.0, 0.0, 0.0

        # Binary classification: correct (1) or incorrect (0)
        binary_pred = np.array([1 if score >= self.threshold else 0 for score in self.scores])
        binary_labels = np.array([1 if gt == pred else 0 for gt, pred in zip(self.ground_truth, self.predictions)])

        if len(set(binary_labels)) < 2 or len(set(binary_pred)) < 2:
            return 0.0, 0.0, 0.0

        # Manual computation without sklearn
        tp = np.sum((binary_pred == 1) & (binary_labels == 1))
        fp = np.sum((binary_pred == 1) & (binary_labels == 0))
        fn = np.sum((binary_pred == 0) & (binary_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1, precision, recall

    def compute_inference_stats(self) -> Tuple[float, float, float]:
        """
        Compute inference time statistics

        Returns:
            mean_time_ms: Mean inference time in milliseconds
            std_time_ms: Standard deviation in milliseconds
            total_time_ms: Total inference time in milliseconds
        """
        if len(self.inference_times) == 0:
            return 0.0, 0.0, 0.0

        times = np.array(self.inference_times)
        return float(np.mean(times)), float(np.std(times)), float(np.sum(times))

    def compute_all_metrics(self) -> MetricsResult:
        """Compute all metrics and return as MetricsResult"""
        rank1_acc = self.compute_rank1_accuracy()
        far, frr = self.compute_far_frr()
        tar_1e3 = self.compute_tar_at_far(1e-3)
        tar_1e2 = self.compute_tar_at_far(1e-2)
        tar_1e1 = self.compute_tar_at_far(1e-1)
        fpr, tpr, eer, auc_score = self.compute_roc_metrics()
        f1, precision, recall = self.compute_f1_precision_recall()
        mean_time, std_time, total_time = self.compute_inference_stats()

        num_identities = len(set(self.ground_truth))

        return MetricsResult(
            rank1_accuracy=rank1_acc,
            far=far,
            frr=frr,
            tar_at_far_1e3=tar_1e3,
            tar_at_far_1e2=tar_1e2,
            tar_at_far_1e1=tar_1e1,
            eer=eer,
            auc_score=auc_score,
            f1=f1,
            precision=precision,
            recall=recall,
            inference_time_mean=mean_time,
            inference_time_std=std_time,
            inference_time_total=total_time,
            model_size_mb=0.0,  # Set separately
            total_parameters=0,  # Set separately
            trainable_parameters=0,  # Set separately
            num_samples=len(self.scores),
            num_identities=num_identities,
            threshold=self.threshold,
        )


class ModelMetrics:
    """Calculate model-related metrics"""

    @staticmethod
    def get_model_size(model_path: str, device: str = 'cpu') -> float:
        """
        Get model file size in MB

        Args:
            model_path: Path to the model file
            device: Device (cpu or cuda)

        Returns:
            Model size in MB
        """
        if not os.path.exists(model_path):
            return 0.0
        return os.path.getsize(model_path) / (1024 * 1024)

    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """
        Count total and trainable parameters in model

        Args:
            model: PyTorch model

        Returns:
            (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    @staticmethod
    def measure_inference_time(model: nn.Module, input_tensor: torch.Tensor,
                               num_runs: int = 100, warmup_runs: int = 10) -> float:
        """
        Measure average inference time in milliseconds

        Args:
            model: PyTorch model
            input_tensor: Sample input tensor
            num_runs: Number of runs for measurement
            warmup_runs: Number of warmup runs

        Returns:
            Average inference time in milliseconds
        """
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)

        # Measure
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)

        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return elapsed_time / num_runs


class MetricsVisualizer:
    """Visualize metrics"""

    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float,
                       output_path: Optional[str] = None) -> None:
        """Plot ROC curve"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠ matplotlib 未安装，跳过ROC曲线绘制")
            return

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_metrics_summary(metrics: 'MetricsResult', output_path: Optional[str] = None) -> None:
        """Plot summary of key metrics"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠ matplotlib 未安装，跳过指标总结绘制")
            return

        import matplotlib.pyplot as plt
        metrics_dict = {
            'Rank-1 Acc': metrics.rank1_accuracy,
            'TAR@FAR=1e-3': metrics.tar_at_far_1e3,
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'F1-Score': metrics.f1,
            'AUC': metrics.auc_score,
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics_dict.keys(), metrics_dict.values())

        # Color bars by value
        colors = plt.cm.RdYlGn(np.array(list(metrics_dict.values())))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_ylabel('Score')
        ax.set_title('Face Recognition Evaluation Metrics')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def load_results_csv(csv_path: str) -> Tuple[List[float], List[int], List[int]]:
    """
    Load results from CSV file
    Expected columns: score, prediction, ground_truth
    """
    scores, predictions, ground_truths = [], [], []

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['score']))
            predictions.append(int(row['prediction']))
            ground_truths.append(int(row['ground_truth']))

    return scores, predictions, ground_truths


def save_metrics_to_csv(metrics: MetricsResult, output_path: str) -> None:
    """Save metrics to CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=asdict(metrics).keys())
        writer.writeheader()
        writer.writerow(asdict(metrics))


def save_metrics_to_json(metrics: MetricsResult, output_path: str) -> None:
    """Save metrics to JSON file"""
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(metrics), f, indent=2)
