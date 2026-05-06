"""
Adaptive Open-set Recognition Framework - Core Module

This module implements per-identity adaptive thresholds for face recognition,
enabling better handling of identities with varying intra-class variance and
improved unknown detection capabilities.

Strategy: Gaussian-based adaptive thresholds (μ - 2σ)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class IdentityStatistics:
    """
    Per-identity statistical profile for adaptive thresholding.

    Attributes:
        identity_name: Person's name
        person_id: Database ID for the person
        mean_genuine_score: Mean cosine similarity of genuine pairs
        std_genuine_score: Standard deviation of genuine scores
        min_genuine_score: Minimum observed genuine score
        max_genuine_score: Maximum observed genuine score
        sample_count: Number of genuine pairs used for statistics
        adaptive_threshold: Computed threshold (μ - k*σ, k=2.0)
        last_updated: Unix timestamp of last update
        recent_scores: Sliding window of recent genuine scores (for online learning)
    """

    identity_name: str
    person_id: int
    mean_genuine_score: float = 0.0
    std_genuine_score: float = 0.0
    min_genuine_score: float = 0.0
    max_genuine_score: float = 1.0
    sample_count: int = 0
    adaptive_threshold: float = 0.5
    last_updated: float = field(default_factory=time.time)
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class AdaptiveDecision:
    """
    Enhanced recognition result with adaptive decision components.

    Attributes:
        identity_name: Matched identity (or "Unknown" if rejected)
        similarity_score: Raw cosine similarity score
        calibrated_score: Score after calibration function
        adaptive_threshold: Per-identity threshold used for decision
        accepted: Whether the match was accepted (True) or rejected (False)
        decision_reason: Reason for decision ("accepted", "below_threshold", "outlier", "uncertainty")
        z_score: Statistical outlier metric (|score - μ| / σ)
        distance_ratio: Ratio of 1st to 2nd best score (uncertainty metric)
    """

    identity_name: str
    similarity_score: float
    calibrated_score: float
    adaptive_threshold: float
    accepted: bool
    decision_reason: str
    z_score: float
    distance_ratio: float


def compute_adaptive_thresholds(
    db_path: str, k: float = 2.0, min_samples: int = 2
) -> Dict[str, IdentityStatistics]:
    """
    Compute per-identity adaptive thresholds from existing database.

    For each person in the database:
    1. Load all embeddings
    2. Compute all genuine pairs (NxN pairwise similarities)
    3. Calculate mean, std, min, max of genuine scores
    4. Compute adaptive threshold: threshold = mean - k*std
    5. Clamp threshold to safe range

    Args:
        db_path: Path to feature database
        k: Number of standard deviations (default 2.0 for ~95% coverage)
        min_samples: Minimum embeddings required per person (default 2)

    Returns:
        Dictionary mapping person_name to IdentityStatistics
    """
    from .feature_db import FeatureDB
    from .matcher import cosine_similarity

    identity_stats = {}

    with FeatureDB(db_path) as db:
        # Get all persons
        cur = db.conn.cursor()
        cur.execute("SELECT id, name FROM person")
        persons = cur.fetchall()

        for person_id, person_name in persons:
            # Load all embeddings for this person
            cur.execute(
                "SELECT feature FROM embedding WHERE person_id = ?", (person_id,)
            )
            rows = cur.fetchall()

            if len(rows) < min_samples:
                # Cold start: use global fallback
                print(
                    f"[Adaptive] {person_name}: {len(rows)} samples (< {min_samples}), using fallback"
                )
                identity_stats[person_name] = IdentityStatistics(
                    identity_name=person_name,
                    person_id=person_id,
                    mean_genuine_score=0.85,  # Default for calibrated space
                    std_genuine_score=0.05,
                    adaptive_threshold=0.75,  # Conservative fallback for calibrated
                    sample_count=0,
                )
                continue

            # Convert BLOBs to numpy arrays
            from .feature_db import _from_blob
            from .operations import similarity_con

            embeddings = [_from_blob(row[0]) for row in rows]

            # Compute all genuine pairs (NxN similarities)
            genuine_scores = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    raw_score = cosine_similarity(embeddings[i], embeddings[j])
                    # ✨ Apply calibration to be consistent with recognition
                    calibrated = similarity_con(raw_score)
                    genuine_scores.append(calibrated)

            if len(genuine_scores) == 0:
                # Only 1 embedding, no pairs
                identity_stats[person_name] = IdentityStatistics(
                    identity_name=person_name,
                    person_id=person_id,
                    mean_genuine_score=0.85,  # Default for calibrated space
                    std_genuine_score=0.05,
                    adaptive_threshold=0.75,
                    sample_count=0,
                )
                continue

            # Compute statistics
            mean_score = float(np.mean(genuine_scores))
            std_score = float(np.std(genuine_scores))
            min_score = float(np.min(genuine_scores))
            max_score = float(np.max(genuine_scores))

            # Compute adaptive threshold: μ - k*σ
            threshold = mean_score - k * std_score

            # Clamp to safe range for calibrated similarity:
            # Calibrated scores typically range 0.5-1.0 for genuine pairs
            # - Min threshold should be at least 0.5 (safe lower bound)
            # - Max threshold is the mean score
            threshold = max(threshold, 0.5)
            threshold = min(threshold, mean_score)

            identity_stats[person_name] = IdentityStatistics(
                identity_name=person_name,
                person_id=person_id,
                mean_genuine_score=mean_score,
                std_genuine_score=std_score,
                min_genuine_score=min_score,
                max_genuine_score=max_score,
                sample_count=len(genuine_scores),
                adaptive_threshold=threshold,
                last_updated=time.time(),
                recent_scores=deque(genuine_scores[-50:], maxlen=50),
            )

            print(
                f"[Adaptive] {person_name}: μ={mean_score:.3f}, σ={std_score:.3f}, "
                f"threshold={threshold:.3f} (n={len(genuine_scores)} pairs)"
            )

    return identity_stats


def adaptive_recognize_face(
    query_embedding: np.ndarray,
    gallery: List,
    identity_stats: Dict[str, IdentityStatistics],
    best_match_name: str,
    best_match_score: float,
    all_scores: List[float],
    calibrated_score: float,
    global_fallback: float = 0.45,
    z_score_threshold: float = 3.0,
    distance_ratio_threshold: float = 1.2,
) -> AdaptiveDecision:
    """
    Perform adaptive recognition with multi-layer decision logic.

    Decision Layers:
    1. Statistical Outlier Detection: Reject if z-score > 3.0
    2. Adaptive Threshold: Reject if score < per-identity threshold
    3. Uncertainty Gate: Reject if match is ambiguous (distance_ratio < 1.2)

    Args:
        query_embedding: Query face embedding
        gallery: Gallery of known embeddings
        identity_stats: Dictionary of per-identity statistics
        best_match_name: Best matching identity (from matcher)
        best_match_score: Raw similarity score of best match
        all_scores: All similarity scores (sorted descending)
        calibrated_score: Calibrated similarity score
        global_fallback: Fallback threshold for unknown identities
        z_score_threshold: Threshold for outlier detection (default 3.0)
        distance_ratio_threshold: Threshold for ambiguous matches (default 1.2)

    Returns:
        AdaptiveDecision with full diagnostic information
    """
    # Get adaptive threshold for matched identity
    if best_match_name in identity_stats:
        stats = identity_stats[best_match_name]
        adaptive_threshold = stats.adaptive_threshold
        identity_mean = stats.mean_genuine_score
        identity_std = stats.std_genuine_score
    else:
        # Cold start: use global fallback
        adaptive_threshold = global_fallback
        identity_mean = 0.5
        identity_std = 0.1

    # Compute uncertainty metrics
    # 1. Z-score: how many standard deviations from mean?
    if identity_std > 1e-6:
        z_score = abs(best_match_score - identity_mean) / identity_std
    else:
        z_score = 0.0

    # 2. Distance ratio: how confident is the best match vs second best?
    if len(all_scores) >= 2 and all_scores[1] > 1e-6:
        distance_ratio = all_scores[0] / all_scores[1]
    else:
        distance_ratio = 10.0  # Very confident if only one match

    # Multi-layer decision logic
    accepted = False
    decision_reason = "unknown"

    # Layer 1: Statistical outlier check
    if z_score > z_score_threshold:
        accepted = False
        decision_reason = "outlier"
    # Layer 2: Adaptive threshold check
    elif calibrated_score < adaptive_threshold:
        accepted = False
        decision_reason = "below_threshold"
    # Layer 3: Uncertainty gate (ambiguous match)
    elif distance_ratio < distance_ratio_threshold:
        accepted = False
        decision_reason = "uncertainty"
    # All layers passed
    else:
        accepted = True
        decision_reason = "accepted"

    return AdaptiveDecision(
        identity_name=best_match_name if accepted else "Unknown",
        similarity_score=best_match_score,
        calibrated_score=calibrated_score,
        adaptive_threshold=adaptive_threshold,
        accepted=accepted,
        decision_reason=decision_reason,
        z_score=z_score,
        distance_ratio=distance_ratio,
    )


def update_identity_statistics(
    stats: IdentityStatistics,
    new_score: float,
    learning_rate: float = 0.1,
    k: float = 2.0,
) -> IdentityStatistics:
    """
    Update identity statistics with new genuine score (online learning).

    Uses Exponential Moving Average (EMA) for mean and sliding window for std.

    Args:
        stats: Current identity statistics
        new_score: New genuine similarity score
        learning_rate: EMA learning rate (0.0-1.0, default 0.1)
        k: Number of standard deviations for threshold (default 2.0)

    Returns:
        Updated IdentityStatistics
    """
    # Update sliding window
    stats.recent_scores.append(new_score)

    # Update mean with EMA
    stats.mean_genuine_score = (
        1 - learning_rate
    ) * stats.mean_genuine_score + learning_rate * new_score

    # Update std from recent window
    if len(stats.recent_scores) >= 2:
        recent_std = float(np.std(list(stats.recent_scores)))
        stats.std_genuine_score = recent_std
    else:
        # Not enough data, keep old std
        pass

    # Recompute adaptive threshold
    threshold = stats.mean_genuine_score - k * stats.std_genuine_score

    # Clamp to safe range for calibrated similarity:
    # Calibrated scores typically range 0.5-1.0 for genuine pairs
    threshold = max(threshold, 0.5)
    threshold = min(threshold, stats.mean_genuine_score)

    stats.adaptive_threshold = threshold

    # Update metadata
    stats.sample_count += 1
    stats.last_updated = time.time()

    return stats


def get_all_scores_for_query(
    query_embedding: np.ndarray, gallery: List, reduce: str = "topk_mean", topk: int = 3
) -> Tuple[List[str], List[float]]:
    """
    Helper function to get all identity scores for uncertainty computation.

    Args:
        query_embedding: Query face embedding
        gallery: Gallery of known embeddings
        reduce: Score reduction method
        topk: Top-k for reduction

    Returns:
        Tuple of (identity_names, scores) sorted by score descending
    """
    from .matcher import _group_gallery, _reduce_scores, cosine_similarity

    grouped = _group_gallery(gallery)

    identity_scores = []
    for name, features in grouped.items():
        scores = [cosine_similarity(query_embedding, feature) for feature in features]
        score = _reduce_scores(scores, reduce=reduce, topk=topk)
        identity_scores.append((name, score))

    # Sort by score descending
    identity_scores.sort(key=lambda x: x[1], reverse=True)

    names = [name for name, _ in identity_scores]
    scores = [score for _, score in identity_scores]

    return names, scores
