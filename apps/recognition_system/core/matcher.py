from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np


@dataclass
class MatchResult:
    name: str
    score: float
    support: int


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return -1.0
    return float(np.dot(a, b) / denom)

# def SimilarityCon(score):
#     if score < 0.0:
#         return 0.0
#     elif score <= 0.25:
#         return score * 2.0
#     elif score <= 0.34:
#         return 3.33 * score - 0.33
#     elif score < 0.6:
#         return 0.38 * score + 0.67
#     else:
#         return 0.25 * score + 0.75

GalleryInput = Union[List[Tuple[str, np.ndarray]], Dict[str, List[np.ndarray]]]


def _group_gallery(gallery: GalleryInput) -> Dict[str, List[np.ndarray]]:
    if isinstance(gallery, dict):
        return gallery

    grouped: Dict[str, List[np.ndarray]] = {}
    for name, feature in gallery:
        grouped.setdefault(name, []).append(feature)
    return grouped


def _reduce_scores(scores: Iterable[float], reduce: str, topk: int) -> float:
    values = sorted(scores, reverse=True)
    if not values:
        return -1.0
    if reduce == "best":
        return float(values[0])
    if reduce == "mean":
        return float(np.mean(values))
    if reduce == "topk_mean":
        limit = max(1, min(topk, len(values)))
        return float(np.mean(values[:limit]))
    raise ValueError(f"Unsupported reduce mode: {reduce}")


def find_best_match(
    query: np.ndarray,
    gallery: GalleryInput,
    reduce: str = "topk_mean",
    topk: int = 3,
) -> MatchResult:
    grouped = _group_gallery(gallery)

    best_name = "Unknown"
    best_score = -1.0
    best_support = 0
    for name, features in grouped.items():
        scores = [cosine_similarity(query, feature) for feature in features]
        score = _reduce_scores(scores, reduce=reduce, topk=topk)
        if score > best_score:
            best_name = name
            best_score = score
            best_support = len(features)

    return MatchResult(name=best_name, score=best_score, support=best_support)

