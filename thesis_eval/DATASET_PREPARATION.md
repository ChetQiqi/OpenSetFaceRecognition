# Dataset Preparation for Adaptive Open-Set Face Recognition

## 1. Dataset Selection

### 1.1 CASIA-WebFace Dataset

We adopt the **CASIA-WebFace** dataset [1] for our open-set recognition experiments. This dataset contains face images of 10,575 identities collected from the web, with significant variations in pose, expression, illumination, and age. For computational efficiency and controlled evaluation, we select a subset of **200 identities** with rich intra-class variations.

**Selection Criteria:**
- Each identity has at least 50 face images
- Diverse in demographics (age, gender, ethnicity)
- High-quality facial features suitable for deep learning embeddings

### 1.2 Feature Extraction

We employ a pre-trained **iResNet50** model to extract 512-dimensional L2-normalized embeddings for each face image. The model is trained on MS-Celeb-1M dataset with ArcFace loss, achieving state-of-the-art recognition performance.

**Preprocessing Pipeline:**
1. Face detection using **MTCNN** (Multi-task Cascaded Convolutional Networks)
2. Face alignment to canonical coordinates (112×112 pixels)
3. Feature extraction using iResNet50 backbone
4. L2 normalization of embedding vectors

---

## 2. Open-Set Protocol

To simulate realistic open-set scenarios, we split the 200 identities into **known persons** and **unknown persons**:

| Category | Count | Purpose |
|----------|-------|---------|
| **Known Persons** | 89 (45%) | Registered in the system gallery |
| **Unknown Persons** | 100 (55%) | Simulate strangers in real-world |

**Rationale:** The 45:55 ratio reflects practical deployments where the number of potential unknown individuals slightly exceeds registered users. Random seed (42) is fixed to ensure reproducibility.

---

## 3. Data Partitioning Strategy

For each known person, we further partition their embeddings into **gallery set** and **test set**:

### 3.1 Gallery Construction

- **Purpose:** Build the reference database for recognition
- **Sampling:** First 10 embeddings per person
- **Total Size:** 89 persons × 10 = **890 gallery samples**

**Design Consideration:** Using 10 samples per person provides:
- Sufficient intra-class variability: C(10,2)=45 genuine pairs for robust adaptive threshold estimation (μ, σ)
- Computational efficiency during matching

### 3.2 Test Set Construction

We create three test subsets to evaluate different aspects of performance:

| Subset | Size | Source | Purpose |
|--------|------|--------|---------|
| **Test Known** | 890 | Known persons (10 per person) | Evaluate Known Class Accuracy (KCA) |
| **Test Unknown** | 12,345 | Unknown persons (all samples) | Evaluate Unknown Detection Rate (UDR) |

**Sampling Strategy:**
- Known persons: Next 10 embeddings after gallery samples (embeddings 11-20)
- Unknown persons: All available embeddings to maximize test coverage (avg. 123 per person)

---

## 4. Dataset Statistics

After preparation, our experimental dataset contains:

```
Total Identities: 189 (200 identities with 11 filtered due to insufficient samples)
├── Known Persons: 89
│   ├── Gallery: 890 embeddings (10 per person)
│   └── Test: 890 embeddings (10 per person)
└── Unknown Persons: 100
    └── Test: 12,345 embeddings (avg. 123 per person)
```

**Embedding Distribution:**
- Feature dimension: 512
- Genuine pair similarity: μ = 0.9537 ± 0.1184
- Impostor pair similarity: μ = 0.0623 ± 0.0867
- Separation margin: 0.89 (excellent discriminability)

---

## 5. Evaluation Protocol

We evaluate system performance using the following metrics:

### 5.1 Known Person Recognition
- **Known Class Accuracy (KCA):** Percentage of test known samples correctly identified
- Metric: `KCA = Correct Known / Total Known × 100%`

### 5.2 Unknown Person Detection
- **Unknown Detection Rate (UDR):** Percentage of test unknown samples correctly rejected
- Metric: `UDR = Correct Unknown / Total Unknown × 100%`

### 5.3 Overall Performance
- **Open-Set Recognition Rate (OSR):** Combined accuracy across both known and unknown
- Metric: `OSR = (Correct Known + Correct Unknown) / Total Samples × 100%`

### 5.4 Additional Metrics
- **Precision (Unknown):** Ratio of true unknowns among all rejected samples
- **F1-Score (Unknown):** Harmonic mean of precision and recall for unknown detection
- **Average Latency:** Mean inference time per query (milliseconds)

---

## 6. Threshold Analysis

To establish baseline thresholds for our adaptive method, we perform threshold analysis on the gallery set:

### 6.1 Score Calibration

We apply a piecewise linear calibration function `similarity_con()` to raw cosine similarity scores:

```
calibrated_score = similarity_con(cosine_similarity)
```

This calibration enhances discrimination in the high-similarity region (>0.6), which is critical for separating genuine matches from high-quality impostor attacks.

### 6.2 Optimal Threshold Selection

Using 1,028 genuine pairs and 100,000 impostor pairs:
- **Equal Error Rate (EER) point:** τ = 0.3073 (FAR = FRR = 1.75%)
- **Conservative threshold (deployed):** τ = 0.45 (FAR ≈ 5%, FRR ≈ 2%)

**Deployment Strategy:** We adopt τ = 0.45 as the fixed threshold baseline, prioritizing low false acceptance rate over false rejection rate, which is critical for security-sensitive applications.

For the adaptive method:
- **Lower bound:** 0.4 (ensures all adaptive thresholds are reasonable)
- **Upper bound:** Unconstrained (limited by μ - 2σ formula)

---

## 7. Reproducibility

All dataset preparation scripts are provided with the following entry points:

```bash
# Extract CASIA features (200 identities)
python scripts/extract_casia_features.py --num-ids 200

# Prepare open-set data (89 known / 100 unknown)
# gallery_size=10, test_size=10 per known person
python scripts/prepare_open_set_optimized.py \
    --db casia \
    --num-known 89 \
    --gallery-size 10 \
    --test-size 10

# Run evaluation
python scripts/compare_with_prepared_data.py \
    --data-dir benchmark/open_set_data_casia200
```

**Random Seed:** All experiments use fixed seed (42) for reproducible splits.

**Note:** Out of 200 identities, 11 were filtered due to having fewer than 20 embeddings (gallery_size + test_size), resulting in 189 valid identities.

---

## References

[1] Yi, D., Lei, Z., Liao, S., & Li, S. Z. (2014). Learning face representation from scratch. arXiv preprint arXiv:1411.7923.

---

## Appendix: Data Distribution Visualization

![Score Distribution](../threshold_analysis/normalized_distribution_comparison.png)

**Figure A1:** Calibrated similarity score distributions for genuine pairs (same person, green) and impostor pairs (different people, red). The distributions show excellent separation, validating our feature extraction quality.

![FAR/FRR Curves](../threshold_analysis/far_frr_curves.png)

**Figure A2:** False Acceptance Rate (FAR) and False Rejection Rate (FRR) as functions of threshold. The Equal Error Rate (EER) point at τ=0.3073 indicates system discriminability. We adopt τ=0.45 for conservative deployment.
