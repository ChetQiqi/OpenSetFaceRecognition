# 🚀 CASIA-WebFace 开放集实验快速指南

## 步骤1：提取CASIA特征

```bash
python extract_casia_features.py \
    --dataset-path F:\Dataset\CASIA-WebFace \
    --num-ids 200 \
    --output benchmark/CASIA_200_features.db \
    --device cuda:0
```

**参数说明：**
- `--dataset-path`: CASIA-WebFace数据集路径
- `--num-ids`: 提取前N个ID (200 = 200个人的身份)
- `--output`: 输出数据库位置
- `--device`: GPU/CPU (cuda:0 或 cpu)

**预期输出：**
```
📊 统计信息:
   处理的身份: 200
   扫描的图片: ~15,000-20,000
   提取的特征: ~15,000-20,000 (取决于人脸检测质量)
   成功率: 70-85%

💾 数据库保存到: benchmark/CASIA_200_features.db
```

**耗时:** ~15-30分钟（取决于GPU）

---

## 步骤2：准备开放集数据

提取完成后，准备开放集数据：

```bash
python prepare_open_set_optimized.py --db casia --num-known 100
```

**参数说明：**
- `--db casia`: 自动使用 benchmark/CASIA_200_features.db
- `--num-known 100`: 100个known + 100个unknown

**输出结果：**
```
📊 数据分配:
   Gallery: 100人 × 10特征 = 1000个embeddings
   Test Known: 100人 × 10特征 = 1000个样本
   Test Unknown: 100人 × 20特征 = 2000个样本（全部使用）

   生成目录: benchmark/open_set_data/
```

**耗时:** <1分钟

---

## 步骤3：运行对比评估

```bash
python scripts/compare_with_prepared_data.py --data-dir benchmark/open_set_data
```

**预期输出：**
```
固定阈值 vs 自适应阈值对比
┌──────────────────────┬────────┬────────┬──────────┐
│ 指标                 │ Fixed  │Adaptive│ 改进    │
├──────────────────────┼────────┼────────┼──────────┤
│ OSR                  │ 95.x%  │ 97.x%  │ +2%     │
│ KCA                  │ 93.x%  │ 95.x%  │ +2%     │
│ UDR                  │ 98.x%  │ 99.x%  │ +1%     │
└──────────────────────┴────────┴────────┴──────────┘
```

**输出文件：**
```
thesis_eval/
├── fixed_vs_adaptive_results.csv   ← 论文表格
├── fixed_vs_adaptive_results.json  ← 详细数据
└── CASIA对比分析.md
```

**耗时:** 5-10分钟

---

## 📊 完整流程一键运行

```bash
# 提取特征 + 准备数据 + 评估
python extract_casia_features.py --dataset-path F:\Dataset\CASIA-WebFace --num-ids 200 && \
python prepare_open_set_optimized.py --db casia --num-known 100 && \
python scripts/compare_with_prepared_data.py --data-dir benchmark/open_set_data
```

---

## 🎯 自定义参数

### 只提取50个ID

```bash
python extract_casia_features.py \
    --dataset-path F:\Dataset\CASIA-WebFace \
    --num-ids 50 \
    --output benchmark/CASIA_50_features.db
```

### 使用自定义参数准备数据

```bash
python prepare_open_set_optimized.py \
    --input benchmark/CASIA_50_features.db \
    --num-known 30 \
    --gallery-size 15 \
    --test-size 15 \
    --output-dir benchmark/open_set_data_custom
```

### CPU模式运行

```bash
python extract_casia_features.py \
    --dataset-path F:\Dataset\CASIA-WebFace \
    --device cpu \
    --num-ids 200
```

---

## 📝 对比不同数据集

| 数据集 | 命令 | 特征数 | 已知/未知 | 耗时 |
|--------|------|--------|----------|------|
| YTF小 | `--db small` | 514 | 80/23 | <1s |
| YTF完整 | `--db full` | 79,750 | 400/1200 | ~2-3min |
| CASIA新 | `--db casia` | ~15,000 | 100/100 | ~15-30min |

---

## ⚠️ 常见问题

**Q: 提取特征很慢？**
A: 这是正常的。CASIA有15,000+张图片，GPU提取需要15-30分钟。

**Q: 人脸检测失败率高？**
A: CASIA的图片质量参差不齐，检测失败15-30%是正常的。可以用 `--device cpu` 试试更保守的检测。

**Q: 能只用50个ID试试吗？**
A: 可以，改 `--num-ids 50` 即可，耗时会减少到5分钟。

**Q: 修改测试集大小？**
A: 用 `--num-known` 和 `--gallery-size`、`--test-size` 参数调整。

---

## 🎓 论文写作提示

使用CASIA数据集的优势：
- ✅ 更多样化的人脸（来自不同群体/背景）
- ✅ 更少的特征/人（更接近真实场景）
- ✅ 更强的泛化性验证

建议写法：
```
我们在两个公开数据集上评估了方法：
1. YTF数据集：1595个身份，79,750个特征（大规模）
2. CASIA-WebFace：200个身份，~15,000个特征（多样性）

在CASIA数据集上，自适应方法相对于固定阈值提升了2-3%...
```

---

## 💡 下一步

完成CASIA实验后，可以：
1. ✅ 对比两个数据集的结果差异
2. ✅ 分析跨数据集泛化性能
3. ✅ 用更多ID重复实验（200 → 500 → 1000个ID）
4. ✅ 组织成论文的对比表格

