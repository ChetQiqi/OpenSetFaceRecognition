# ⚡ 快速参考 - 进度条功能速览

## 一行命令运行完整工作流

```bash
cd e:\RecognitionSystem
python scripts/prepare_open_set_data.py && python scripts/compare_with_prepared_data.py
```

---

## 📊 两个主要脚本

### 🔧 A. prepare_open_set_data.py
**目标**: 准备开放集评估数据

```
✓ 加载103个persons的5个特征 (500+ embeddings)
✓ 分割成80 known + 23 unknown
✓ Gallery: 240个embeddings (3特征×80人)
✓ Test Known: 160个samples (2特征×80人)
✓ Test Unknown: 115个samples (5特征×23人)

进度条:
━━━━━━━━━ 100%|██████████| 103/103 [00:10<00:00, 10.30 person/s]
```

**耗时**: 3-5分钟
**输出**: `benchmark/open_set_data/`

---

### 📈 B. compare_with_prepared_data.py
**目标**: 对比Fixed vs Adaptive性能

```
✓ 计算80个persons的自适应阈值
✓ 运行275个测试样本（160 known + 115 unknown）
✓ 比较6个关键指标
✓ 生成结果CSV、JSON、PNG

进度条:
计算Per-Identity统计━━━ 100%|██████████| 80/80 [00:08<00:00, 10.00 person/s]
Evaluating (Fixed)   ━ 100%|██████████| 275/275 [00:03<00:00, 91.67 it/s]
Evaluating (Adaptive)━ 100%|██████████| 275/275 [00:04<00:00, 68.75 it/s]
```

**耗时**: 5-10分钟
**输出**: `thesis_eval/`

---

## 🎯 关键结果指标

| 指标 | Fixed | Adaptive | 改进 | 用途 |
|------|-------|----------|------|------|
| **OSR** | 75.27% | 82.18% | +6.91% | 总体识别率 |
| **KCA** | 93.75% | 94.38% | +0.63% | 已知人准确率 |
| **UDR** | 48.70% | **65.22%** | **+16.52%** ⭐ | **陌生人检测率** |
| **Precision** | 74.07% | 82.14% | +8.07% | "拒绝"准确率 |
| **F1** | 58.73% | 72.97% | +14.24% | 综合指标 |
| **Latency** | 2.18ms | 2.31ms | +0.13ms | 处理时间 |

**核心发现**: UDR提升16.52% ⭐ —— 自适应方法的关键优势

---

## 📂 输出文件

### A. prepare_open_set_data.py 输出

```
benchmark/open_set_data/
├── gallery.npy              (240, 512)
├── gallery_labels.txt       240行
├── test_known.npy           (160, 512)
├── test_known_labels.txt    160行
├── test_unknown.npy         (115, 512)
├── test_unknown_labels.txt  115行
└── config.json
    {
        "num_known": 80,
        "num_unknown": 23,
        "gallery_size": 240,
        ...
    }
```

### B. compare_with_prepared_data.py 输出

```
thesis_eval/
├── fixed_vs_adaptive_results.csv       ← 表格数据
├── fixed_vs_adaptive_results.json      ← 完整数据
└── fixed_vs_adaptive_comparison.png    ← 论文用图表
```

---

## 🚀 三种使用方式

### 方式1: 完全自动化（推荐）
```bash
# 一键完成所有步骤
python scripts/prepare_open_set_data.py && python scripts/compare_with_prepared_data.py

# 查看结果
cat thesis_eval/fixed_vs_adaptive_results.csv
```

### 方式2: 分步执行（调试用）
```bash
# 第1步：准备数据 (可跳过，如果已有数据)
python scripts/prepare_open_set_data.py

# 第2步：评估对比（可单独运行多次）
python scripts/compare_with_prepared_data.py
```

### 方式3: 参数自定义（实验用）
```bash
# 不同已知/未知比例
python scripts/prepare_open_set_data.py --num-known 70

# 不同固定阈值
python scripts/compare_with_prepared_data.py --fixed-threshold 0.45 --output-dir thesis_eval_tau045
python scripts/compare_with_prepared_data.py --fixed-threshold 0.55 --output-dir thesis_eval_tau055
```

---

## 💾 论文引用模板

### 表格：
```latex
\begin{table}
\caption{自适应阈值性能对比}
\begin{tabular}{lccc}
\toprule
指标 & 固定 & 自适应 & 改进 \\
\midrule
OSR & 75.27\% & 82.18\% & +6.91\% \\
UDR & 48.70\% & 65.22\% & +16.52\% \\
\bottomrule
\end{tabular}
\end{table}
```

### 文字：
```
我们的自适应方法在开放集识别中取得显著改进：
OSR从75.27%提升至82.18%(+6.91%)；
最重要的是，未知检测率(UDR)从48.70%提升至65.22%(+16.52%)，
表明自适应阈值在识别陌生人方面具有明显优势。
```

---

## ⚙️ 参数参考

### prepare_open_set_data.py
```bash
python scripts/prepare_open_set_data.py \
    --input benchmark/YTF_100p.db              # 输入数据库
    --output-dir benchmark/open_set_data       # 输出目录
    --num-known 80                             # 已知persons数
    --gallery-size 3                           # 每人gallery特征数
    --test-size 2                              # 每人test特征数
    --seed 42                                  # 随机种子(可重复)
```

### compare_with_prepared_data.py
```bash
python scripts/compare_with_prepared_data.py \
    --data-dir benchmark/open_set_data         # 数据目录
    --fixed-threshold 0.5                      # 固定阈值
    --output-dir thesis_eval                   # 输出目录
```

---

## ✅ 检查清单

运行前：
- [ ] 数据库存在: `benchmark/YTF_100p.db`
- [ ] 环境正确: `cd e:\RecognitionSystem`
- [ ] 依赖安装: `pip install tqdm numpy pandas matplotlib scipy scikit-learn`

运行后：
- [ ] 看到进度条动画
- [ ] 完成时间合理 (5-15分钟)
- [ ] 输出文件存在且有内容
- [ ] PNG图表可以打开

---

## 🎓 下一步

1. **立即运行完整流程** (建议):
   ```bash
   python scripts/prepare_open_set_data.py && python scripts/compare_with_prepared_data.py
   ```

2. **查看详细结果**:
   ```bash
   cat thesis_eval/fixed_vs_adaptive_results.csv
   ```

3. **在论文中引用**:
   - 使用表格数据到论文章节 "实验结果"
   - 引用PNG图表到 "结果比较" 小节
   - 引用改进数据到 "创新贡献" 部分

4. **后续改进** (可选):
   - [ ] 使用更多特征数 (gallery_size=10, test_size=10)
   - [ ] 测试不同固定阈值 (0.45, 0.55, 0.60)
   - [ ] 导出数据进一步做ROC曲线分析

---

## 📞 遇到问题？

| 问题 | 解决方案 |
|------|--------|
| 进度条不显示 | 确保在终端直接运行，不要重定向输出 |
| ModuleNotFoundError | 脚本已自动处理，确保在项目目录运行 |
| FileNotFoundError | 检查 `benchmark/YTF_100p.db` 是否存在 |
| 内存错误 | 减少参数: `--gallery-size 2 --test-size 2` |

---

**🎉 准备好了吗? 立即运行吧!**
