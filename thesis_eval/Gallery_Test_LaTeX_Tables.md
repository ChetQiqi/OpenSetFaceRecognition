# Gallery与测试集构建的LaTeX表格

## 清晰版本 - 每个Known Person的划分

```latex
\begin{table}[htbp]
\centering
\caption{Gallery与Test Known集的构建策略}
\label{tab:gallery_test_split}
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Embedding ID} & \textbf{1-10} & \textbf{11-20} & \textbf{总计} \\
\hline
\textbf{用途} & Gallery (注册库) & Test Known (已知人测试) & - \\
\hline
\textbf{每人数量} & 10条 & 10条 & 20条 \\
\hline
\textbf{总人数} & 89人 & 89人 & 89人 \\
\hline
\textbf{集合大小} & 890条 & 890条 & 1,780条 \\
\hline
\textbf{用途说明} & 构建识别参考库 & 评估已知类准确率 & - \\
& 计算自适应阈值 & (KCA) & \\
& (μ-2σ) & & \\
\hline
\end{tabular}
\end{table}
```

## 详细版本 - 包含统计信息

```latex
\begin{table}[htbp]
\centering
\caption{开放集数据集的详细划分}
\label{tab:dataset_partition}
\begin{tabular}{|l|r|r|r|}
\hline
\textbf{数据集} & \textbf{身份数} & \textbf{样本数} & \textbf{用途} \\
\hline
\multicolumn{4}{|c|}{\textbf{已知人 (Known Persons) - 89个}} \\
\hline
Gallery (前10条/人) & 89 & 890 & 构建注册库，计算自适应阈值 \\
Test Known (第11-20条/人) & 89 & 890 & 评估Known Class Accuracy (KCA) \\
\hline
\multicolumn{4}{|c|}{\textbf{未知人 (Unknown Persons) - 100个}} \\
\hline
Test Unknown (所有样本) & 100 & 12,345 & 评估Unknown Detection Rate (UDR) \\
\hline
\multicolumn{4}{|c|}{\textbf{总计}} \\
\hline
全部数据集 & 189 & 14,125 & - \\
\hline
\end{tabular}
\end{table}
```

## 结构化版本 - 显示时间序列

```latex
\begin{table}[htbp]
\centering
\caption{每个Known Person的Embedding序列划分}
\label{tab:embedding_sequence}
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
ID & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 & 17 & 18 & 19 & 20 \\
\hline
\text{集合} & \multicolumn{10}{c|}{\textbf{Gallery}} & \multicolumn{10}{c|}{\textbf{Test Known}} \\
\hline
\end{tabular}
\end{table}
```

## 统计指标版本 - 包含genuine pairs信息

```latex
\begin{table}[htbp]
\centering
\caption{Gallery集的统计特性}
\label{tab:gallery_statistics}
\begin{tabular}{|l|c|c|}
\hline
\textbf{指标} & \textbf{值} & \textbf{说明} \\
\hline
每人Gallery样本数 & 10 & Embeddings 1-10 \\
\hline
每人可生成的Genuine Pairs & $\binom{10}{2}=45$ & 用于计算μ和σ \\
\hline
总Genuine Pairs (89人) & $89 \times 45 = 4,005$ & 自适应阈值估计样本 \\
\hline
每人Test Known样本数 & 10 & Embeddings 11-20 \\
\hline
采样方式 & 连续顺序 & 确保时间顺序分割 \\
\hline
\end{tabular}
\end{table}
```

## 对比版本 - vs Unknown

```latex
\begin{table}[htbp]
\centering
\caption{Known vs Unknown数据集对比}
\label{tab:known_vs_unknown}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{属性} & \textbf{Known Persons} & \textbf{Unknown Persons} & \textbf{总计} \\
\hline
身份数量 & 89 (45\%) & 100 (55\%) & 189 \\
\hline
Gallery样本 & 890 (每人10) & 0 & 890 \\
\hline
Test样本 & 890 (每人10) & 12,345 (平均每人123) & 13,235 \\
\hline
样本中位数 & 10 & 123 & - \\
\hline
总样本数 & 1,780 & 12,345 & 14,125 \\
\hline
用途 & 识别/注册 & 拒绝/检测 & - \\
\hline
\end{tabular}
\end{table}
```

## 公式说明版本

```latex
\begin{table}[htbp]
\centering
\caption{数据集构建公式}
\label{tab:dataset_formulas}
\begin{tabular}{|c|l|c|}
\hline
\textbf{集合} & \textbf{公式} & \textbf{结果} \\
\hline
Gallery & $N_{\text{known}} \times G_{\text{size}} = 89 \times 10$ & 890 \\
\hline
Test Known & $N_{\text{known}} \times T_{\text{size}} = 89 \times 10$ & 890 \\
\hline
Test Unknown & $\sum_{i=1}^{100} n_i \approx 100 \times 123$ & 12,345 \\
\hline
Total & Gallery + Test Known + Test Unknown & 14,125 \\
\hline
Genuine Pairs & $\binom{G_{\text{size}}}{2} \times N_{\text{known}} = 45 \times 89$ & 4,005 \\
\hline
\end{tabular}
\end{table}
```

---

## 推荐使用

在你的论文中，建议这样使用：

**第3.2.2节（Gallery与测试集构建）**：
- 使用**清晰版本**或**详细版本**表格
- 配合文字说明

**第3.3节（数据集统计）**：
- 使用**统计指标版本**表格
- 说明genuine pairs的生成方式

**整体数据集概览**：
- 使用**对比版本**或**公式说明版本**
- 展现Known vs Unknown的完整对比

---

## 在Markdown中的查看方式

如果你想在文档中直接使用，也可以用Markdown表格替代：

```markdown
| Embedding ID | 1-10 (Gallery) | 11-20 (Test Known) |
|---|---|---|
| 每人数量 | 10条 | 10条 |
| 总人数 | 89人 | 89人 |
| 集合大小 | 890条 | 890条 |
| 用途 | 构建库/计算T_i | 评估KCA |
```

---

选择哪个版本放入你的论文呢？我可以帮你调整格式或风格！
