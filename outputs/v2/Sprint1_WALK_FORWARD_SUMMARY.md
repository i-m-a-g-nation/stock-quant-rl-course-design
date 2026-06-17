# Sprint 1 Walk-forward 验证总结

> 分支：v2-research-platform
> 运行时间：2026-06-17
> 本项目仅用于课程实验和方法验证，不构成投资建议。

---

## 1. 实验配置

| 参数 | 值 |
|------|-----|
| 数据文件 | `data/features/SPY_features_2015_2025.csv` |
| 样本总数 | 2706（删除最后一行无 next_return） |
| 特征数量 | 17（数值型技术特征，排除 OHLCV 和 Close/Adj Close） |
| 目标变量 | next_return > 0 → 1，否则 0 |
| 上涨比例 | 54.7% |
| 初始训练窗口 | 500 行 |
| 测试窗口大小 | 126 行（约半年交易日） |
| fold 数量 | 17 |
| 交易成本 | 0.05%（仅仓位变化时扣除） |

## 2. 模型列表

| 模型 | 说明 |
|------|------|
| DummyMostFrequent | 始终预测多数类（上涨） |
| DummyStratified | 按类别比例随机预测 |
| LogisticRegression | max_iter=1000, random_state=42 |
| RandomForestClassifier | n_estimators=200, max_depth=5, random_state=42 |

## 3. 分类指标跨 fold 汇总

| 模型 | accuracy | balanced_accuracy | roc_auc | pred_pos_rate |
|------|----------|-------------------|---------|---------------|
| DummyMostFrequent | 0.5528 ± 0.0470 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 |
| DummyStratified | 0.4981 ± 0.0418 | 0.4909 ± 0.0426 | 0.4909 ± 0.0426 | 0.5710 ± 0.0139 |
| LogisticRegression | 0.5500 ± 0.0497 | 0.4987 ± 0.0052 | 0.4872 ± 0.0589 | 0.9757 ± 0.1001 |
| RandomForestClassifier | 0.4977 ± 0.0462 | 0.5027 ± 0.0355 | 0.5002 ± 0.0579 | 0.5462 ± 0.2875 |

## 4. 回测指标跨 fold 汇总

| 模型 | total_return | sharpe | max_drawdown |
|------|-------------|--------|--------------|
| DummyMostFrequent | 0.0706 ± 0.1294 | 1.0900 ± 1.3338 | -0.1071 ± 0.0784 |
| DummyStratified | 0.0145 ± 0.1012 | 0.1643 ± 1.3359 | -0.0794 ± 0.0442 |
| LogisticRegression | 0.0696 ± 0.1295 | 1.0828 ± 1.3303 | -0.1069 ± 0.0786 |
| RandomForestClassifier | 0.0257 ± 0.1093 | 0.6193 ± 1.3607 | -0.0871 ± 0.0616 |

## 5. 关键发现

### 5.1 LogisticRegression 反映多数类偏置

LogisticRegression 的预测上涨比例（pred_pos_rate）为 97.57%，接近 DummyMostFrequent 的 100%。这表明 LR 在大多数 fold 中仍然倾向于预测上涨，与 V1 的结论一致。LR 的 accuracy（0.5500）与 DummyMostFrequent（0.5528）非常接近，balanced_accuracy（0.4987）甚至略低于 Dummy（0.5000），说明 LR 未能学到有效的方向区分信号。

### 5.2 RandomForestClassifier 跨窗口不稳定

RandomForest 的预测上涨比例标准差为 0.2875（28.75%），远高于其他模型，说明 RF 在不同窗口的预测行为差异很大。RF 的 roc_auc（0.5002 ± 0.0579）接近随机水平，且标准差较大，表明模型在不同时间段的表现不稳定。

### 5.3 DummyMostFrequent 回测表现不低

DummyMostFrequent 的平均 total_return 为 7.06%，高于 LogisticRegression（6.96%）和 RandomForest（2.57%）。这是因为 SPY 在测试期间整体呈上涨趋势，始终持有的策略自然获得正收益。这一结果再次说明，在趋势市场中，简单的买入持有可能优于复杂的主动交易策略。

### 5.4 所有模型 ROC-AUC 接近 0.5

四个模型的 ROC-AUC 均在 0.49~0.50 之间，标准差较大（0.04~0.06），说明模型在不同时间段的区分能力接近随机水平，且跨窗口波动较大。

### 5.5 跨窗口不稳定性

所有模型的指标标准差均较大（如 total_return std 在 0.10~0.13 之间），说明模型在不同时间段的表现差异显著。这是金融时间序列预测的典型特征：市场结构随时间变化，历史模式难以持续预测未来。

## 6. 不能夸大的结论

1. LogisticRegression 的 accuracy 略高于 0.5，但不能说明模型有稳定预测能力
2. 所有模型的 ROC-AUC 接近 0.5，区分能力有限
3. 回测结果为课程实验级简化，不代表真实交易
4. 跨窗口不稳定性说明模型泛化能力有限
5. DummyMostFrequent 的正收益源于市场趋势，而非预测能力

## 7. 下一步建议

1. **Stage 8：收益率回归** — 预测 next_return 的大小而非方向
2. **Stage 9：三分类交易信号** — 引入"震荡/不交易"类别
3. **模型扩展** — RidgeClassifier、GradientBoosting 等
4. **基线扩展** — MA 交叉、动量、波动率过滤等策略
5. **特征选择** — 基于 permutation importance 筛选有效特征

---

> 本文档为 Sprint 1 实验总结，仅供课程扩展和研究验证参考。
> 不构成投资建议。
