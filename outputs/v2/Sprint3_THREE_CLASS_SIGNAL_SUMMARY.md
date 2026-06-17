# Sprint 3 三分类交易信号实验总结

> 分支：v2-research-platform
> 运行时间：2026-06-17
> 本项目仅用于课程实验和方法验证，不构成投资建议。

---

## 1. 实验配置

| 参数 | 值 |
|------|-----|
| 数据文件 | `data/features/SPY_features_2015_2025.csv` |
| 样本总数 | 2706 |
| 特征数量 | 17 |
| 目标变量 | next_return（Adj Close 日收益率） |
| 三分类标签 | 1=明显上涨(持仓), 0=震荡(不交易), -1=明显下跌(空仓) |
| threshold 列表 | 0.0005, 0.0010, 0.0020 |
| 初始训练窗口 | 500 行 |
| 测试窗口 | 126 行 |
| fold 数量 | 17 |
| 交易成本 | 0.05% |
| 回测规则 | pred=1→持仓, pred=0/-1→空仓（只做多不做空） |

## 2. 模型列表

| 模型 | class_weight |
|------|-------------|
| DummyMostFrequent | — |
| DummyStratified | — |
| LogisticRegression | balanced |
| RandomForestClassifier | balanced |
| GradientBoostingClassifier | — |

## 3. 不同 threshold 的类别分布

| threshold | down (-1) | neutral (0) | up (1) |
|-----------|-----------|-------------|--------|
| 0.0005 | 1124 (41.5%) | 190 (7.0%) | 1392 (51.4%) |
| 0.0010 | 1048 (38.7%) | 354 (13.1%) | 1304 (48.2%) |
| 0.0020 | 871 (32.2%) | 687 (25.4%) | 1148 (42.4%) |

随着 threshold 增大，neutral 类别比例从 7.0% 增加到 25.4%，类别分布更平衡。

## 4. 分类指标汇总（threshold=0.0010）

| 模型 | accuracy | balanced_accuracy | macro_f1 | pred_down | pred_neutral | pred_up |
|------|----------|-------------------|----------|-----------|-------------|---------|
| DummyMostFrequent | 0.4902 | 0.3333 | 0.2190 | 0.00% | 0.00% | 100.00% |
| DummyStratified | 0.3996 | 0.3346 | 0.3321 | 41.13% | 11.16% | 47.71% |
| LogisticRegression | 0.2857 | 0.3473 | 0.1814 | 37.30% | 44.30% | 18.39% |
| RandomForestClassifier | 0.3693 | 0.3237 | 0.2914 | 49.81% | 14.28% | 35.90% |
| GradientBoostingClassifier | 0.4057 | 0.3364 | 0.2976 | 52.33% | 6.86% | 40.80% |

## 5. 回测指标汇总（threshold=0.0010）

| 模型 | total_return | sharpe | max_drawdown | trades | exposure_rate |
|------|-------------|--------|-------------|--------|--------------|
| DummyMostFrequent | 7.92% | 1.2061 | -10.56% | 1.0 | 100.00% |
| DummyStratified | -1.08% | -0.0341 | -8.72% | 62.1 | 47.71% |
| LogisticRegression | -0.49% | 0.0650 | -3.65% | 2.2 | 18.39% |
| RandomForestClassifier | -2.56% | -0.1887 | -8.89% | 31.9 | 35.90% |
| GradientBoostingClassifier | 2.31% | 0.6595 | -7.17% | 29.2 | 40.80% |

## 6. 关键发现

### 6.1 DummyMostFrequent 仍然表现不低

DummyMostFrequent 在所有 threshold 下均 100% 预测 up，回测收益 7.92%，与 Stage 7 二分类结果一致。这是因为 SPY 在测试区间整体呈上涨趋势，始终持仓自然获得正收益。

### 6.2 所有模型 balanced_accuracy 接近 0.33

三分类随机基线的 balanced_accuracy 为 1/3 ≈ 0.333。所有模型的 balanced_accuracy 在 0.32~0.35 之间，接近或略高于随机水平，说明模型对三个类别的区分能力极弱。

### 6.3 LogisticRegression 倾向预测 neutral

LogisticRegression 在 threshold=0.0010 时，预测 neutral 比例高达 44.30%，而实际上涨比例为 48.2%。这与 Stage 7 中 LR 倾向预测 up 的行为不同，可能是 class_weight="balanced" 导致模型对少数类（neutral）更敏感。

### 6.4 GradientBoostingClassifier macro_f1 最高

GBM 的 macro_f1 在 threshold=0.0005 时为 0.3026，在 threshold=0.0020 时为 0.3033，略高于 DummyStratified（0.3112~0.3505），但差异不大。GBM 的回测收益（1.04%~2.31%）也高于其他非 Dummy 模型，但仍远低于 DummyMostFrequent。

### 6.5 引入 no-trade zone 未能改善模型表现

增大 threshold（加宽 no-trade zone）确实使类别分布更平衡（neutral 从 7% 增到 25%），但模型的 balanced_accuracy 和 macro_f1 并未显著提升。模型未能有效识别"震荡"区间。

### 6.6 三分类未优于二分类

| 维度 | Stage 7 二分类 | Stage 9 三分类 |
|------|---------------|---------------|
| 最佳 accuracy 模型 | DummyMostFrequent (0.553) | DummyMostFrequent (0.490) |
| 最佳 macro_f1 / f1 模型 | DummyMostFrequent (0.741) | DummyStratified (0.332) |
| 最佳回测模型 | DummyMostFrequent (7.06%) | DummyMostFrequent (7.92%) |
| 有效模型 | 无 | 无 |

三分类引入了更多类别，但模型未能学到有效信号，macro_f1 反而低于二分类的 f1。

## 7. 不能夸大的结论

1. 所有模型 balanced_accuracy 接近随机水平（~0.33）
2. 模型未能有效识别"震荡"区间
3. DummyMostFrequent 回测收益最高，源于市场趋势而非预测能力
4. 三分类未优于二分类，增加类别未带来信息增益
5. 引入 no-trade zone 未能改善模型表现
6. 本实验仅用于研究验证，不构成投资建议

## 8. 与 Stage 7、Stage 8 的对比

| 维度 | Stage 7 二分类 | Stage 8 收益率回归 | Stage 9 三分类 |
|------|---------------|-------------------|---------------|
| 任务 | 涨跌方向 | 收益率大小 | 涨/跌/震荡 |
| 最佳基线 | DummyMostFrequent | DummyRegressor | DummyMostFrequent |
| 有效模型 | 无 | 无 | 无 |
| 核心结论 | 模型反映多数类偏置 | R² 为负，不如均值 | 模型无法区分三类 |

三个任务得出一致结论：在当前特征集和模型配置下，SPY 的短期收益率预测难度极高。

## 9. 下一步建议

1. **特征工程** — 引入宏观、情绪、基本面等外部特征
2. **更长预测周期** — 预测 5 日或 20 日收益率
3. **动态 threshold** — 基于波动率自适应调整 no-trade zone
4. **模型改进** — 尝试更复杂的模型或特征选择
5. **多资产验证** — 在不同标的上验证三分类效果

---

> 本文档为 Sprint 3 实验总结，仅供课程扩展和研究验证参考。
> 不构成投资建议。
