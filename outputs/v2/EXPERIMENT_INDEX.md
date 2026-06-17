# V2 实验索引

> 分支：v2-research-platform
> 本索引记录 V2 研究型实验平台的所有实验运行。

---

## 实验列表

| ID | 日期 | Stage | 说明 | 状态 |
|----|------|-------|------|------|
| V2-001 | 2026-06-17 | Stage 7 | Walk-forward 验证（方向分类, 17 folds） | ✅ 完成 |
| V2-002 | 2026-06-17 | Stage 8 | 收益率回归 Walk-forward 验证（17 folds） | ✅ 完成 |
| V2-003 | 2026-06-17 | Stage 9 | 三分类交易信号（3 thresholds × 17 folds × 5 models） | ✅ 完成 |
| V2-004 | 2026-06-17 | Stage 10 | 多周期预测与动态阈值（4 horizons × 2 tasks × 17 folds × 5 models） | ✅ 完成 |

---

## V2-001: Walk-forward 验证

**日期**：2026-06-17

**配置**：
- 数据：SPY_features_2015_2025.csv（2706 样本 × 17 特征）
- 目标：next_return > 0 → 1（二分类方向预测）
- 方法：expanding window walk-forward
- 初始训练：500 行
- 测试窗口：126 行
- fold 数：17
- 交易成本：0.05%

**模型**：
- DummyMostFrequent
- DummyStratified
- LogisticRegression
- RandomForestClassifier

**核心结果**：

| 模型 | accuracy_mean | balanced_accuracy_mean | roc_auc_mean | total_return_mean |
|------|---------------|------------------------|--------------|-------------------|
| DummyMostFrequent | 0.5528 | 0.5000 | 0.5000 | 0.0706 |
| DummyStratified | 0.4981 | 0.4909 | 0.4909 | 0.0145 |
| LogisticRegression | 0.5500 | 0.4987 | 0.4872 | 0.0696 |
| RandomForestClassifier | 0.4977 | 0.5027 | 0.5002 | 0.0257 |

**关键发现**：
- LR 预测上涨比例 97.57%，接近 DummyMostFrequent，反映多数类偏置
- RF 跨窗口不稳定（pred_pos_rate std=28.75%）
- 所有模型 ROC-AUC 接近 0.5，区分能力有限
- DummyMostFrequent 回测收益最高（7.06%），源于市场趋势

**输出文件**：
- `outputs/v2/tables/SPY_walk_forward_summary.csv`
- `outputs/v2/tables/SPY_walk_forward_classification_metrics.csv`
- `outputs/v2/tables/SPY_walk_forward_backtest_metrics.csv`
- `outputs/v2/tables/SPY_walk_forward_predictions.csv`
- `outputs/v2/figures/SPY_walk_forward_roc_auc_by_fold.png`
- `outputs/v2/figures/SPY_walk_forward_balanced_accuracy_by_fold.png`
- `outputs/v2/figures/SPY_walk_forward_total_return_by_fold.png`
- `outputs/v2/Sprint1_WALK_FORWARD_SUMMARY.md`

---

## V2-002: 收益率回归 Walk-forward 验证

**日期**：2026-06-17

**配置**：
- 数据：SPY_features_2015_2025.csv（2706 样本 × 17 特征）
- 目标：next_return（下一交易日收益率）
- 方法：expanding window walk-forward
- 初始训练：500 行
- 测试窗口：126 行
- fold 数：17
- 信号阈值：0.05%
- 交易成本：0.05%

**模型**：
- DummyRegressor_mean
- LinearRegression
- Ridge
- RandomForestRegressor
- GradientBoostingRegressor

**核心结果**：

| 模型 | MAE | RMSE | R² | direction_acc | total_return |
|------|-----|------|-----|---------------|-------------|
| DummyRegressor_mean | 0.007622 | 0.010922 | -0.0066 | 55.56% | 1.59% |
| LinearRegression | 0.008016 | 0.011304 | -0.0899 | 49.39% | -0.58% |
| Ridge | 0.007868 | 0.011130 | -0.0514 | 49.63% | 1.57% |
| RandomForestRegressor | 0.008529 | 0.012055 | -0.3057 | 50.89% | 1.59% |
| GradientBoostingRegressor | 0.010531 | 0.014369 | -1.0857 | 48.13% | 0.92% |

**关键发现**：
- 所有模型 R² 为负，预测效果不如均值基线
- 预测收益率幅度极小，缺乏区分度
- 方向命中率接近随机，无模型显著优于 Dummy
- 相关系数接近 0，预测值与真实值无显著关联
- 回测收益较低，不具备稳定的交易意义

**输出文件**：
- `outputs/v2/tables/SPY_return_regression_summary.csv`
- `outputs/v2/tables/SPY_return_regression_metrics.csv`
- `outputs/v2/tables/SPY_return_regression_backtest_metrics.csv`
- `outputs/v2/tables/SPY_return_regression_predictions.csv`
- `outputs/v2/figures/SPY_return_regression_rmse_by_fold.png`
- `outputs/v2/figures/SPY_return_regression_direction_accuracy_by_fold.png`
- `outputs/v2/figures/SPY_return_regression_total_return_by_fold.png`
- `outputs/v2/figures/SPY_return_regression_pred_vs_true.png`
- `outputs/v2/Sprint2_RETURN_REGRESSION_SUMMARY.md`

---

## V2-003: 三分类交易信号 Walk-forward 验证

**日期**：2026-06-17

**配置**：
- 数据：SPY_features_2015_2025.csv（2706 样本 × 17 特征）
- 目标：三分类（-1=下跌, 0=震荡, 1=上涨）
- 方法：expanding window walk-forward
- 初始训练：500 行
- 测试窗口：126 行
- fold 数：17
- thresholds：0.0005, 0.0010, 0.0020
- 交易成本：0.05%

**模型**：
- DummyMostFrequent
- DummyStratified
- LogisticRegression（class_weight=balanced）
- RandomForestClassifier（class_weight=balanced）
- GradientBoostingClassifier

**核心结果（threshold=0.0010）**：

| 模型 | balanced_accuracy | macro_f1 | total_return | trades |
|------|-------------------|----------|-------------|--------|
| DummyMostFrequent | 0.3333 | 0.2190 | 7.92% | 1.0 |
| DummyStratified | 0.3346 | 0.3321 | -1.08% | 62.1 |
| LogisticRegression | 0.3473 | 0.1814 | -0.49% | 2.2 |
| RandomForestClassifier | 0.3237 | 0.2914 | -2.56% | 31.9 |
| GradientBoostingClassifier | 0.3364 | 0.2976 | 2.31% | 29.2 |

**关键发现**：
- 所有模型 balanced_accuracy 接近随机水平（~0.33）
- DummyMostFrequent 回测收益最高（7.92%），源于市场趋势
- LogisticRegression 倾向预测 neutral（44.30%）
- 引入 no-trade zone 未能改善模型表现
- 三分类未优于二分类

**输出文件**：
- `outputs/v2/tables/SPY_three_class_threshold_summary.csv`
- `outputs/v2/tables/SPY_three_class_metrics.csv`
- `outputs/v2/tables/SPY_three_class_backtest_metrics.csv`
- `outputs/v2/tables/SPY_three_class_confusion_matrices.csv`
- `outputs/v2/tables/SPY_three_class_predictions.csv`
- `outputs/v2/figures/SPY_three_class_*.png`
- `outputs/v2/Sprint3_THREE_CLASS_SIGNAL_SUMMARY.md`

---

## V2-004: 多周期预测与动态阈值实验

**日期**：2026-06-17

**配置**：
- 数据：SPY_features_2015_2025.csv
- horizons：1, 5, 10, 20 日
- task_types：binary, three_class
- 固定阈值：h=1: 0.001, h=5: 0.003, h=10: 0.005, h=20: 0.010
- 动态阈值：k=0.50, threshold = k × vol_20 × √h
- fold 数：17
- 模型：5 个

**核心结果（二分类）**：

| horizon | DummyMostFrequent bal_acc | GBM bal_acc | RF bal_acc |
|---------|---------------------------|-------------|------------|
| 1d | 0.5000 | 0.5039 | 0.4973 |
| 5d | 0.5000 | 0.5278 | 0.5108 |
| 10d | 0.5000 | 0.5495 | 0.5579 |
| 20d | 0.5000 | 0.5579 | 0.5468 |

**核心结果（三分类固定阈值）**：

| horizon | DummyMostFrequent bal_acc | GBM bal_acc | RF bal_acc |
|---------|---------------------------|-------------|------------|
| 1d | 0.3333 | 0.3364 | 0.3237 |
| 5d | 0.3333 | 0.3554 | 0.3608 |
| 10d | 0.3333 | 0.3660 | 0.3592 |
| 20d | 0.3333 | 0.3726 | 0.3648 |

**关键发现**：
- 更长 horizon 使 balanced_accuracy 略有提升（2-5 个百分点）
- 动态阈值改善了类别平衡，但未改善模型预测能力
- DummyMostFrequent 回测收益最高，源于市场趋势
- 所有模型未显著优于 Dummy 基线
- 简化回测 20d DummyMostFrequent 2110.78% 是重叠收益复利放大，non-overlap 口径下仅为 9.29%

**输出文件**：
- `outputs/v2/tables/SPY_multi_horizon_summary.csv`
- `outputs/v2/tables/SPY_multi_horizon_metrics.csv`
- `outputs/v2/tables/SPY_multi_horizon_label_distribution.csv`
- `outputs/v2/figures/SPY_multi_horizon_*.png`
- `outputs/v2/Sprint4_MULTI_HORIZON_SIGNAL_SUMMARY.md`

---

## V2 阶段总结

**日期**：2026-06-17

**文档**：`outputs/v2/V2_STAGE7_TO_STAGE10_REVIEW.md`

**核心结论**：
- 当前仅靠价量技术特征，难以形成稳定短期交易信号
- 更长 horizon 有轻微改善，但不足以支持稳定交易
- 动态阈值改善类别分布，但未明显改善预测能力
- 复杂模型未稳定优于简单基线
- 后续方向：外部特征 > 多资产验证 > 模型扩展

**输出文件**：
- `outputs/v2/V2_STAGE7_TO_STAGE10_REVIEW.md`
- `outputs/v2/tables/SPY_v2_stage_summary.csv`

---

> 本索引随实验进展持续更新。
