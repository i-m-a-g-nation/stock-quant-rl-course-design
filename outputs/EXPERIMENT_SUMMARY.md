# 实验总摘要 — Enhanced Experiments

> 自动生成于实验完成后。本项目仍为课程实验，不构成投资建议。

---

## 1. 本次实验增强内容

1. **监督学习增强**：新增 DummyClassifier 基线（MostFrequent / Stratified），
   增加 balanced_accuracy、roc_auc、混淆矩阵等指标，验证模型是否优于随机基线。
2. **特征工程增强**：新增 8 个相对化特征（open_close_ratio 等），
   定义 FEATURE_COLS_ORIGINAL / RELATIVE / ALL 三组特征集。
3. **回测增强**：同时回测 BuyAndHold、AlwaysLong、LogisticRegression、
   RandomForestClassifier、DummyMostFrequent、DummyStratified 六个策略。
4. **Q-learning 增强**：将全样本训练/评估改为训练集(80%)训练、测试集(20%)评估，
   测试集使用 epsilon=0 贪心策略，同时对比 RandomPolicy 和 BuyAndHold。
5. **Permutation Importance**：对 LR 和 RF 在测试集上计算 permutation importance。

---

## 2. 监督学习模型指标摘要

| 模型 | accuracy | balanced_accuracy | f1 | roc_auc | positive_rate_pred |
|------|----------|-------------------|-----|---------|-------------------|
| DummyMostFrequent | 0.5886 | 0.5 | 0.741 | 0.5 | 1.0 |
| DummyStratified | 0.4649 | 0.4611 | 0.5151 | 0.4611 | 0.5148 |
| LogisticRegression | 0.5886 | 0.5 | 0.741 | 0.509 | 1.0 |
| RandomForestClassifier | 0.4262 | 0.5092 | 0.0772 | 0.4791 | 0.0332 |

### 混淆矩阵

| 模型 | TN | FP | FN | TP |
|------|-----|-----|-----|-----|
| DummyMostFrequent | 0 | 223 | 0 | 319 |
| DummyStratified | 98 | 125 | 165 | 154 |
| LogisticRegression | 0 | 223 | 0 | 319 |
| RandomForestClassifier | 218 | 5 | 306 | 13 |

---

## 3. 多策略回测结果摘要

| 策略 | cumulative_return | annual_return | sharpe | max_drawdown | calmar | win_rate | trade_count |
|------|-------------------|---------------|--------|--------------|--------|----------|-------------|
| BuyAndHold | 0.623849 | 0.238218 | 1.4874 | -0.189989 | 1.2538 | 0.5896 | 1 |
| AlwaysLong | 0.623849 | 0.238218 | 1.4874 | -0.189989 | 1.2538 | 0.5896 | 1 |
| LogisticRegression | 0.623849 | 0.238218 | 1.4874 | -0.189989 | 1.2538 | 0.5896 | 1 |
| RandomForestClassifier | 0.146831 | 0.066418 | 0.8745 | -0.017646 | 3.7639 | 0.4444 | 18 |
| DummyMostFrequent | 0.623849 | 0.238218 | 1.4874 | -0.189989 | 1.2538 | 0.5896 | 1 |
| DummyStratified | -0.143332 | -0.066625 | -0.6484 | -0.202616 | -0.3288 | 0.3554 | 258 |

---

## 4. Q-learning 样本外测试摘要

### 训练/测试集信息

- **train_train_start**: 2015-03-30
- **train_train_end**: 2023-11-01
- **train_train_samples**: 2165
- **train_q_table_states**: 27
- **train_action_space**: 2
- **train_episodes**: 50
- **train_final_epsilon**: 0.2335
- **test_test_start**: 2023-11-02
- **test_test_end**: 2025-12-31
- **test_test_samples**: 542

### 测试集策略对比

| 策略 | total_return | sharpe_ratio | max_drawdown | number_of_trades |
|------|--------------|--------------|--------------|------------------|
| RandomPolicy | -0.112089 | -0.3813 | -0.242896 | 278 |
| BuyAndHold | 0.581495 | 1.416 | -0.189989 | 0 |
| QLearning | -0.095907 | -1.3472 | -0.095907 | 58 |

---

## 5. 主要发现

### 5.1 逻辑回归 vs 多数类基线

- 逻辑回归 accuracy (0.5886) **未优于**多数类基线 (0.5886)，需关注模型是否仅学到类别偏置。

### 5.2 模型是否偏向预测上涨

- LogisticRegression: 预测上涨比例 (100.00%) vs 真实上涨比例 (58.86%)，偏差 +41.14%，**存在偏向**。
- RandomForestClassifier: 预测上涨比例 (3.32%) vs 真实上涨比例 (58.86%)，偏差 -55.54%，**存在偏向**。

### 5.3 Q-learning 是否跑赢 BuyAndHold

- Q-learning (-0.0959) **未跑赢** BuyAndHold (0.5815)。

### 5.4 Q-learning 是否降低最大回撤

- Q-learning 最大回撤 (-0.0959) **小于** BuyAndHold (-0.1900)，风险有所降低。

### 5.5 LSTM 过拟合分析

请参考 `outputs/figures/SPY_lstm_training_curve.png` 中的训练/测试损失曲线。
若训练损失持续下降而测试损失上升，则存在过拟合迹象。

---

## 6. 后续报告需要更新的位置

以下章节需要根据增强实验结果更新：

1. **摘要**：更新主要实验结论和指标数据。
2. **4.2 机器学习预测结果**：
   - 添加 DummyClassifier 基线对比。
   - 添加 balanced_accuracy、roc_auc、混淆矩阵分析。
   - 讨论 positive_rate_pred vs positive_rate_true。
   - 引用 permutation importance 结果。
3. **4.4 回测与风险评估结果**：
   - 替换为六策略对比表。
   - 添加权益曲线和回撤曲线图。
4. **4.5 强化学习实验结果**：
   - 添加训练集/测试集分离说明。
   - 添加测试集上 Q-learning vs BuyAndHold vs Random 对比。
5. **4.6 综合分析**：综合各部分结果进行分析。
6. **4.7 结论与展望**：更新结论。
7. **附件代码片段**：更新为最新代码。

---

## 7. 新增/更新文件清单

### 新增输出文件

| 文件 | 说明 |
|------|------|
| `outputs/tables/SPY_ml_model_metrics_enhanced.csv` | 增强 ML 指标表 |
| `outputs/tables/SPY_ml_test_predictions_enhanced.csv` | 增强测试预测 |
| `outputs/tables/SPY_ml_confusion_matrices.csv` | 混淆矩阵 |
| `outputs/tables/SPY_feature_columns_enhanced.csv` | 特征字段说明 |
| `outputs/tables/SPY_strategy_backtest_enhanced_metrics.csv` | 多策略回测指标 |
| `outputs/tables/SPY_strategy_equity_curves_enhanced.csv` | 多策略权益曲线 |
| `outputs/tables/SPY_permutation_importance.csv` | Permutation importance |
| `outputs/tables/SPY_rl_train_test_summary.csv` | RL 训练/测试摘要 |
| `outputs/tables/SPY_rl_test_strategy_metrics.csv` | RL 测试集策略指标 |
| `outputs/tables/SPY_rl_test_equity_curves.csv` | RL 测试集权益曲线 |
| `outputs/figures/SPY_ml_metrics_enhanced.png` | ML 指标对比图 |
| `outputs/figures/SPY_permutation_importance_top15.png` | Perm Imp 图 |
| `outputs/figures/SPY_strategy_equity_curves_enhanced.png` | 多策略权益图 |
| `outputs/figures/SPY_strategy_drawdowns_enhanced.png` | 多策略回撤图 |
| `outputs/figures/SPY_rl_test_equity_curves.png` | RL 测试集权益图 |
| `outputs/figures/SPY_rl_test_drawdowns.png` | RL 测试集回撤图 |

### 更新的源代码文件

| 文件 | 变更说明 |
|------|----------|
| `src/ml_models.py` | 新增 DummyClassifier、增强指标、permutation importance |
| `src/features.py` | 新增相对化特征、特征列集合定义 |
| `src/backtest.py` | 新增多策略回测函数 |
| `src/rl_agent.py` | 新增训练集/测试集分离评估 |
| `run_stage2.py` | 增加特征字段说明保存 |
| `run_stage3.py` | 增加增强指标输出 |
| `run_stage4.py` | 增加多策略回测 |
| `run_stage6.py` | 增加训练集/测试集分离 Q-learning |
| `README.md` | 增加 Enhanced Experiments 说明 |

---

**免责声明**：本项目仍为课程实验，不构成投资建议。
