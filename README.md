# 公开发布版说明

本仓库为**公开发布版（public version）**，仅包含代码和轻量文档。完整的数据、模型、输出文件等请参阅本地完整开发版（local full version）。

- GitHub 仓库：public version
- 本地完整开发版：包含所有 outputs、data、models 等
- 有意设计：public 版不包含所有 outputs，不是实验缺失
- 完整结果可通过运行 `run_stage1.py` 到 `run_stage6.py` 重新生成
- 最终报告 Markdown 位于 `docs/大数据课设报告_最终版.md`
- docx 报告仅保留在本地 full 版

---

# 基于机器学习与强化学习的股票交易策略建模及量化风险评估研究

> 课程设计项目。所有模型输出仅供课程研究参考，不构成实际投资建议。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

核心依赖：pandas, numpy, matplotlib, scikit-learn, joblib, yfinance, torch, jupyter, nbformat

### 运行全部阶段

```bash
python run_stage1.py  # 数据获取与清洗
python run_stage2.py  # 特征工程
python run_stage3.py  # 机器学习基线
python run_stage4.py  # 风险指标 + 回测
python run_stage5.py  # LSTM 方向预测
python run_stage6.py  # 强化学习环境 + Q-learning
```

---

## 阶段 1：金融数据获取与清洗

### 运行

```bash
python run_stage1.py
```

### 生成文件

| 文件 | 说明 |
|------|------|
| `data/raw/SPY_raw_2015_2025.csv` | SPY 原始行情数据 (2015-2025) |
| `data/processed/SPY_clean_2015_2025.csv` | 清洗后的行情数据 |
| `outputs/tables/SPY_data_quality_report.csv` | 数据质量报告 |
| `outputs/figures/SPY_close_price.png` | 收盘价 + 移动均线图 |
| `outputs/figures/SPY_volume.png` | 成交量柱状图 |
| `outputs/figures/SPY_return_distribution.png` | 日收益率分布直方图 |

> `data/` 和 `outputs/` 下的生成物如被 `.gitignore` 忽略，可通过 `python run_stage1.py` 复现。

## 阶段 2：特征工程

### 运行

```bash
python run_stage2.py
```

### 生成文件

| 文件 | 说明 |
|------|------|
| `data/features/SPY_features_2015_2025.csv` | 特征数据集 (OHLCV + 9项技术特征) |
| `outputs/tables/SPY_feature_summary.csv` | 特征摘要报告 |

> 阶段 2 依赖阶段 1 的 `data/processed/SPY_clean_2015_2025.csv`，请确保已运行 `python run_stage1.py`。

## 阶段 3：机器学习基线模型

### 运行

```bash
python run_stage3.py
```

### 生成文件

| 文件 | 说明 |
|------|------|
| `outputs/tables/SPY_ml_baseline_metrics.csv` | 模型评估指标 |
| `outputs/models/SPY_logistic_regression.joblib` | 逻辑回归模型 |
| `outputs/models/SPY_random_forest_classifier.joblib` | 随机森林模型 |

> 阶段 3 依赖阶段 2 的 `data/features/SPY_features_2015_2025.csv`，请确保已运行 `python run_stage2.py`。

## 阶段 4：风险指标评估 + 回测

### 运行

```bash
python run_stage4.py
```

### 生成文件

| 文件 | 说明 |
|------|------|
| `outputs/tables/SPY_market_returns.csv` | 市场日收益率、累计收益、回撤序列 |
| `outputs/tables/SPY_risk_metrics.csv` | 10 项风险/统计指标 |
| `outputs/tables/SPY_prediction_direction_analysis.csv` | 预测方向命中率统计 |
| `outputs/tables/SPY_strategy_backtest.csv` | 策略回测明细 |
| `outputs/tables/SPY_strategy_metrics.csv` | 策略回测指标 |
| `outputs/figures/SPY_strategy_equity_curve.png` | 策略 vs 买入持有资金曲线 |

> 阶段 4 依赖阶段 1 的清洗数据和阶段 3 的预测结果。回测为简化版本，未充分考虑交易成本、滑点等现实约束。

## 阶段 5：LSTM 方向预测

### 运行

```bash
python run_stage5.py
```

### 生成文件

| 文件 | 说明 |
|------|------|
| `outputs/tables/SPY_lstm_metrics.csv` | LSTM 评估指标 |
| `outputs/tables/SPY_lstm_test_predictions.csv` | 测试集预测结果 |
| `outputs/models/SPY_lstm_model.pt` | PyTorch 模型 |
| `outputs/figures/SPY_lstm_training_curve.png` | 训练/测试损失与准确率曲线 |

> 阶段 5 依赖阶段 2 的 `data/features/SPY_features_2015_2025.csv`。需要 PyTorch 环境。

## 阶段 6：强化学习交易环境 + Q-learning

### 运行

```bash
python run_stage6.py
```

### 生成文件

| 文件 | 说明 |
|------|------|
| `outputs/tables/SPY_rl_env_baseline_metrics.csv` | 随机策略 vs 买入持有指标对比 |
| `outputs/tables/SPY_rl_env_baseline_history.csv` | 策略历史记录 |
| `outputs/figures/SPY_rl_env_baseline_equity.png` | 权益曲线图 |
| `outputs/tables/SPY_rl_qlearning_metrics.csv` | Random/BuyAndHold/QLearning 三策略对比 |
| `outputs/tables/SPY_rl_qlearning_history.csv` | Q-learning 历史记录 |
| `outputs/tables/SPY_rl_qlearning_training_log.csv` | Q-learning 训练日志 |
| `outputs/figures/SPY_rl_qlearning_equity.png` | 三策略权益曲线对比 |

> 阶段 6 依赖阶段 2 的特征数据。Q-learning 为课程级 tabular 离散状态简化实现，效果弱于 Buy & Hold，主要用于展示 RL 交易环境和智能体训练流程。未使用 PPO/DQN/A2C/stable_baselines3 等深度 RL 方法。

---

## Enhanced Experiments（增强实验）

在原有阶段 1-6 基础上，新增以下增强实验内容。

### 运行方式

按原有顺序运行即可，增强功能已集成到各阶段中：

```bash
python run_stage1.py  # 数据获取与清洗（无变化）
python run_stage2.py  # 特征工程（新增相对化特征）
python run_stage3.py  # 机器学习（新增 DummyClassifier + 增强指标）
python run_stage4.py  # 回测（新增多策略对比）
python run_stage5.py  # LSTM（无变化）
python run_stage6.py  # Q-learning（新增训练集/测试集分离）
python scripts/collect_experiment_summary.py  # 生成实验总摘要
```

### 增强内容概览

| 阶段 | 增强内容 |
|------|----------|
| 阶段 2 | 新增 8 个相对化特征，定义 FEATURE_COLS_ORIGINAL / RELATIVE / ALL |
| 阶段 3 | 新增 DummyClassifier 基线、balanced_accuracy/roc_auc/混淆矩阵、permutation importance |
| 阶段 4 | 新增 6 策略回测对比（BuyAndHold/AlwaysLong/LR/RF/DummyMostFrequent/DummyStratified） |
| 阶段 6 | Q-learning 训练集(80%)/测试集(20%)分离，测试集 epsilon=0 贪心评估 |

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
| `outputs/EXPERIMENT_SUMMARY.md` | 实验总摘要 |

> 免责声明：本项目仍为课程实验，不构成投资建议。

---

## V2 Research Platform

本项目包含两个版本：

1. **main/V1**：课程设计版，包含 Stage 1–6
2. **v2-research-platform**：研究扩展版，包含 Stage 7–10

### V2 主要内容

- **Stage 7**：Walk-forward 验证
- **Stage 8**：收益率回归预测
- **Stage 9**：三类信号分类（买入/持有/卖出）
- **Stage 10**：多周期信号预测

### 重要说明

- Stage 10 已修正 long-horizon overlapping return 的解释
- 20d DummyMostFrequent 2110.78% 是重叠收益复利放大，不是真实可执行收益
- non-overlap 口径约为 9.29%
- 本项目不构成投资建议

### 查看 V2 版本

切换到 `v2-research-platform` 分支查看 V2 研究平台代码和结果。
