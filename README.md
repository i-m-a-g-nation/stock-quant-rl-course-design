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
