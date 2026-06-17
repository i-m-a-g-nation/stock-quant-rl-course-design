# V2 Research Platform Plan

> 分支：v2-research-platform
> 本文档为 V2 研究型量化实验平台的规划文档。
> V1 课程版已完成，V2 在此基础上进化为更严谨的研究型实验平台。

---

## 1. V1 当前状态

V1 课程版已全部完成，包括：

| 模块 | 状态 | 说明 |
|------|------|------|
| 数据获取与清洗 | ✅ | SPY 2015-2025，2766 行原始数据 |
| 特征工程 | ✅ | 2707 行 × 23 列（6 原始 + 9 技术 + 8 相对化） |
| 监督学习 | ✅ | DummyClassifier + LogisticRegression + RandomForestClassifier + 混淆矩阵 + Permutation Importance |
| LSTM | ✅ | 单层 LSTM，accuracy=0.454，roc_auc=0.4916 |
| 风险指标 | ✅ | 年化收益、波动率、夏普、最大回撤、VaR、CVaR 等 |
| 六策略回测 | ✅ | BuyAndHold / AlwaysLong / LR / RF / DummyMostFrequent / DummyStratified |
| Tabular Q-learning | ✅ | 训练集/测试集分离，测试集 B&H 58.15%，QL -9.59% |
| 双版本管理 | ✅ | public 版（GitHub 轻量展示）+ full 版（本地完整） |
| 课程报告 | ✅ | docx + REPORT_UPDATE_DRAFT.md |

### V1 关键结论

- LogisticRegression accuracy=0.5886，与 DummyMostFrequent 基线相同，二者均 100% 预测上涨
- RandomForestClassifier accuracy=0.4262，对上涨类别召回率仅 0.0408
- LSTM roc_auc=0.4916，接近随机水平
- Q-learning 测试集总收益 -9.59%，未跑赢 BuyAndHold 58.15%
- Q-learning 最大回撤 -9.59%，低于 BuyAndHold -19.00%
- 短期方向预测难度高，复杂模型不一定优于简单基线

### V1 版本边界

- **public 版**：轻量 GitHub 展示版，仅含代码 + 轻量 summary
- **full 版**：本地完整版，含 data、outputs/tables、outputs/figures、outputs/models、docs/docx
- **V2 不应破坏 V1 课程提交结果**：V1 outputs 保持不变，V2 输出放入独立目录

---

## 2. V2 总目标

V2 不以"提高收益"为唯一目标，而是把项目进化为更严谨的研究型量化实验平台。

### 核心目标

1. **更严谨的时间序列验证**：从单次 80/20 切分升级为 walk-forward validation
2. **更真实的交易回测**：引入手续费、滑点、换手率、交易明细
3. **更强但可解释的基线策略**：补充 MA 交叉、动量、波动率过滤等策略
4. **更多预测任务**：方向分类、收益率回归、三分类交易信号
5. **更规范的强化学习环境**：更清晰的 observation/action/reward 设计
6. **更完整的实验记录**：实验索引、运行日志、配置管理
7. **更清晰的风险评估**：跨窗口稳定性、收益分布、尾部风险
8. **后续可用于论文、答辩或作品集展示**

---

## 3. V2 预测任务升级

### 当前 V1 主线

V1 主线是"预测下一交易日涨跌方向"（二分类），不是直接预测价格。

### V2 不建议做的事

**不建议直接把主线改成预测 next_close 价格。** 原因：

- 价格序列存在趋势和尺度问题
- 模型容易学成"明天价格约等于今天价格"（朴素预测）
- 价格误差小不代表交易信号有效
- 回归预测的评估指标（MAE/RMSE）与交易盈利没有直接对应关系

### V2 建议扩展为三类任务

#### Task A：方向分类（保留并增强）

目标：
- 预测 next_return > 0（保留当前二分类主线）
- 输出 accuracy、balanced_accuracy、ROC-AUC、混淆矩阵、预测上涨比例
- 接入 walk-forward 验证和回测
- 与 Dummy 基线严格对比

#### Task B：收益率回归（新增）

目标：
- 预测 next_return 或 future_5d_return
- 输出 MAE、RMSE、R²、方向命中率（IC 或相关系数）
- 可用于信号强度和仓位大小设计
- 不直接解释为价格预测能力
- 可做分位数组合回测

#### Task C：三分类交易信号（新增）

目标：
- next_return > threshold → 上涨/可持仓（信号 = 1）
- abs(next_return) <= threshold → 震荡/不交易（信号 = 0）
- next_return < -threshold → 下跌/空仓或避险（信号 = -1）
- 减少无意义交易，结合交易成本更合理
- threshold 可取 0 或动态阈值（如波动率的一定比例）

---

## 4. 模块规划

### Module A：V1 冻结与 V2 分支管理

- `v1.0-course-final` tag 标记课程版历史点（如需要）
- `v2-research-platform` 用于研究型开发
- V2 输出放入 `outputs/v2/`
- 不覆盖 V1 outputs

目录结构：

```
outputs/
├── v1/                    # V1 输出（当前 outputs/ 下的文件）
│   ├── tables/
│   ├── figures/
│   └── models/
├── v2/                    # V2 输出
│   ├── tables/
│   ├── figures/
│   ├── models/
│   └── runs/
└── EXPERIMENT_SUMMARY.md  # 全局摘要
```

### Module B：Walk-forward 时间序列验证

**目标**：从单次 80/20 切分升级为 walk-forward validation。

**设计**：

- 支持 rolling window（固定窗口大小）和 expanding window（窗口递增）
- 每个窗口训练集早于测试集，严格无未来信息泄漏
- 每个窗口独立训练模型、独立预测、独立评估
- 输出跨窗口的均值和标准差，衡量模型稳定性

**输出指标**：

- 每窗口：accuracy、balanced_accuracy、ROC-AUC、预测上涨比例
- 每窗口：策略收益、Sharpe、最大回撤
- 汇总：跨窗口均值 ± 标准差

**建议新增**：

```
src/validation.py
run_stage7_walk_forward.py
outputs/v2/tables/SPY_walk_forward_metrics.csv
outputs/v2/tables/SPY_walk_forward_window_details.csv
outputs/v2/figures/SPY_walk_forward_roc_auc.png
outputs/v2/figures/SPY_walk_forward_returns.png
```

### Module C：Backtest V2

**目标**：更真实的交易回测框架。

**增强点**：

- 手续费（可配置，如 0.1%）
- 滑点（可配置，如 0.05%）
- 换手率统计
- 持仓天数统计
- 交易明细日志（每笔开仓/平仓记录）
- signal shift(1)（已有，保持）
- benchmark 对比（BuyAndHold）
- 样本外表现统计

**建议新增**：

```
src/backtest_v2.py
outputs/v2/tables/SPY_trade_log_v2.csv
outputs/v2/tables/SPY_backtest_v2_metrics.csv
outputs/v2/figures/SPY_backtest_v2_equity.png
outputs/v2/figures/SPY_backtest_v2_drawdown.png
```

### Module D：更强基线策略

**优先补充的策略**：

| 策略 | 说明 |
|------|------|
| BuyAndHold | 始终持有（已有） |
| AlwaysCash | 始终空仓（零收益基准） |
| AlwaysLong | 始终满仓（等价 BuyAndHold） |
| MovingAverageCross | 短期均线上穿长期均线做多，下穿空仓 |
| Momentum_20D | 过去 20 日收益 > 0 则做多，否则空仓 |
| VolatilityFilter | 波动率高于阈值时空仓，否则持有 |
| RandomPolicy | 随机交易（已有） |
| DummyMostFrequent | 多数类基线（已有） |
| DummyStratified | 分层随机基线（已有） |

**每个策略输出**：

- total_return / annual_return / annual_volatility
- Sharpe / max_drawdown / Calmar
- win_rate / trades / turnover

### Module E：模型升级

**候选模型**：

| 模型 | 说明 |
|------|------|
| RidgeClassifier | 正则化线性分类，防止过拟合 |
| SVM (LinearSVC) | 支持向量机，适合高维特征 |
| GradientBoosting | 梯度提升树，sklearn 实现 |
| XGBoost / LightGBM | 更高效的梯度提升（可选） |
| MLP | 多层感知机，简单神经网络 |

**要求**：

- 必须和 Dummy / BuyAndHold / Momentum 严格比较
- 不只报告 accuracy
- 必须报告 balanced_accuracy、ROC-AUC、混淆矩阵、预测上涨比例
- 必须进入回测
- 不允许夸大模型能力
- 如果模型不优于基线，如实报告

### Module F：收益率回归 Stage 8

**建议后续新增**：

```
run_stage8_return_regression.py
src/regression_models.py
outputs/v2/tables/SPY_return_regression_metrics.csv
outputs/v2/tables/SPY_return_regression_predictions.csv
outputs/v2/figures/SPY_return_regression_predictions.png
outputs/v2/figures/SPY_return_regression_residuals.png
```

**模型**：

- LinearRegression（基线）
- Ridge（正则化）
- RandomForestRegressor（集成）
- GradientBoostingRegressor（集成）

**指标**：

- MAE（平均绝对误差）
- RMSE（均方根误差）
- R²（决定系数）
- IC（Information Coefficient，预测值与真实值的相关系数）
- 方向命中率（预测方向正确的比例）
- 分位数组合回测（按预测收益率分组，比较各组表现）

### Module G：强化学习环境 V2

**增强点**：

- 更规范的 observation（标准化、可配置特征子集）
- 更清楚的 action（支持多级仓位：0%、25%、50%、75%、100%）
- reward = 收益 - 交易成本 - 风险惩罚（可选）
- 训练集、验证集、测试集严格分离
- 与 BuyAndHold、RandomPolicy、Momentum 严格比较
- 后续可选 DQN/PPO/Stable-Baselines3，但当前不实现

### Module H：实验管理与可复现

**建议新增**：

```
configs/
    experiment_config.yaml        # 实验配置
    walk_forward_config.yaml      # WF 配置
    backtest_config.yaml          # 回测配置
src/experiment_logger.py          # 实验日志记录
outputs/v2/runs/                  # 每次运行的独立目录
outputs/v2/EXPERIMENT_INDEX.md    # 实验索引
```

---

## 5. 优先级

### P0（Sprint 1）

- 确认 v2-research-platform 分支
- 建立 V2 规划文档（本文档）
- 新增 walk-forward validation
- 新增 backtest_v2 框架
- 建立 outputs/v2/ 目录结构

### P1（Sprint 2）

- 三分类交易信号
- 收益率回归 Stage 8
- 更多基线策略
- 模型扩展（Ridge、GBM 等）

### P2（后续）

- DQN/PPO
- FinRL 风格重构
- Web dashboard
- 多资产组合

---

## 6. Sprint 1

**时间**：1–2 天

**约束**：

- 不改 V1 stage1–6
- 不改报告
- 不改 public
- 新增 stage7 walk-forward 验证
- 输出 V2 独立结果到 outputs/v2/

**预期新增文件**：

```
docs/V2_RESEARCH_PLATFORM_PLAN.md    # 本文档
src/validation.py                    # walk-forward 验证模块
run_stage7_walk_forward.py           # stage7 运行脚本
outputs/v2/README.md                 # V2 输出说明
outputs/v2/tables/                   # V2 表格
outputs/v2/figures/                  # V2 图表
```

**验收标准**：

- run_stage7_walk_forward.py 可独立运行
- 输出 walk-forward 验证指标
- 不影响 V1 任何文件
- 不影响 public 版

---

## 7. 给下一位代码 agent 的 Sprint 1 实现 Prompt

```
你是 stock-quant-rl-course-design 的 V2 开发 agent。

工作目录：C:\Users\25494\Desktop\大数据课程设计\stock-quant-rl-course-design
当前分支：v2-research-platform

你的任务是实现 Sprint 1：Walk-forward Validation。

严格约束：
1. 不要修改 run_stage1.py 到 run_stage6.py
2. 不要修改 src/ 下已有文件（ml_models.py、backtest.py 等）
3. 不要修改 docs/ 下的报告文件
4. 不要进入 public 版目录
5. 不要 push
6. 不要夸大模型能力

允许新增：
- src/validation.py
- run_stage7_walk_forward.py
- outputs/v2/ 目录及子目录
- outputs/v2/README.md

实现要求：

1. 读取 data/features/SPY_features_2015_2025.csv（2707 行 × 23 列）
2. 构造 target_direction 标签（next_return > 0 → 1，否则 0）
3. 实现 expanding window walk-forward validation：
   - 初始训练窗口：前 500 行
   - 步长：100 行
   - 每个窗口：训练集早于测试集
   - 每个窗口独立训练 LogisticRegression 和 RandomForestClassifier
   - 每个窗口评估 accuracy、balanced_accuracy、roc_auc、预测上涨比例
4. 汇总跨窗口指标的均值和标准差
5. 同时评估 DummyMostFrequent 基线
6. 保存结果到 outputs/v2/tables/
7. 绘制 ROC-AUC 跨窗口变化图到 outputs/v2/figures/
8. 读取 outputs/tables/SPY_market_returns.csv 计算策略收益
9. 为每个模型计算策略收益、Sharpe、最大回撤
10. 打印实验摘要

不要写"模型表现很好"。如果模型不优于 Dummy 基线，如实报告。
```

---

## 8. V2 与 V1 的关系

| 项目 | V1 | V2 |
|------|----|----|
| 目标 | 课程作业完成 | 研究型实验平台 |
| 验证方式 | 单次 80/20 切分 | Walk-forward validation |
| 回测 | 简化版（手续费 0.1%） | 增强版（手续费 + 滑点 + 交易明细） |
| 基线策略 | 6 个 | 9+ 个 |
| 预测任务 | 二分类方向 | 二分类 + 三分类 + 收益率回归 |
| 模型 | LR + RF + LSTM | + Ridge + GBM + MLP |
| 强化学习 | Tabular Q-learning | 增强环境 + 后续可选 DQN/PPO |
| 输出目录 | outputs/ | outputs/v2/ |
| 分支 | master | v2-research-platform |

---

> 本文档为 V2 规划文档，不用于最终提交。
> V2 开发不应破坏 V1 课程提交结果。
