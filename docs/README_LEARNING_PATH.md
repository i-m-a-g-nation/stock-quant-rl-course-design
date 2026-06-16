# 学习顺序与复习计划

> 适用状态：项目已经完成，但还没有完全理解。  
> 最推荐顺序：先讲清全局，再理解指标，然后看代码，最后背问答。

## 1. 文档阅读顺序

| 顺序 | 文档 | 目标 |
|---:|---|---|
| 1 | `ONE_PAGE_CHEATSHEET.md` | 先获得项目全貌 |
| 2 | `PROJECT_FULL_EXPLANATION.md` | 从零理解六个阶段与真实结果 |
| 3 | `FINANCE_BASICS.md` | 补齐金融基础 |
| 4 | `ALGORITHM_NOTES.md` | 理解 LR、RF、LSTM、Q-learning 和指标 |
| 5 | `CODE_WALKTHROUGH.md` | 把概念对应到代码 |
| 6 | `DEFENSE_QA.md` | 练习口头表达 |
| 7 | `GLOSSARY.md` | 查漏补缺 |

## 2. 两小时紧急复习

适合答辩前时间很紧的情况。

### 第 0 到 20 分钟：先看什么

阅读 `ONE_PAGE_CHEATSHEET.md` 两遍。第一遍看路线，第二遍重点记数字。

必须记住：

```text
数据：2766 行清洗数据，2707 行特征数据
LR：0.5886，但全部预测上涨
RF：0.4188，recall 0.0282
LSTM：0.4540，AUC 0.4916
SPY 最大回撤：-34.10%
Q-learning：+6.56%
Buy & Hold：+227.12%
```

### 第 20 到 60 分钟：再看什么

阅读 `PROJECT_FULL_EXPLANATION.md`：

1. 看第 3 节技术路线。
2. 看第 7 到 10 节四类模型与回测。
3. 看第 14 节不一致与风险。
4. 背第 15 节 30 秒版总结。

### 第 60 到 90 分钟：看哪些代码

阅读 `CODE_WALKTHROUGH.md` 的关键片段：

1. 标签构造。
2. 时间顺序切分。
3. 回测 `signal.shift(1)`。
4. LSTM 先 `dropna` 再构造标签。
5. RL 下一期奖励。
6. Q-learning 离散化与更新公式。

### 第 90 到 120 分钟：背哪些问答

优先练习 `DEFENSE_QA.md`：

```text
Q3、Q14、Q26、Q27、Q28、Q35、Q44、Q47、Q56、Q58、Q65
```

### 两小时内可以暂时不深挖

- LSTM 每个门的完整矩阵推导。
- 所有 15 个输入特征的精确数值分布。
- 每个输出 CSV 的全部字段。
- 报告辅助绘图脚本的实现细节。

## 3. 半天复习计划

建议安排 4 小时。

### 第 1 小时：项目与金融基础

1. 阅读 `PROJECT_FULL_EXPLANATION.md`。
2. 阅读 `FINANCE_BASICS.md`。
3. 用自己的话解释 SPY、ETF、收益率、波动率、最大回撤、Sharpe、VaR、CVaR。

检查标准：不看文档也能回答“为什么高收益不等于低风险”。

### 第 2 小时：算法

阅读 `ALGORITHM_NOTES.md`：

1. 逻辑回归的 sigmoid。
2. 随机森林的多树投票。
3. LSTM 的 20 日窗口和门控。
4. Q-learning 的六要素、离散化和更新公式。
5. 分类指标与风险指标表。

检查标准：能解释为什么复杂模型不一定更好。

### 第 3 小时：代码

阅读 `CODE_WALKTHROUGH.md`，然后实际打开：

```text
src/features.py
src/ml_models.py
src/backtest.py
src/lstm_model.py
src/trading_env.py
src/rl_agent.py
```

优先找到：

```text
pct_change
shift(-1)
shift(1)
dropna
_create_sequences
next_idx = current_step + 1
discretize
Q[state][action] += ...
```

### 第 4 小时：模拟答辩

1. 阅读 `DEFENSE_QA.md`。
2. 随机抽取 20 题，每题口头回答 20 到 40 秒。
3. 最后背诵 30 秒版和 1 分钟版项目总结。

### 半天必须会

- 六阶段流程。
- 下一日标签与信号后移。
- LR 全猜上涨的谨慎解释。
- RF 只预测 14 个上涨日。
- LSTM AUC 接近 0.5。
- Q-learning 状态、动作、奖励、更新公式。
- Stage 4 与 Stage 6 成本口径不同。
- RL 没有严格样本外测试。

### 半天可以不背

- 每张图的文件大小。
- 历史遗留文件名。
- 全部 80 多个术语。
- 所有 stage check 历史记录细节。

## 4. 一天完整复习计划

建议分上午、下午、晚上。

### 上午：建立体系

1. 阅读 `PROJECT_FULL_EXPLANATION.md`。
2. 阅读 `FINANCE_BASICS.md`。
3. 对照 `GLOSSARY.md` 查不懂术语。
4. 手写一遍技术路线：

```text
SPY -> 清洗 -> 特征 -> LR/RF -> 风险与回测 -> LSTM -> RL -> Q-learning
```

### 下午：代码与结果

1. 阅读 `CODE_WALKTHROUGH.md`。
2. 打开 6 个 `run_stage*.py`，确认各入口调用关系。
3. 打开 6 个核心实现：

```text
src/data_cleaner.py
src/features.py
src/ml_models.py
src/backtest.py
src/lstm_model.py
src/rl_agent.py
```

4. 查看关键 CSV：

```text
outputs/tables/SPY_ml_baseline_metrics.csv
outputs/tables/SPY_risk_metrics.csv
outputs/tables/SPY_strategy_metrics.csv
outputs/tables/SPY_lstm_metrics.csv
outputs/tables/SPY_rl_qlearning_metrics.csv
```

5. 用自己的话写出每个结果的保守解释。

### 晚上：答辩训练

1. 阅读 `ALGORITHM_NOTES.md`。
2. 对着 `DEFENSE_QA.md` 完成一轮 65 题快速口答。
3. 第二轮只练薄弱题。
4. 最后用 `ONE_PAGE_CHEATSHEET.md` 做 10 分钟复盘。

### 一天必须会

除半天计划内容外，还要会：

- OHLCV 与 Adj Close 区别。
- rolling 为什么产生 NaN。
- LSTM 输入形状约为 `(样本数, 20, 9)`。
- VaR 与 CVaR 区别。
- 买入持有 `trades=0` 的统计口径。
- epsilon 下限是 0.05，但 50 轮后实际是 0.2335。
- 当前 Q 状态按代码重跑为 27 个均被访问。
- 模板适配版报告在 `../report/`，不在项目根目录内。

### 一天仍然不必死背

- 每个函数所有参数。
- 神经网络反向传播的完整推导。
- DQN、PPO 的实现细节，因为项目未实现。
- 所有图片布局细节。

## 5. 答辩必背卡片

### 卡片 1：项目定位

这是课程实验和方法验证，不是实际投资系统，不构成投资建议。

### 卡片 2：结果原则

不能只看 accuracy，必须结合类别分布、precision、recall、F1、风险与回测。

### 卡片 3：未来函数

今天的信息只能用于下一期。Stage 4 用 `shift(1)`，Stage 6 当前动作使用下一期收益结算。

### 卡片 4：复杂模型

复杂模型不必然更好。金融短期收益具有高噪声、弱信号和分布变化。

### 卡片 5：RL 边界

Q-learning 是课程级表格型实现，状态和动作较简化，而且没有严格样本外评价。

## 6. 最可能被继续追问的地方

| 追问点 | 回答重点 |
|---|---|
| LR 为什么看起来最好 | 实际全部预测上涨，accuracy 等于上涨比例 |
| RF 回测为什么风险看起来很低 | RF 基本空仓，541 天中 527 天收益为 0 |
| Stage 4 win_rate 为什么低 | 它是正收益日比例，不是逐笔交易胜率 |
| LSTM 为什么没有提升 | 基础特征弱，市场噪声高，复杂度不保证泛化 |
| RL 是否有数据泄漏风险 | 训练和评价同一历史段，没有严格样本外测试 |
| 买入持有 trades 为什么是 0 | 统计内部仓位变化，不计初始建仓 |
| epsilon 是否衰减到 0.05 | 没有；0.05 是下限，50 轮后为 0.2335 |
| Q 表访问多少状态 | 当前代码重跑得到 27 个状态均访问 |
| heatmap 从哪里来 | 报告辅助脚本，不是 `run_stage2.py` |
| 报告在哪里 | 工作区同级 `../report/` |

