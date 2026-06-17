# V2 Project Showcase

> 基于机器学习与强化学习的股票交易策略研究型实验平台
> Tag: v2.0-stage7-10-research-checkpoint

---

## 一句话介绍

从课程设计进化为研究型量化实验平台，通过 walk-forward 验证系统性评估价量技术特征在短期股票预测中的有效性。

## V1 → V2 升级

| 维度 | V1（课程版） | V2（研究版） |
|------|-------------|-------------|
| 验证方式 | 单次 80/20 切分 | Walk-forward（17 folds） |
| 预测任务 | 二分类方向 | 二分类 + 回归 + 三分类 + 多周期 |
| 基线对比 | 2 个 Dummy | 多层 Dummy + BuyAndHold |
| 回测 | 单策略 | 多策略 + 多口径（overlap / non-overlap） |
| 结论 | 模型准确率有限 | 系统性证明价量特征信号弱 |

## Stage 7–10 做了什么

| Stage | 任务 | 核心发现 |
|-------|------|----------|
| 7 | Walk-forward 二分类 | ROC-AUC 接近 0.5，方向信号不稳定 |
| 8 | 收益率回归 | R² 为负，不优于均值基线 |
| 9 | 三分类交易信号 | no-trade zone 未改善预测能力 |
| 10 | 多周期预测 | 长 horizon 略有改善但回测口径需谨慎 |

## 最重要的 5 个发现

1. **LogisticRegression 与 DummyMostFrequent 表现一致**：两者 accuracy 均为 0.5886，均 100% 预测上涨，说明模型仅复制了类别分布偏置。

2. **所有回归模型 R² 为负**：预测收益率幅度不如直接用均值，next_return 难以稳定预测。

3. **三分类 no-trade zone 未带来改善**：引入"震荡"类别后 macro_f1 仍低于 0.35，接近随机水平。

4. **20d 回测 2110.78% 是重叠收益放大**：non-overlap 口径下修正为 9.29%，说明长 horizon 回测必须谨慎解释。

5. **DummyMostFrequent 在所有配置下收益最高**：源于 SPY 长期上涨趋势，主动交易策略未能超越简单持有。

## 技术栈

- **语言**：Python 3.10+
- **数据处理**：pandas, numpy
- **机器学习**：scikit-learn（LogisticRegression, RandomForest, GradientBoosting, DummyClassifier）
- **深度学习**：PyTorch（LSTM，V1 部分）
- **可视化**：matplotlib
- **版本控制**：Git（v2-research-platform 分支）

## 项目亮点

1. **阶段化实验架构**：10 个 Stage 各有独立脚本和结论
2. **Walk-forward 验证**：避免单次切分的偶然性
3. **多任务覆盖**：分类、回归、三分类、多周期
4. **多层基线**：DummyMostFrequent / DummyStratified / BuyAndHold
5. **回测口径意识**：主动识别并修正重叠收益问题
6. **结论克制**：不夸大模型能力，明确局限性

## 局限性

1. 仅使用 SPY 价量技术特征，无外部信息源
2. 单一标的，未做多资产泛化验证
3. 回测为简化版，未考虑真实交易摩擦
4. 未做滚动训练（rolling retrain）
5. 未做严格超参数搜索

## 下一步路线

```
P1: 引入外部特征（VIX、利率、行业 ETF）
P2: 多资产验证（QQQ、DIA、IWM、Sector ETFs）
P3: 更稳健回测框架（滑点、仓位约束、rolling retrain）
P4: 模型扩展（SVM、XGBoost），优先级最低
```

---

> 本项目仅用于课程实验、研究验证和工程能力展示，不构成实际投资建议。
