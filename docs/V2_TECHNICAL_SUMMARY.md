# V2 Technical Summary

> 分支：v2-research-platform
> Tag: v2.0-stage7-10-research-checkpoint
> 本项目仅用于课程实验、研究验证和工程能力展示，不构成实际投资建议。

---

## 1. 项目定位

本项目 V1 是课程设计版本，完成了 SPY 数据获取、特征工程、监督学习、LSTM、回测、风险评估和 Q-learning 实验。

V2 不再以"提高收益"为唯一目标，而是把项目扩展为研究型量化实验平台，重点验证：

1. 当前价量技术特征是否具有稳定短期预测信号；
2. 不同预测任务（方向分类、收益率回归、三分类信号）是否能改善模型表现；
3. 不同预测周期（1d、5d、10d、20d）是否比下一日更有信号；
4. 不同回测口径（简化 vs non-overlap）是否会影响结论；
5. 复杂模型是否真正优于简单基线。

## 2. V2 方法路线

### Stage 7：Walk-forward 二分类方向预测

**任务**：预测下一交易日涨跌方向（binary classification）。

**方法**：
- Expanding window walk-forward validation（17 folds）
- 初始训练窗口 500 行，测试窗口 126 行
- 模型：DummyMostFrequent / DummyStratified / LogisticRegression / RandomForestClassifier

**结果**：
- LogisticRegression ROC-AUC = 0.4872 ± 0.0589，接近 0.5（随机水平）
- LogisticRegression balanced_accuracy = 0.4987 ± 0.0052
- DummyMostFrequent balanced_accuracy = 0.5000
- 所有模型预测上涨比例接近 100%（LR 97.57%，Dummy 100%）

**结论**：二分类方向信号不稳定，模型接近多数类偏置。

### Stage 8：收益率回归

**任务**：预测 next_return 大小（regression）。

**方法**：
- 与 Stage 7 相同的 walk-forward 框架
- 模型：DummyRegressor / LinearRegression / Ridge / RandomForestRegressor / GradientBoostingRegressor

**结果**：
- 所有模型 R² 为负（-0.01 到 -1.09），预测效果不如均值基线
- 相关系数接近 0（pearson_corr 在 -0.0003 到 0.056 之间）
- DummyRegressor MAE = 0.007622，其他模型 MAE 更高
- 方向命中率：Dummy 55.56%，LR 49.39%，GBM 48.13%

**结论**：next_return 幅度难稳定预测，回归模型未优于 DummyRegressor。

### Stage 9：三分类交易信号

**任务**：上涨 / 震荡不交易 / 下跌（three-class classification）。

**方法**：
- 固定阈值 no-trade zone（threshold = 0.0005, 0.0010, 0.0020）
- 三分类模型，macro-F1 与 balanced_accuracy 评估
- 模型：DummyMostFrequent / DummyStratified / LR / RF / GBM

**结果（threshold=0.0010）**：
- DummyMostFrequent balanced_accuracy = 0.3333（始终预测 up）
- GBM balanced_accuracy = 0.3364，macro_f1 = 0.2976
- RF balanced_accuracy = 0.3237，macro_f1 = 0.2914
- LR 倾向预测 neutral（44.30%），balanced_accuracy = 0.3473

**结论**：
- 引入 no-trade zone 后三分类未优于二分类
- balanced_accuracy 接近随机基线（0.3333）
- macro_f1 较低（0.18-0.33）
- DummyMostFrequent 回测收益最高，主要来自市场趋势

### Stage 10：多周期预测与动态阈值

**任务**：预测 1d、5d、10d、20d horizon 的方向。

**方法**：
- 二分类 + 三分类
- 固定阈值 + 动态波动率阈值（threshold_h = k × rolling_vol_20 × √h，k=0.50）
- walk-forward 验证（17 folds）

**结果**：
- 更长 horizon 使 balanced_accuracy 略有提升（+2-5 个百分点）
- 动态阈值改善了类别平衡，但未改善模型预测能力
- 简化回测的 20d DummyMostFrequent 2110.78% 是重叠收益复利放大
- non-overlap 口径下修正为 9.29%

**结论**：多周期预测略有改善但不稳定，回测口径需谨慎解释。

## 3. 关键横向结论

| 结论 | 证据 | 含义 |
|------|------|------|
| 下一日方向预测信号弱 | Stage 7 ROC-AUC 接近 0.5 | 当前价量特征难以稳定预测短期涨跌 |
| 收益率幅度难预测 | Stage 8 不优于 DummyRegressor，R² 为负 | 回归不能简单替代方向分类 |
| no-trade zone 不足以解决问题 | Stage 9 macro-F1 低，未优于二分类 | 三分类交易逻辑更合理，但信号仍弱 |
| 更长 horizon 有轻微改善 | Stage 10 中 10d/20d 指标略高 | 可继续研究，但不能夸大 |
| 回测口径会显著影响结论 | 20d 重叠收益从 2110.78% 修正到 non-overlap 9.29% | 必须谨慎解释长期 horizon 回测 |
| 复杂模型未稳定优于简单基线 | GBM/RF 在部分配置下略优，但差异不大 | 瓶颈在特征信号，不在模型复杂度 |

## 4. 为什么"模型没赢"也是有价值的结果

1. **金融短期信号本身弱**：有效市场假说认为价格已经反映了所有公开信息，短期方向预测天然困难。

2. **SPY 作为宽基 ETF 长期上涨趋势明显**：DummyMostFrequent / AlwaysLong 容易得到较高回测收益，这本身就是强基线。

3. **accuracy 容易被多数类误导**：当上涨日占比 55% 时，始终预测上涨的 accuracy 已达 55%。balanced_accuracy、AUC、混淆矩阵更能揭示模型真实能力。

4. **回测收益必须结合交易成本、仓位暴露、回撤和基线比较**：单独看收益数字没有意义。

5. **当前结果说明简单技术特征不足以形成稳定交易优势**：这为后续引入外部特征和多资产验证提供了依据。

6. **排除法也是研究方法**：证明"当前方法不行"本身就是有价值的结论，可以指导后续方向。

## 5. 当前项目优势

1. **阶段化实验清晰**：Stage 7-10 各有独立目标和结论
2. **每个 stage 有独立脚本**：`run_stage7_walk_forward.py` 到 `run_stage10_multi_horizon_signal.py`
3. **保留 V1 课程版与 V2 研究版**：互不干扰
4. **有 walk-forward 验证**：避免单次 80/20 切分的偶然性
5. **有 Dummy / 基线模型**：始终与简单基线对比
6. **有分类、回归、三分类、多周期多个任务**：全面验证不同建模方式
7. **主动识别了重叠收益回测风险**：补充了 non-overlap 口径
8. **结论克制**：不夸大模型能力，不构成投资建议

## 6. 当前局限

1. **当前主要使用价量技术特征**：没有宏观变量、VIX、利率、行业 ETF、情绪数据
2. **标的只有 SPY**：未做多资产泛化验证
3. **回测仍是简化版**：未加入真实滑点、税费、流动性冲击
4. **多周期收益存在窗口重叠问题**：已补充 non-overlap 口径，但仍是简化近似
5. **未做严格超参数搜索**：模型使用默认参数
6. **未做滚动训练（rolling retrain）**：模型在整个测试区间不更新

## 7. 下一阶段路线

### P0：整理 V2 展示文档与 README

- 完成 V2 Technical Summary（本文档）
- 完成 V2 Project Showcase
- 更新 README

### P1：引入外部特征（Stage 11）

候选特征：
- VIX 波动率指数
- 10Y Treasury Yield
- Federal Funds Rate
- Dollar Index（DXY）
- Sector ETF returns（XLK、XLF、XLE 等）
- QQQ / DIA / IWM 联动特征
- 市场宽度（涨跌比、新高新低比）

### P2：多资产验证（Stage 12）

标的：
- SPY（S&P 500）
- QQQ（NASDAQ 100）
- DIA（Dow Jones）
- IWM（Russell 2000）
- XLK（科技）
- XLF（金融）
- XLE（能源）

目标：验证模型是否只在 SPY 上表现，还是具有一定跨资产稳定性。

### P3：更稳健回测框架（Stage 13）

补充：
- 手续费 + 滑点
- 持仓约束
- 非重叠 horizon
- 换手率统计
- 仓位暴露分析
- rolling retrain
- benchmark normalized comparison

### P4：模型扩展（Stage 14，优先级最低）

候选：
- SVM
- XGBoost / LightGBM
- MLP
- Transformer-based models

**明确**：模型扩展排在外部特征和多资产验证之后。当前瓶颈是特征信号，不是模型能力。

## 8. 最终判断

当前 V2 已经证明，仅依赖 SPY 价量技术特征，很难构建稳定短期交易信号。更长 horizon 和动态阈值提供了轻微改善迹象，但不足以支持真实交易结论。下一阶段应优先扩展信息源和验证范围，而不是继续盲目堆叠模型。

---

**免责声明**：本项目仅用于课程实验、研究验证和工程能力展示，不构成实际投资建议。
