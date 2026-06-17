# V2 Stage 7-10 Review

> 分支：v2-research-platform
> 生成时间：2026-06-17
> 本项目仅用于课程实验和方法验证，不构成投资建议。

---

## 1. V2 当前阶段目标

V2 的目标不是提高收益，而是验证当前价量技术特征在不同预测任务、不同预测周期下是否具有稳定短期信号。通过 walk-forward 验证、多模型对比、多周期实验，客观评估现有特征和方法的局限性。

## 2. Stage 7：Walk-forward 二分类方向预测

**配置**：17 folds，expanding window（初始 500 行，步长 126 行）

**模型**：DummyMostFrequent / DummyStratified / LogisticRegression / RandomForestClassifier

**结果**：
- LogisticRegression ROC-AUC = 0.4872 ± 0.0589，接近 0.5（随机水平）
- LogisticRegression balanced_accuracy = 0.4987 ± 0.0052
- DummyMostFrequent balanced_accuracy = 0.5000
- 所有模型预测上涨比例接近 100%（LR 97.57%，Dummy 100%）

**结论**：二分类方向信号不稳定，模型接近多数类偏置。

## 3. Stage 8：收益率回归

**配置**：17 folds，与 Stage 7 相同的 walk-forward 框架

**模型**：DummyRegressor / LinearRegression / Ridge / RandomForestRegressor / GradientBoostingRegressor

**结果**：
- 所有模型 R² 为负（-0.01 到 -1.09），预测效果不如均值基线
- 相关系数接近 0（pearson_corr 在 -0.0003 到 0.056 之间）
- DummyRegressor MAE = 0.007622，其他模型 MAE 更高
- 方向命中率：Dummy 55.56%，LR 49.39%，GBM 48.13%

**结论**：next_return 幅度难稳定预测，回归模型未优于 DummyRegressor。

## 4. Stage 9：三分类交易信号

**配置**：3 thresholds（0.0005, 0.0010, 0.0020），17 folds

**模型**：DummyMostFrequent / DummyStratified / LR / RF / GBM

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

## 5. Stage 10：多周期预测与动态阈值

**配置**：horizons = [1, 5, 10, 20]，binary + three_class，17 folds

**动态阈值**：threshold_h = k × rolling_vol_20 × √h，k=0.50

**标签分布变化**：
- 二分类：h=1 时 up=55.0%，h=20 时 up=69.1%（长期上涨趋势）
- 三分类（动态阈值）：neutral 从 h=1 的 43.4% 降至 h=20 的 32.0%

**二分类 balanced_accuracy**：

| horizon | Dummy | GBM | RF |
|---------|-------|-----|-----|
| 1d | 0.5000 | 0.5039 | 0.4973 |
| 5d | 0.5000 | 0.5278 | 0.5108 |
| 10d | 0.5000 | 0.5495 | 0.5579 |
| 20d | 0.5000 | 0.5579 | 0.5468 |

**回测收益（non-overlap 口径）**：

| horizon | Dummy | GBM | RF |
|---------|-------|-----|-----|
| 1d | 7.92% | 0.41% | 0.71% |
| 5d | 8.26% | 8.47% | 3.50% |
| 10d | 8.26% | 4.75% | 3.80% |
| 20d | 9.29% | 8.19% | 5.06% |

**关键发现**：
- 更长 horizon 使 balanced_accuracy 略有提升（+2-5 个百分点），但仍接近随机水平
- 动态阈值改善了类别平衡，但未改善模型预测能力
- 简化回测的 20d DummyMostFrequent 2110.78% 是重叠收益复利放大，non-overlap 口径下仅为 9.29%
- DummyMostFrequent 在 non-overlap 口径下仍为最高，源于市场趋势

## 6. 横向总结表

| Stage | 任务 | 验证方式 | 最佳观察 | 主要问题 | 结论 |
|-------|------|----------|----------|----------|------|
| 7 | 二分类方向 | walk-forward | 部分模型略高于随机 | 不稳定、接近多数类偏置 | 信号弱 |
| 8 | 收益率回归 | walk-forward | RMSE 数值不大 | R² 为负、相关性弱 | 不优于均值 |
| 9 | 三分类信号 | walk-forward | no-trade zone 降低交易 | macro-F1 低、接近随机 | 不优于二分类 |
| 10 | 多周期预测 | walk-forward | 10d/20d 略改善 | 长周期回测口径需谨慎 | 有探索价值但不稳定 |

## 7. 总体结论

1. **当前仅靠价量技术特征，难以形成稳定短期交易信号**。四个 Stage 的实验一致表明，模型在 walk-forward 验证下的表现接近或低于简单基线。

2. **更长 horizon 有轻微改善，但不足以支持稳定交易**。10d/20d 的 balanced_accuracy 略高于 1d，但改善幅度有限（2-5 个百分点），且仍接近随机水平。

3. **动态阈值改善类别分布，但未明显改善预测能力**。引入波动率动态阈值后 neutral 比例更平衡，但模型的 macro_f1 和 balanced_accuracy 未因此提升。

4. **复杂模型未稳定优于简单基线**。GBM 和 RF 在部分配置下略优于 Dummy，但差异不大且标准差较大，跨窗口表现不稳定。

5. **后续不应继续盲目堆模型**。当前瓶颈在于特征本身的信号质量，而非模型复杂度。

## 8. 下一阶段建议

### P0（立即）

- Stage 10 选择性 commit
- 保留 non-overlap 回测作为更保守的参考
- 在报告中明确说明多周期回测口径限制

### P1（优先）

引入外部特征：
- VIX 波动率指数
- SPY 成交量扩展特征
- 宏观利率（如 10 年期国债收益率）
- 市场宽度（如涨跌比）
- Sector ETF 相对强弱

### P2（后续）

多资产验证：
- QQQ（科技）
- DIA（工业）
- IWM（小盘）
- Sector ETFs

### P3（可选）

模型扩展（优先级最低）：
- SVM
- XGBoost / LightGBM
- MLP

**明确**：模型扩展排在外部特征和多资产验证之后。当前瓶颈是特征信号，不是模型能力。

## 9. 回测口径说明

本实验包含两种回测口径：

1. **Simplified（简化口径）**：对所有测试样本的 future_h_return 做 cumprod。h>1 时存在重叠收益复利放大，total_return 仅用于同口径模型比较，不可解释为真实交易收益。

2. **Non-overlap（非重叠口径）**：每 h 天取一个样本，避免重叠收益。更保守，更接近实际可执行收益。

两种口径的 DummyMostFrequent 对比：

| horizon | simplified | non_overlap |
|---------|-----------|-------------|
| 1d | 7.92% | 7.92% |
| 5d | 64.98% | 8.26% |
| 10d | 260.79% | 8.26% |
| 20d | 2110.78% | 9.29% |

---

> 本文档为 V2 Stage 7-10 阶段检查报告。
> 本项目仅用于课程实验和方法验证，不构成投资建议。
