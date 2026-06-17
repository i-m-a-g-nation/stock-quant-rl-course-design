# V2 Outputs

> V2 研究型实验平台输出目录
> 本目录仅保存 V2 研究型实验结果，不覆盖 V1 课程版 outputs。

## 目录说明

| 子目录 | 说明 |
|--------|------|
| `tables/` | CSV 格式的实验指标和明细 |
| `figures/` | 图表（PNG 格式） |

## 当前实验

### Stage 7: Walk-forward 时间序列验证（方向分类）

将 V1 的单次 80/20 切分升级为 expanding window walk-forward validation。
每个窗口独立训练、独立评估，输出跨窗口均值和标准差，衡量模型稳定性。

- 运行命令：`python run_stage7_walk_forward.py`
- 主线任务：预测下一交易日涨跌方向（二分类）
- 模型：DummyMostFrequent / DummyStratified / LogisticRegression / RandomForestClassifier

### Stage 8: 收益率回归 Walk-forward 验证

预测下一交易日收益率 next_return（回归任务），与 Stage 7 的方向分类形成对比。
使用相同的 walk-forward 框架，评估回归模型和基于预测收益率的交易信号。

- 运行命令：`python run_stage8_return_regression.py`
- 主线任务：预测 next_return（日收益率）
- 模型：DummyRegressor / LinearRegression / Ridge / RandomForestRegressor / GradientBoostingRegressor

### Stage 9: 三分类交易信号 Walk-forward 验证

引入"震荡/不交易"类别，将方向预测从二分类扩展为三分类。
比较不同 no-trade zone 宽度（threshold=0.0005/0.0010/0.0020）对类别分布和交易信号的影响。

- 运行命令：`python run_stage9_three_class_signal.py`
- 主线任务：三分类方向预测（-1=下跌, 0=震荡, 1=上涨）
- 模型：DummyMostFrequent / DummyStratified / LogisticRegression / RandomForestClassifier / GradientBoostingClassifier

### Stage 10: 多周期预测与动态阈值实验

预测周期从 1 日扩展到 1/5/10/20 日，比较二分类和三分类任务，
测试固定阈值和波动率动态阈值对类别分布和模型表现的影响。

- 运行命令：`python run_stage10_multi_horizon_signal.py`
- 主线任务：多周期方向预测 + 动态阈值三分类
- 模型：DummyMostFrequent / DummyStratified / LR / RF / GBM

## 后续规划

| Stage | 说明 | 状态 |
|-------|------|------|
| Stage 7 | Walk-forward 验证（方向分类） | ✅ 完成 |
| Stage 8 | 收益率回归 | ✅ 完成 |
| Stage 9 | 三分类交易信号 | ✅ 完成 |
| Stage 10 | 多周期预测与动态阈值 | ✅ 完成 |

## V2 阶段总结

| 文件 | 说明 |
|------|------|
| `V2_STAGE7_TO_STAGE10_REVIEW.md` | V2 阶段检查报告（含横向总结表） |
| `Sprint1_WALK_FORWARD_SUMMARY.md` | Stage 7 实验总结 |
| `Sprint2_RETURN_REGRESSION_SUMMARY.md` | Stage 8 实验总结 |
| `Sprint3_THREE_CLASS_SIGNAL_SUMMARY.md` | Stage 9 实验总结 |
| `Sprint4_MULTI_HORIZON_SIGNAL_SUMMARY.md` | Stage 10 实验总结 |
| `tables/SPY_v2_stage_summary.csv` | 四阶段横向对比表 |

## 注意事项

1. V2 输出不覆盖 V1 课程版 outputs
2. 当前主线仍是下一交易日涨跌方向分类
3. 结果仅用于课程扩展和研究验证
4. 不构成投资建议
5. 多周期回测存在重叠收益放大问题，已补充 non-overlap 口径
