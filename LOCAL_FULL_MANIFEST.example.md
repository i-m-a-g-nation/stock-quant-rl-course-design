# 本地完整开发版文件清单示例

以下是本地完整开发版（local full version）中可能包含的文件清单，仅供参考。

## 数据文件
- `data/raw/SPY_raw_2015_2025.csv`
- `data/processed/SPY_clean_2015_2025.csv`
- `data/features/SPY_features_2015_2025.csv`

## 模型文件
- `outputs/models/SPY_logistic_regression.joblib`
- `outputs/models/SPY_random_forest_classifier.joblib`
- `outputs/models/SPY_lstm_model.pt`

## 输出表格
- `outputs/tables/SPY_data_quality_report.csv`
- `outputs/tables/SPY_feature_summary.csv`
- `outputs/tables/SPY_ml_baseline_metrics.csv`
- `outputs/tables/SPY_ml_test_predictions.csv`
- `outputs/tables/SPY_ml_feature_importance.csv`
- `outputs/tables/SPY_market_returns.csv`
- `outputs/tables/SPY_risk_metrics.csv`
- `outputs/tables/SPY_prediction_direction_analysis.csv`
- `outputs/tables/SPY_strategy_backtest.csv`
- `outputs/tables/SPY_strategy_metrics.csv`
- `outputs/tables/SPY_lstm_metrics.csv`
- `outputs/tables/SPY_lstm_test_predictions.csv`
- `outputs/tables/SPY_rl_env_baseline_metrics.csv`
- `outputs/tables/SPY_rl_env_baseline_history.csv`
- `outputs/tables/SPY_rl_qlearning_metrics.csv`
- `outputs/tables/SPY_rl_qlearning_history.csv`
- `outputs/tables/SPY_rl_qlearning_training_log.csv`

## 输出图表
- `outputs/figures/SPY_close_price.png`
- `outputs/figures/SPY_volume.png`
- `outputs/figures/SPY_return_distribution.png`
- `outputs/figures/SPY_strategy_equity_curve.png`
- `outputs/figures/SPY_lstm_training_curve.png`
- `outputs/figures/SPY_rl_env_baseline_equity.png`
- `outputs/figures/SPY_rl_qlearning_equity.png`

## 增强实验输出
- `outputs/tables/SPY_ml_model_metrics_enhanced.csv`
- `outputs/tables/SPY_ml_test_predictions_enhanced.csv`
- `outputs/tables/SPY_ml_confusion_matrices.csv`
- `outputs/tables/SPY_feature_columns_enhanced.csv`
- `outputs/tables/SPY_strategy_backtest_enhanced_metrics.csv`
- `outputs/tables/SPY_strategy_equity_curves_enhanced.csv`
- `outputs/tables/SPY_permutation_importance.csv`
- `outputs/tables/SPY_rl_train_test_summary.csv`
- `outputs/tables/SPY_rl_test_strategy_metrics.csv`
- `outputs/tables/SPY_rl_test_equity_curves.csv`
- `outputs/figures/SPY_ml_metrics_enhanced.png`
- `outputs/figures/SPY_permutation_importance_top15.png`
- `outputs/figures/SPY_strategy_equity_curves_enhanced.png`
- `outputs/figures/SPY_strategy_drawdowns_enhanced.png`
- `outputs/figures/SPY_rl_test_equity_curves.png`
- `outputs/figures/SPY_rl_test_drawdowns.png`

## 其他文件
- `outputs/EXPERIMENT_SUMMARY.md`
- `outputs/REPORT_UPDATE_DRAFT.md`
- `outputs/presentation/*.pptx`
- `outputs/pdf/*.pdf`

## 注意事项
- 以上文件清单仅供参考，实际文件可能有所不同
- 这些文件不包含在 public 版中，需要本地运行生成
- 完整结果可通过运行 `run_stage1.py` 到 `run_stage6.py` 重新生成