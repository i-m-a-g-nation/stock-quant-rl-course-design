"""
=============================================================================
阶段 8: 收益率回归 Walk-forward 验证 —— 一键运行脚本（V2）
=============================================================================
流程:
  1. 读取 data/features/SPY_features_2015_2025.csv
  2. 构造 next_return 目标（下一交易日收益率）
  3. Expanding window walk-forward 验证
  4. 评估 DummyRegressor / LinearRegression / Ridge / RF / GBM
  5. 输出回归指标、方向命中率、基于预测收益率的简单回测
  6. 绘制 RMSE、方向命中率、收益跨 fold 变化图

用法:  python run_stage8_return_regression.py

约束:
  - 不修改 run_stage1.py 到 run_stage7_walk_forward.py
  - 不修改 src/ 下已有文件（validation.py、ml_models.py 等）
  - 输出到 outputs/v2/
=============================================================================
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.regression_validation import run_return_regression_walk_forward

logger = setup_logging("stage8")

FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "SPY_features_2015_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2"

# Walk-forward 参数
INITIAL_TRAIN_SIZE = 500
TEST_SIZE = 126
THRESHOLD = 0.0005       # 信号阈值（覆盖交易成本）
TRANSACTION_COST = 0.0005


def run_stage8():
    logger.info("=" * 60)
    logger.info("  阶段 8: 收益率回归 Walk-forward 验证 (V2)")
    logger.info("=" * 60)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"特征文件不存在: {FEATURE_PATH}\n请先运行 python run_stage2.py"
        )

    # 模型配置
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    models_config = {
        "DummyRegressor_mean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=42, max_depth=5,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }

    logger.info("\n>>> [Step 1/2] 运行收益率回归 Walk-forward 验证")
    logger.info("初始训练窗口: %d 行", INITIAL_TRAIN_SIZE)
    logger.info("测试窗口大小: %d 行", TEST_SIZE)
    logger.info("信号阈值: %.4f%%", THRESHOLD * 100)
    logger.info("交易成本: %.4f%%", TRANSACTION_COST * 100)
    logger.info("模型: %s", list(models_config.keys()))

    summary_df, results_df = run_return_regression_walk_forward(
        feature_path=str(FEATURE_PATH),
        models_config=models_config,
        initial_train_size=INITIAL_TRAIN_SIZE,
        test_size=TEST_SIZE,
        threshold=THRESHOLD,
        transaction_cost=TRANSACTION_COST,
        horizon=1,
        output_dir=str(OUTPUT_DIR),
    )

    # 打印摘要
    logger.info("\n>>> [Step 2/2] 实验摘要")

    if len(summary_df) > 0:
        logger.info("\n--- 回归指标跨 fold 汇总 ---")
        for _, row in summary_df.iterrows():
            logger.info("  [%s] n_folds=%d", row["model"], row["n_folds"])
            logger.info("    MAE:              %.6f ± %.6f", row.get("mae_mean", 0), row.get("mae_std", 0))
            logger.info("    RMSE:             %.6f ± %.6f", row.get("rmse_mean", 0), row.get("rmse_std", 0))
            logger.info("    R²:               %.4f ± %.4f", row.get("r2_mean", 0), row.get("r2_std", 0))
            logger.info("    pearson_corr:     %s", _fmt(row.get("pearson_corr_mean"), row.get("pearson_corr_std")))
            logger.info("    spearman_corr:    %s", _fmt(row.get("spearman_corr_mean"), row.get("spearman_corr_std")))
            logger.info("    direction_acc:    %.4f ± %.4f", row.get("direction_accuracy_mean", 0), row.get("direction_accuracy_std", 0))
            logger.info("    pred_mean:        %.6f ± %.6f", row.get("pred_mean_mean", 0), row.get("pred_mean_std", 0))
            logger.info("    true_mean:        %.6f ± %.6f", row.get("true_mean_mean", 0), row.get("true_mean_std", 0))

        logger.info("\n--- 回测指标跨 fold 汇总 ---")
        for _, row in summary_df.iterrows():
            logger.info("  [%s]", row["model"])
            logger.info("    total_return:     %s", _fmt(row.get("total_return_mean"), row.get("total_return_std")))
            logger.info("    sharpe:           %s", _fmt(row.get("sharpe_mean"), row.get("sharpe_std")))
            logger.info("    max_drawdown:     %s", _fmt(row.get("max_drawdown_mean"), row.get("max_drawdown_std")))
            logger.info("    avg_position:     %.4f ± %.4f", row.get("avg_position_mean", 0), row.get("avg_position_std", 0))

    logger.info("\n" + "=" * 60)
    logger.info("  阶段 8 全部完成！")
    logger.info("  输出目录:  outputs/v2/")
    logger.info("  汇总表:    outputs/v2/tables/SPY_return_regression_summary.csv")
    logger.info("  回归明细:  outputs/v2/tables/SPY_return_regression_metrics.csv")
    logger.info("  回测明细:  outputs/v2/tables/SPY_return_regression_backtest_metrics.csv")
    logger.info("  RMSE:      outputs/v2/figures/SPY_return_regression_rmse_by_fold.png")
    logger.info("  Dir Acc:   outputs/v2/figures/SPY_return_regression_direction_accuracy_by_fold.png")
    logger.info("  Return:    outputs/v2/figures/SPY_return_regression_total_return_by_fold.png")
    logger.info("  Pred vs T: outputs/v2/figures/SPY_return_regression_pred_vs_true.png")
    logger.info("=" * 60)


def _fmt(mean, std):
    import math
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "NaN"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


if __name__ == "__main__":
    try:
        run_stage8()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段 8 运行失败，错误详情:\n%s", traceback.format_exc())
        sys.exit(1)
