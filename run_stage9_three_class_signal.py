"""
=============================================================================
阶段 9: 三分类交易信号 Walk-forward 验证 —— 一键运行脚本（V2）
=============================================================================
流程:
  1. 读取 data/features/SPY_features_2015_2025.csv
  2. 对 threshold = [0.0005, 0.0010, 0.0020] 分别运行三分类实验
  3. 标签: 1=明显上涨(持仓), 0=震荡(不交易), -1=明显下跌(空仓)
  4. 评估 DummyMostFrequent / DummyStratified / LR / RF / GBM
  5. 输出分类指标、混淆矩阵、回测指标、跨 fold 汇总

用法:  python run_stage9_three_class_signal.py

约束:
  - 不修改 run_stage1.py 到 run_stage8_return_regression.py
  - 不修改 src/ 下已有文件
  - 输出到 outputs/v2/
=============================================================================
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.three_class_validation import run_three_class_walk_forward

logger = setup_logging("stage9")

FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "SPY_features_2015_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2"

INITIAL_TRAIN_SIZE = 500
TEST_SIZE = 126
TRANSACTION_COST = 0.0005
THRESHOLDS = [0.0005, 0.0010, 0.0020]


def run_stage9():
    logger.info("=" * 60)
    logger.info("  阶段 9: 三分类交易信号 Walk-forward 验证 (V2)")
    logger.info("=" * 60)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"特征文件不存在: {FEATURE_PATH}")

    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    models_config = {
        "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
        "DummyStratified": DummyClassifier(strategy="stratified", random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5, class_weight="balanced"),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    }

    logger.info("\n>>> 运行三分类 Walk-forward 验证")
    logger.info("Thresholds: %s", THRESHOLDS)
    logger.info("模型: %s", list(models_config.keys()))

    summary_df, results_df = run_three_class_walk_forward(
        feature_path=str(FEATURE_PATH),
        thresholds=THRESHOLDS,
        models_config=models_config,
        initial_train_size=INITIAL_TRAIN_SIZE,
        test_size=TEST_SIZE,
        transaction_cost=TRANSACTION_COST,
        output_dir=str(OUTPUT_DIR),
    )

    # 打印摘要
    logger.info("\n>>> 实验摘要")

    if len(summary_df) > 0:
        for threshold in THRESHOLDS:
            t_data = summary_df[summary_df["threshold"] == threshold]
            if len(t_data) == 0:
                continue
            logger.info("\n--- Threshold = %.4f ---", threshold)
            for _, row in t_data.iterrows():
                logger.info("  [%s] n_folds=%d", row["model"], row["n_folds"])
                logger.info("    accuracy:     %.4f ± %.4f", row.get("accuracy_mean", 0), row.get("accuracy_std", 0))
                logger.info("    bal_acc:      %.4f ± %.4f", row.get("balanced_accuracy_mean", 0), row.get("balanced_accuracy_std", 0))
                logger.info("    macro_f1:     %.4f ± %.4f", row.get("macro_f1_mean", 0), row.get("macro_f1_std", 0))
                logger.info("    pred_down:    %.4f  pred_neutral: %.4f  pred_up: %.4f",
                             row.get("pred_down_rate_mean", 0), row.get("pred_neutral_rate_mean", 0), row.get("pred_up_rate_mean", 0))
                logger.info("    total_return: %s  sharpe: %s  trades: %.1f",
                             _fmt(row.get("total_return_mean"), row.get("total_return_std")),
                             _fmt(row.get("sharpe_mean"), row.get("sharpe_std")),
                             row.get("trades_mean", 0))

    logger.info("\n" + "=" * 60)
    logger.info("  阶段 9 全部完成！")
    logger.info("  输出目录:  outputs/v2/")
    logger.info("  汇总表:    outputs/v2/tables/SPY_three_class_threshold_summary.csv")
    logger.info("  分类明细:  outputs/v2/tables/SPY_three_class_metrics.csv")
    logger.info("  回测明细:  outputs/v2/tables/SPY_three_class_backtest_metrics.csv")
    logger.info("  混淆矩阵: outputs/v2/tables/SPY_three_class_confusion_matrices.csv")
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
        run_stage9()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段 9 运行失败:\n%s", traceback.format_exc())
        sys.exit(1)
