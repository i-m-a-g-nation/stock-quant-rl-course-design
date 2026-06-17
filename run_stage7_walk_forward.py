"""
=============================================================================
阶段 7: Walk-forward 时间序列验证 —— 一键运行脚本（V2）
=============================================================================
流程:
  1. 读取 data/features/SPY_features_2015_2025.csv
  2. 构造方向标签 (target_direction)
  3. Expanding window walk-forward 验证
  4. 评估 DummyMostFrequent / DummyStratified / LogisticRegression / RandomForestClassifier
  5. 输出分类指标、回测指标、跨 fold 汇总
  6. 绘制 ROC-AUC、Balanced Accuracy、Total Return 跨 fold 变化图

用法:  python run_stage7_walk_forward.py

约束:
  - 不修改 run_stage1.py 到 run_stage6.py
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
from src.validation import run_walk_forward

logger = setup_logging("stage7")

FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "SPY_features_2015_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2"

# Walk-forward 参数
INITIAL_TRAIN_SIZE = 500   # 初始训练窗口大小
TEST_SIZE = 126            # 每个测试窗口大小（约半年交易日）
TRANSACTION_COST = 0.0005  # 交易成本 0.05%


def run_stage7():
    logger.info("=" * 60)
    logger.info("  阶段 7: Walk-forward 时间序列验证 (V2)")
    logger.info("=" * 60)

    # 检查特征文件
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"特征文件不存在: {FEATURE_PATH}\n请先运行 python run_stage2.py"
        )

    # 模型配置
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    models_config = {
        "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
        "DummyStratified": DummyClassifier(strategy="stratified", random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=42, max_depth=5,
        ),
    }

    logger.info("\n>>> [Step 1/2] 运行 Walk-forward 验证")
    logger.info("初始训练窗口: %d 行", INITIAL_TRAIN_SIZE)
    logger.info("测试窗口大小: %d 行", TEST_SIZE)
    logger.info("交易成本: %.4f%%", TRANSACTION_COST * 100)
    logger.info("模型: %s", list(models_config.keys()))

    summary_df, results_df = run_walk_forward(
        feature_path=str(FEATURE_PATH),
        models_config=models_config,
        initial_train_size=INITIAL_TRAIN_SIZE,
        test_size=TEST_SIZE,
        transaction_cost=TRANSACTION_COST,
        output_dir=str(OUTPUT_DIR),
    )

    # 打印摘要
    logger.info("\n>>> [Step 2/2] 实验摘要")

    if len(summary_df) > 0:
        logger.info("\n--- 分类指标跨 fold 汇总 ---")
        for _, row in summary_df.iterrows():
            logger.info("  [%s] n_folds=%d", row["model"], row["n_folds"])
            logger.info("    accuracy:       %.4f ± %.4f", row.get("accuracy_mean", 0), row.get("accuracy_std", 0))
            logger.info("    balanced_acc:   %.4f ± %.4f", row.get("balanced_accuracy_mean", 0), row.get("balanced_accuracy_std", 0))
            logger.info("    roc_auc:        %s", _fmt_auc(row.get("roc_auc_mean"), row.get("roc_auc_std")))
            logger.info("    pred_pos_rate:  %.4f ± %.4f", row.get("positive_rate_pred_mean", 0), row.get("positive_rate_pred_std", 0))

        logger.info("\n--- 回测指标跨 fold 汇总 ---")
        for _, row in summary_df.iterrows():
            logger.info("  [%s]", row["model"])
            logger.info("    total_return:   %s", _fmt_metric(row.get("total_return_mean"), row.get("total_return_std")))
            logger.info("    sharpe:         %s", _fmt_metric(row.get("sharpe_mean"), row.get("sharpe_std")))
            logger.info("    max_drawdown:   %s", _fmt_metric(row.get("max_drawdown_mean"), row.get("max_drawdown_std")))

    logger.info("\n" + "=" * 60)
    logger.info("  阶段 7 全部完成！")
    logger.info("  输出目录:  outputs/v2/")
    logger.info("  汇总表:    outputs/v2/tables/SPY_walk_forward_summary.csv")
    logger.info("  分类明细:  outputs/v2/tables/SPY_walk_forward_classification_metrics.csv")
    logger.info("  回测明细:  outputs/v2/tables/SPY_walk_forward_backtest_metrics.csv")
    logger.info("  ROC-AUC:   outputs/v2/figures/SPY_walk_forward_roc_auc_by_fold.png")
    logger.info("  Bal Acc:   outputs/v2/figures/SPY_walk_forward_balanced_accuracy_by_fold.png")
    logger.info("  Return:    outputs/v2/figures/SPY_walk_forward_total_return_by_fold.png")
    logger.info("=" * 60)


def _fmt_auc(mean, std):
    """格式化 AUC 值（处理 NaN）"""
    import math
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "NaN"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _fmt_metric(mean, std):
    """格式化指标值"""
    import math
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "NaN"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


if __name__ == "__main__":
    try:
        run_stage7()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段 7 运行失败，错误详情:\n%s", traceback.format_exc())
        sys.exit(1)
