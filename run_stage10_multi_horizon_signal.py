"""
=============================================================================
阶段 10: 多周期预测与动态阈值实验 —— 一键运行脚本（V2）
=============================================================================
流程:
  1. 读取 data/features/SPY_features_2015_2025.csv
  2. 对 horizon = [1, 5, 10, 20] 运行二分类和三分类实验
  3. 三分类支持固定阈值和波动率动态阈值
  4. 评估 DummyMostFrequent / DummyStratified / LR / RF / GBM
  5. 输出分类指标、回测指标、标签分布、跨 fold 汇总

用法:  python run_stage10_multi_horizon_signal.py

约束:
  - 不修改 run_stage1.py 到 run_stage9_three_class_signal.py
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
from src.multi_horizon_validation import run_multi_horizon_walk_forward

logger = setup_logging("stage10")

FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "SPY_features_2015_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2"

INITIAL_TRAIN_SIZE = 500
TEST_SIZE = 126
TRANSACTION_COST = 0.0005
HORIZONS = [1, 5, 10, 20]
TASK_TYPES = ["binary", "three_class"]
FIXED_THRESHOLDS = {1: 0.001, 5: 0.003, 10: 0.005, 20: 0.010}
VOL_K_VALUES = [0.50]


def run_stage10():
    logger.info("=" * 60)
    logger.info("  阶段 10: 多周期预测与动态阈值实验 (V2)")
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

    logger.info("\n>>> 运行多周期 Walk-forward 验证")
    logger.info("Horizons: %s", HORIZONS)
    logger.info("Task types: %s", TASK_TYPES)
    logger.info("Fixed thresholds: %s", FIXED_THRESHOLDS)
    logger.info("Vol k values: %s", VOL_K_VALUES)
    logger.info("模型: %s", list(models_config.keys()))

    summary_df, results_df, label_dist_df = run_multi_horizon_walk_forward(
        feature_path=str(FEATURE_PATH),
        horizons=HORIZONS,
        task_types=TASK_TYPES,
        fixed_thresholds=FIXED_THRESHOLDS,
        vol_k_values=VOL_K_VALUES,
        models_config=models_config,
        initial_train_size=INITIAL_TRAIN_SIZE,
        test_size=TEST_SIZE,
        transaction_cost=TRANSACTION_COST,
        output_dir=str(OUTPUT_DIR),
    )

    # 打印摘要
    logger.info("\n>>> 实验摘要")

    # 标签分布
    if len(label_dist_df) > 0:
        logger.info("\n--- 标签分布 ---")
        for h in HORIZONS:
            h_data = label_dist_df[label_dist_df["horizon"] == h]
            for task in TASK_TYPES:
                t_data = h_data[h_data["task_type"] == task]
                if len(t_data) == 0:
                    continue
                parts = []
                for _, row in t_data.iterrows():
                    parts.append(f"lbl={int(row['label'])}: {row['count']}({row['ratio']:.1%})")
                logger.info("  h=%d %s: %s", h, task, ", ".join(parts))

    # 二分类结果
    if len(summary_df) > 0:
        binary_df = summary_df[summary_df["task_type"] == "binary"]
        if len(binary_df) > 0:
            logger.info("\n--- 二分类结果 ---")
            for h in HORIZONS:
                h_data = binary_df[binary_df["horizon"] == h]
                if len(h_data) == 0:
                    continue
                logger.info("  Horizon=%d:", h)
                for _, row in h_data.iterrows():
                    logger.info("    [%s] bal_acc=%.4f±%.4f f1=%.4f±%.4f ret=%.4f sharpe=%.4f",
                                row["model"],
                                row.get("balanced_accuracy_mean", 0), row.get("balanced_accuracy_std", 0),
                                row.get("f1_mean", 0), row.get("f1_std", 0),
                                row.get("total_return_mean", 0), row.get("sharpe_mean", 0))

        # 三分类结果（fixed）
        three_fixed = summary_df[(summary_df["task_type"] == "three_class") &
                                 (summary_df["threshold_mode"] == "fixed")]
        if len(three_fixed) > 0:
            logger.info("\n--- 三分类结果（固定阈值） ---")
            for h in HORIZONS:
                h_data = three_fixed[three_fixed["horizon"] == h]
                if len(h_data) == 0:
                    continue
                logger.info("  Horizon=%d (threshold=%.3f):", h, FIXED_THRESHOLDS.get(h, 0.001))
                for _, row in h_data.iterrows():
                    logger.info("    [%s] bal_acc=%.4f macro_f1=%.4f ret=%.4f sharpe=%.4f",
                                row["model"],
                                row.get("balanced_accuracy_mean", 0),
                                row.get("macro_f1_mean", 0),
                                row.get("total_return_mean", 0),
                                row.get("sharpe_mean", 0))

    logger.info("\n" + "=" * 60)
    logger.info("  阶段 10 全部完成！")
    logger.info("  输出目录:  outputs/v2/")
    logger.info("  汇总表:    outputs/v2/tables/SPY_multi_horizon_summary.csv")
    logger.info("  分类明细:  outputs/v2/tables/SPY_multi_horizon_metrics.csv")
    logger.info("  标签分布:  outputs/v2/tables/SPY_multi_horizon_label_distribution.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_stage10()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段 10 运行失败:\n%s", traceback.format_exc())
        sys.exit(1)
