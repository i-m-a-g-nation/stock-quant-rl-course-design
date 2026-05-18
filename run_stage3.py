"""
=============================================================================
阶段3: 机器学习基线模型 —— 一键运行脚本
=============================================================================
流程:
  1. 读取 data/features/SPY_features_2015_2025.csv
  2. 构造方向标签 (target_direction)
  3. 时间序列划分训练/测试集 (80/20)
  4. 训练 LogisticRegression + RandomForestClassifier
  5. 评估 accuracy / precision / recall / f1
  6. 保存测试集预测结果
  7. 保存特征重要性
  8. 保存模型到 outputs/models/

用法:  python run_stage3.py
=============================================================================
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.ml_models import (
    load_feature_data, build_labels,
    train_test_split_time_series, train_and_evaluate,
    save_metrics, save_predictions, save_feature_importance,
)

logger = setup_logging("stage3")


def run_stage3():
    """阶段3 主流程"""
    logger.info("=" * 60)
    logger.info("  阶段3: 机器学习基线模型")
    logger.info("=" * 60)

    logger.info("\n>>> [Step 1/4] 加载特征数据")
    df = load_feature_data()

    logger.info("\n>>> [Step 2/4] 构造标签 + 划分训练/测试集")
    df = build_labels(df)
    train, test = train_test_split_time_series(df)

    logger.info("\n>>> [Step 3/4] 训练模型 + 评估")
    metrics, fitted_models, predictions = train_and_evaluate(train, test)
    save_metrics(metrics)
    save_predictions(predictions)

    logger.info("\n>>> [Step 4/4] 特征重要性分析")
    save_feature_importance(fitted_models)

    logger.info("\n--- 评估指标摘要 ---")
    for _, row in metrics.iterrows():
        logger.info("  %s | %s = %s", row["模型"], row["指标"], row["值"])

    logger.info("\n" + "=" * 60)
    logger.info("  阶段3 全部完成！")
    logger.info("  评估指标:     outputs/tables/SPY_ml_baseline_metrics.csv")
    logger.info("  测试预测:     outputs/tables/SPY_ml_test_predictions.csv")
    logger.info("  特征重要性:   outputs/tables/SPY_ml_feature_importance.csv")
    logger.info("  模型:         outputs/models/SPY_logistic_regression.joblib")
    logger.info("  模型:         outputs/models/SPY_random_forest_classifier.joblib")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_stage3()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段3 运行失败，错误详情:\n%s", traceback.format_exc())
        sys.exit(1)
