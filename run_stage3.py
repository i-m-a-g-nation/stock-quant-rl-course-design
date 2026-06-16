"""
=============================================================================
阶段3: 机器学习基线模型 —— 一键运行脚本
=============================================================================
流程:
  1. 读取 data/features/SPY_features_2015_2025.csv
  2. 构造方向标签 (target_direction)
  3. 时间序列划分训练/测试集 (80/20)
  4. 训练 DummyClassifier + LogisticRegression + RandomForestClassifier
  5. 评估 accuracy / balanced_accuracy / precision / recall / f1 / roc_auc
  6. 保存测试集预测结果 + 混淆矩阵
  7. 保存特征重要性 + permutation importance
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
    save_enhanced_metrics, save_enhanced_predictions, save_confusion_matrices,
    plot_enhanced_metrics,
    compute_permutation_importance, save_permutation_importance,
    plot_permutation_importance_top15,
)

logger = setup_logging("stage3")


def run_stage3():
    """阶段3 主流程"""
    logger.info("=" * 60)
    logger.info("  阶段3: 机器学习基线模型")
    logger.info("=" * 60)

    logger.info("\n>>> [Step 1/5] 加载特征数据")
    df = load_feature_data()

    logger.info("\n>>> [Step 2/5] 构造标签 + 划分训练/测试集")
    df = build_labels(df)
    train, test = train_test_split_time_series(df)

    logger.info("\n>>> [Step 3/5] 训练模型 + 评估")
    metrics, fitted_models, predictions, confusion_df = train_and_evaluate(train, test)
    save_metrics(metrics)
    save_predictions(predictions)

    # 保存增强输出
    save_enhanced_metrics(metrics)
    save_enhanced_predictions(predictions)
    save_confusion_matrices(confusion_df)
    plot_enhanced_metrics(metrics)

    logger.info("\n>>> [Step 4/5] 特征重要性分析")
    save_feature_importance(fitted_models)

    logger.info("\n>>> [Step 5/5] Permutation importance 分析")
    perm_imp = compute_permutation_importance(fitted_models, test)
    if len(perm_imp) > 0:
        save_permutation_importance(perm_imp)
        plot_permutation_importance_top15(perm_imp)

    logger.info("\n--- 评估指标摘要 ---")
    for _, row in metrics.iterrows():
        logger.info("  %s | %s = %s", row["模型"], row["指标"], row["值"])

    logger.info("\n" + "=" * 60)
    logger.info("  阶段3 全部完成！")
    logger.info("  评估指标:     outputs/tables/SPY_ml_baseline_metrics.csv")
    logger.info("  增强指标:     outputs/tables/SPY_ml_model_metrics_enhanced.csv")
    logger.info("  测试预测:     outputs/tables/SPY_ml_test_predictions.csv")
    logger.info("  增强预测:     outputs/tables/SPY_ml_test_predictions_enhanced.csv")
    logger.info("  混淆矩阵:    outputs/tables/SPY_ml_confusion_matrices.csv")
    logger.info("  特征重要性:   outputs/tables/SPY_ml_feature_importance.csv")
    logger.info("  Perm Imp:     outputs/tables/SPY_permutation_importance.csv")
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
