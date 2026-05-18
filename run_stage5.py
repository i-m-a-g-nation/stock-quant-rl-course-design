"""
=============================================================================
阶段5: LSTM 方向预测 —— 一键运行脚本
=============================================================================
流程:
  1. 读取 data/features/SPY_features_2015_2025.csv
  2. 构造标签 + 特征标准化 + 滑动窗口序列
  3. 训练单层 LSTM 二分类模型
  4. 评估 accuracy / precision / recall / f1 / roc_auc
  5. 保存指标、预测结果、模型、训练曲线

用法:  python run_stage5.py
=============================================================================
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.lstm_model import run_pipeline

logger = setup_logging("stage5")


def run_stage5():
    logger.info("=" * 60)
    logger.info("  阶段5: LSTM 方向预测")
    logger.info("=" * 60)

    try:
        import torch
        logger.info("PyTorch %s, device=%s", torch.__version__,
                    "cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        logger.error("PyTorch 未安装，请先安装: pip install torch")
        sys.exit(1)

    metrics = run_pipeline()

    logger.info("\n--- LSTM 指标摘要 ---")
    for _, row in metrics.iterrows():
        logger.info("  %s = %s", row["指标"], row["值"])

    logger.info("\n" + "=" * 60)
    logger.info("  阶段5 全部完成！")
    logger.info("  指标:   outputs/tables/SPY_lstm_metrics.csv")
    logger.info("  预测:   outputs/tables/SPY_lstm_test_predictions.csv")
    logger.info("  模型:   outputs/models/SPY_lstm_model.pt")
    logger.info("  曲线:   outputs/figures/SPY_lstm_training_curve.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_stage5()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段5 运行失败，错误详情:\n%s", traceback.format_exc())
        sys.exit(1)
