"""
=============================================================================
阶段2: 特征工程 —— 一键运行脚本
=============================================================================
流程:
  1. 读取 data/processed/SPY_clean_2015_2025.csv
  2. 构建基础技术特征 + 相对化特征
  3. 保存特征数据到 data/features/SPY_features_2015_2025.csv
  4. 生成特征摘要到 outputs/tables/SPY_feature_summary.csv
  5. 保存特征字段说明到 outputs/tables/SPY_feature_columns_enhanced.csv

用法:  python run_stage2.py
=============================================================================
"""
import sys
import traceback
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.features import (
    build_features, save_features, generate_feature_summary,
    save_feature_summary, save_feature_columns_description,
)
from src.data_cleaner import load_clean_data

logger = setup_logging("stage2")


def run_stage2():
    """阶段2 主流程"""
    logger.info("=" * 60)
    logger.info("  阶段2: 特征工程")
    logger.info("=" * 60)

    # Step 1: 加载清洗数据
    logger.info("\n>>> [Step 1/4] 加载清洗数据")
    df_clean = load_clean_data("SPY_clean_2015_2025.csv")

    # Step 2: 构建特征
    logger.info("\n>>> [Step 2/4] 构建特征")
    df_feat = build_features(df_clean)
    save_features(df_feat)

    # Step 3: 生成摘要
    logger.info("\n>>> [Step 3/4] 生成特征摘要")
    summary = generate_feature_summary(df_feat)
    save_feature_summary(summary)

    # Step 4: 保存特征字段说明
    logger.info("\n>>> [Step 4/4] 保存特征字段说明")
    save_feature_columns_description()

    # 打印关键指标
    logger.info("--- 特征摘要 ---")
    for _, row in summary.iterrows():
        name = str(row["指标"])
        if name in ["总行数", "总列数", "起始日期", "结束日期", "缺失值总数"]:
            logger.info("  %s: %s", name, row["值"])

    logger.info("\n" + "=" * 60)
    logger.info("  阶段2 全部完成！")
    logger.info("  特征数据:   data/features/SPY_features_2015_2025.csv")
    logger.info("  特征摘要:   outputs/tables/SPY_feature_summary.csv")
    logger.info("  特征说明:   outputs/tables/SPY_feature_columns_enhanced.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_stage2()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段2 运行失败，错误详情:\n%s", traceback.format_exc())
        sys.exit(1)
