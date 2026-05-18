"""
=============================================================================
阶段1: 数据获取与探索 —— 一键运行脚本
=============================================================================
流程:
  1. 下载 SPY 历史行情 (2015-01-01 ~ 2026-01-01)
  2. 数据清洗（缺失值、重复值、异常值处理）
  3. 生成数据质量报告
  4. 绘制三张数据概览图

用法:  python run_stage1.py
=============================================================================
"""
import sys
import traceback
from pathlib import Path

# 将项目根目录加入 sys.path，确保 src 可被导入
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging, ensure_dirs
from src.data_loader import load_data
from src.data_cleaner import clean_data, save_clean_data
from src.data_quality import generate_quality_report, save_quality_report
from src.plot_data_overview import generate_all_plots

logger = setup_logging("stock_quant")


def run_stage1():
    """阶段1 主流程"""
    logger.info("=" * 60)
    logger.info("  阶段1: 数据获取与探索")
    logger.info("=" * 60)

    # 确保目录结构
    ensure_dirs()

    # ---- Step 1: 数据下载 ----
    logger.info("\n>>> [Step 1/4] 数据下载")
    try:
        df_raw = load_data(force_download=False)
    except Exception as e:
        logger.error("数据下载失败: %s", e)
        raise

    # ---- Step 2: 数据清洗 ----
    logger.info("\n>>> [Step 2/4] 数据清洗")
    try:
        df_clean = clean_data(df_raw)
        save_clean_data(df_clean)
    except Exception as e:
        logger.error("数据清洗失败: %s", e)
        raise

    # ---- Step 3: 质量报告 ----
    logger.info("\n>>> [Step 3/4] 生成质量报告")
    try:
        report = generate_quality_report(df_clean)
        save_quality_report(report)
        # 简要打印关键指标
        logger.info("--- 质量报告摘要 ---")
        key_metrics = report[report["指标"].isin([
            "数据总行数", "起始日期", "结束日期", "交易天数",
        ])]
        for _, row in key_metrics.iterrows():
            logger.info("  %s: %s", row["指标"], row["值"])
    except Exception as e:
        logger.error("质量报告生成失败: %s", e)
        raise

    # ---- Step 4: 数据可视化 ----
    logger.info("\n>>> [Step 4/4] 数据可视化")
    try:
        generate_all_plots(df_clean)
    except Exception as e:
        logger.error("可视化生成失败: %s", e)
        raise

    logger.info("\n" + "=" * 60)
    logger.info("  阶段1 全部完成！")
    logger.info("  原始数据:   data/raw/SPY_raw_2015_2025.csv")
    logger.info("  清洗数据:   data/processed/SPY_clean_2015_2025.csv")
    logger.info("  质量报告:   outputs/tables/SPY_data_quality_report.csv")
    logger.info("  可视化图:   outputs/figures/  (SPY_close_price.png, SPY_volume.png, SPY_return_distribution.png)")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_stage1()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段1 运行失败，错误详情:\n%s", traceback.format_exc())
        sys.exit(1)
