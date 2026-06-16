"""
=============================================================================
阶段4: 风险指标评估 + 多策略回测 —— 一键运行脚本
=============================================================================
流程:
  1. 计算市场收益率序列 + 风险指标
  2. 预测方向命中率分析
  3. 基于阶段3预测信号的简易方向策略回测（原有）
  4. 增强多策略回测对比（新增）

用法:  python run_stage4.py
=============================================================================
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.risk_metrics import (
    load_clean_data, compute_market_returns, save_market_returns,
    compute_risk_metrics, save_risk_metrics,
    compute_direction_analysis, save_direction_analysis,
)
from src.backtest import run_backtest, run_enhanced_backtest

logger = setup_logging("stage4")


def run_stage4():
    """阶段4 主流程"""
    logger.info("=" * 60)
    logger.info("  阶段4: 风险指标评估 + 多策略回测")
    logger.info("=" * 60)

    logger.info("\n>>> [Step 1/5] 风险指标：收益率序列")
    df = load_clean_data()
    ret = compute_market_returns(df)
    save_market_returns(ret)

    logger.info("\n>>> [Step 2/5] 风险指标：指标计算")
    risk_report = compute_risk_metrics(ret)
    save_risk_metrics(risk_report)

    logger.info("\n>>> [Step 3/5] 风险指标：方向命中率")
    direction_analysis = compute_direction_analysis()
    if direction_analysis is not None:
        save_direction_analysis(direction_analysis)

    logger.info("\n>>> [Step 4/5] 简易方向策略回测（原有）")
    bt, bt_metrics = run_backtest()

    logger.info("\n>>> [Step 5/5] 增强多策略回测（新增）")
    enhanced_metrics, equity_df = run_enhanced_backtest()

    logger.info("\n--- 风险指标摘要 ---")
    for _, row in risk_report.iterrows():
        logger.info("  %s = %s", row["指标"], row["值"])

    if direction_analysis is not None:
        logger.info("\n--- 预测方向分析摘要 ---")
        for _, row in direction_analysis.iterrows():
            logger.info("  %s = %s", row["指标"], row["值"])

    logger.info("\n--- 原回测指标摘要 ---")
    for _, row in bt_metrics.iterrows():
        logger.info("  %s = %s", row["指标"], row["值"])

    logger.info("\n--- 增强多策略回测摘要 ---")
    for _, row in enhanced_metrics.iterrows():
        logger.info("  [%s] cum_ret=%.4f sharpe=%.4f max_dd=%.4f",
                    row["strategy"], row["cumulative_return"],
                    row["sharpe"], row["max_drawdown"])

    logger.info("\n" + "=" * 60)
    logger.info("  阶段4 全部完成！")
    logger.info("  回测明细:     outputs/tables/SPY_strategy_backtest.csv")
    logger.info("  回测指标:     outputs/tables/SPY_strategy_metrics.csv")
    logger.info("  资金曲线:     outputs/figures/SPY_strategy_equity_curve.png")
    logger.info("  --- 增强输出 ---")
    logger.info("  策略对比表:   outputs/tables/SPY_strategy_backtest_enhanced_metrics.csv")
    logger.info("  权益曲线:     outputs/tables/SPY_strategy_equity_curves_enhanced.csv")
    logger.info("  权益图:       outputs/figures/SPY_strategy_equity_curves_enhanced.png")
    logger.info("  回撤图:       outputs/figures/SPY_strategy_drawdowns_enhanced.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_stage4()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段4 运行失败，错误详情:\n%s", traceback.format_exc())
        sys.exit(1)
