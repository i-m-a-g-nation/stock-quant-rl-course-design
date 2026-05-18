"""
=============================================================================
阶段6: 强化学习交易环境与基线策略 —— 一键运行脚本
=============================================================================
流程:
  1. 加载特征数据，创建 StockTradingEnv
  2. 运行随机策略 baseline
  3. 运行买入持有 baseline
  4. 计算并对比评估指标
  5. 保存结果与权益曲线图

用法:  python run_stage6.py
=============================================================================
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import (
    DATA_FEATURES, OUTPUT_TABLES, OUTPUT_FIGURES, setup_logging,
)
from src.trading_env import (
    StockTradingEnv, run_random_baseline, run_buy_and_hold, compute_metrics,
    INPUT_FILE,
)
from src.rl_agent import train_qlearning, evaluate_qlearning

logger = setup_logging("stage6")


def run_stage6():
    logger.info("=" * 60)
    logger.info("  阶段6: 交易环境 + 基线策略")
    logger.info("=" * 60)

    try:
        import gymnasium
        logger.info("gymnasium %s 已安装（本次未使用，仅纯 Python 环境）",
                    gymnasium.__version__)
    except ImportError:
        logger.info("gymnasium 未安装（使用纯 Python 环境）")

    # 1. 加载数据
    logger.info("\n>>> [Step 1/4] 加载数据 + 创建环境")
    feat_path = DATA_FEATURES / INPUT_FILE
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path}\n请先运行 python run_stage2.py")

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    env = StockTradingEnv(df, transaction_cost=0.001)
    logger.info("环境创建: %d 步, action=2 (0=空仓,1=持有), cost=0.1%%",
                env.n)

    # 2-3. 运行基线
    logger.info("\n>>> [Step 2/4] 随机策略 baseline")
    hist_random = run_random_baseline(env, seed=42)
    m_random = compute_metrics(hist_random, "Random")

    logger.info("\n>>> [Step 3/4] 买入持有 baseline")
    hist_bh = run_buy_and_hold(env)
    m_bh = compute_metrics(hist_bh, "BuyAndHold")

    # 4. 汇总 + 保存
    logger.info("\n>>> [Step 4/4] 汇总指标 + 绘图")
    metrics = pd.DataFrame([m_random, m_bh])

    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_TABLES / "SPY_rl_env_baseline_metrics.csv",
                   index=False, encoding="utf-8-sig")
    logger.info("指标已保存: %s",
                OUTPUT_TABLES / "SPY_rl_env_baseline_metrics.csv")

    # 合并历史（含 strategy 列）
    hist_merged = pd.concat([hist_random, hist_bh], ignore_index=True)
    hist_merged.to_csv(
        OUTPUT_TABLES / "SPY_rl_env_baseline_history.csv",
        index=False, encoding="utf-8-sig",
    )
    logger.info("历史已保存: %s (%d 行)",
                OUTPUT_TABLES / "SPY_rl_env_baseline_history.csv",
                len(hist_merged))

    # 权益曲线图（使用 history["equity"]，与 final_equity 一致）
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist_random["equity"], linewidth=1.2, color="#1f77b4",
            label="Random Policy")
    ax.plot(hist_bh["equity"], linewidth=1.2, color="#d62728",
            label="Buy & Hold")
    ax.set_title("SPY RL Environment — Baseline Equity Curves",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95,
            f"Random Final: {m_random['final_equity']:.4f}\n"
            f"B&H Final:  {m_bh['final_equity']:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "SPY_rl_env_baseline_equity.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("权益曲线已保存: %s",
                OUTPUT_FIGURES / "SPY_rl_env_baseline_equity.png")

    # 打印摘要
    logger.info("\n--- 基线指标摘要 ---")
    for _, row in metrics.iterrows():
        logger.info("  [%s] total_ret=%.4f sharpe=%.4f max_dd=%.4f trades=%s",
                    row["strategy"], row["total_return"],
                    row["sharpe_ratio"], row["max_drawdown"],
                    row["number_of_trades"])

    # =========================================================
    # 5-7. Q-learning 训练 + 评估
    # =========================================================
    logger.info("\n>>> [Step 5/7] 创建 Q-learning 环境")
    env_ql = StockTradingEnv(df, transaction_cost=0.001)
    logger.info("状态空间: 27 (3x3x3), 动作空间: 2")

    logger.info("\n>>> [Step 6/7] 训练 Q-learning (50 episodes)")
    Q, train_log = train_qlearning(env_ql, episodes=50)
    train_log.to_csv(OUTPUT_TABLES / "SPY_rl_qlearning_training_log.csv",
                     index=False, encoding="utf-8-sig")
    logger.info("训练日志已保存: %s",
                OUTPUT_TABLES / "SPY_rl_qlearning_training_log.csv")

    logger.info("\n>>> [Step 7/7] 评估 Q-learning + 输出")
    hist_ql = evaluate_qlearning(env_ql, Q)
    m_ql = compute_metrics(hist_ql, "QLearning")

    # 汇总三策略指标
    all_metrics = pd.DataFrame([m_random, m_bh, m_ql])
    all_metrics.to_csv(OUTPUT_TABLES / "SPY_rl_qlearning_metrics.csv",
                       index=False, encoding="utf-8-sig")
    logger.info("Q-learning 指标已保存: %s",
                OUTPUT_TABLES / "SPY_rl_qlearning_metrics.csv")

    hist_ql.to_csv(OUTPUT_TABLES / "SPY_rl_qlearning_history.csv",
                   index=False, encoding="utf-8-sig")
    logger.info("Q-learning 历史已保存: %s (%d 行)",
                OUTPUT_TABLES / "SPY_rl_qlearning_history.csv", len(hist_ql))

    # Q-learning 权益曲线
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist_random["equity"], linewidth=1.0, color="#1f77b4",
            alpha=0.5, label="Random")
    ax.plot(hist_bh["equity"], linewidth=1.0, color="#d62728",
            alpha=0.5, label="Buy & Hold")
    ax.plot(hist_ql["equity"], linewidth=1.5, color="#2ca02c",
            label="Q-Learning")
    ax.set_title("SPY RL — Strategy Comparison (Q-Learning)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("Equity")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95,
            f"Random: {m_random['final_equity']:.4f}  "
            f"B&H: {m_bh['final_equity']:.4f}  "
            f"QL: {m_ql['final_equity']:.4f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "SPY_rl_qlearning_equity.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Q-learning 权益图已保存: %s",
                OUTPUT_FIGURES / "SPY_rl_qlearning_equity.png")

    logger.info("\n--- Q-learning 指标摘要 ---")
    logger.info("  [QLearning] total_ret=%.4f sharpe=%.4f max_dd=%.4f trades=%s",
                m_ql["total_return"], m_ql["sharpe_ratio"],
                m_ql["max_drawdown"], m_ql["number_of_trades"])

    logger.info("\n" + "=" * 60)
    logger.info("  阶段6 全部完成！")
    logger.info("  baseline:  outputs/tables/SPY_rl_env_baseline_*.csv")
    logger.info("  Q-learning: outputs/tables/SPY_rl_qlearning_*.csv")
    logger.info("  权益图:    outputs/figures/SPY_rl_*_equity.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_stage6()
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception:
        logger.error("阶段6 运行失败，错误详情:\n%s", traceback.format_exc())
        sys.exit(1)
