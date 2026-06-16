"""
简易回测模块：基于阶段3预测信号的方向策略回测
支持多策略对比
"""
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.utils import OUTPUT_TABLES, OUTPUT_FIGURES, setup_logging, timer

logger = setup_logging("backtest")

MARKET_RETURNS_FILE = "SPY_market_returns.csv"
PREDICTIONS_FILE = "SPY_ml_test_predictions.csv"
BACKTEST_FILE = "SPY_strategy_backtest.csv"
METRICS_FILE = "SPY_strategy_metrics.csv"
EQUITY_PLOT_FILE = "SPY_strategy_equity_curve.png"

# 增强实验输出
ENHANCED_BACKTEST_FILE = "SPY_strategy_backtest_enhanced.csv"
ENHANCED_EQUITY_FILE = "SPY_strategy_equity_curves_enhanced.csv"
ENHANCED_METRICS_FILE = "SPY_strategy_backtest_enhanced_metrics.csv"
ENHANCED_EQUITY_PLOT = "SPY_strategy_equity_curves_enhanced.png"
ENHANCED_DRAWDOWN_PLOT = "SPY_strategy_drawdowns_enhanced.png"

MODEL_COL = "RandomForestClassifier_pred"
TRADING_DAYS = 252
PLOT_DPI = 150
DEFAULT_TRANSACTION_COST = 0.001


@timer
def load_data():
    ret_path = OUTPUT_TABLES / MARKET_RETURNS_FILE
    pred_path = OUTPUT_TABLES / PREDICTIONS_FILE
    if not ret_path.exists():
        raise FileNotFoundError(f"收益率文件不存在: {ret_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")
    ret = pd.read_csv(ret_path, index_col=0, parse_dates=True)
    pred = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    common_dates = ret.index.intersection(pred.index)
    ret = ret.loc[common_dates]
    pred = pred.loc[common_dates]
    logger.info("数据对齐: %d 行", len(common_dates))
    return ret, pred


@timer
def compute_strategy_returns(ret, pred):
    market_ret = ret["market_return_1d"]
    # signal = 当天模型预测信号；position = 实际用于当天收益计算的仓位（信号前移一天）
    signal = pred[MODEL_COL].astype(float)
    position = signal.shift(1)
    strategy_ret = position * market_ret
    bt = pd.DataFrame({
        "market_return_1d": market_ret,
        f"{MODEL_COL}": pred[MODEL_COL],
        "signal": signal,
        "position": position,
        "strategy_return_1d": strategy_ret,
        "strategy_equity": (1 + strategy_ret.fillna(0)).cumprod(),
    }, index=pred.index)
    bt = bt.dropna(subset=["strategy_return_1d"])
    logger.info("策略收益序列: %d 行", len(bt))
    return bt


@timer
def compute_backtest_metrics(bt):
    rets = bt["strategy_return_1d"]
    total_return = (1 + rets).prod() - 1
    ann_return = rets.mean() * TRADING_DAYS
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    equity = bt["strategy_equity"]
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    win_rate = (rets > 0).mean()
    trades = (bt["position"].diff().abs() > 0).sum()
    rows = [
        {"指标": "total_return", "值": round(total_return, 6)},
        {"指标": "annual_return", "值": round(ann_return, 6)},
        {"指标": "annual_volatility", "值": round(ann_vol, 6)},
        {"指标": "sharpe_ratio", "值": round(sharpe, 4)},
        {"指标": "max_drawdown", "值": round(max_dd, 6)},
        {"指标": "win_rate", "值": round(win_rate, 4)},
        {"指标": "number_of_trades", "值": int(trades)},
    ]
    metrics = pd.DataFrame(rows)
    logger.info("回测指标完成，共 %d 项", len(metrics))
    return metrics


@timer
def save_backtest(bt):
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_TABLES / BACKTEST_FILE
    bt.to_csv(fp, encoding="utf-8-sig")
    logger.info("回测明细: %s", fp)
    return fp


@timer
def save_backtest_metrics(metrics):
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_TABLES / METRICS_FILE
    metrics.to_csv(fp, index=False, encoding="utf-8-sig")
    logger.info("回测指标: %s", fp)
    return fp


@timer
def plot_equity_curve(bt):
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_FIGURES / EQUITY_PLOT_FILE
    market_equity = (1 + bt["market_return_1d"].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt.index, bt["strategy_equity"], linewidth=1.2, color="#1f77b4",
            label=f"Strategy ({MODEL_COL})")
    ax.plot(bt.index, market_equity, linewidth=1.0, color="#d62728",
            alpha=0.6, label="Buy & Hold")
    ax.set_title("SPY Direction Strategy Backtest", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Equity", fontsize=11)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    final_val = bt["strategy_equity"].iloc[-1]
    market_final = market_equity.iloc[-1]
    ax.text(0.02, 0.95, f"Strategy Final: {final_val:.4f}\nBuy & Hold: {market_final:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    fig.tight_layout()
    fig.savefig(fp, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("资金曲线: %s", fp)
    return fp


@timer
def run_backtest():
    logger.info("[Step 1/3] 加载数据")
    ret, pred = load_data()
    logger.info("[Step 2/3] 计算策略收益")
    bt = compute_strategy_returns(ret, pred)
    save_backtest(bt)
    logger.info("[Step 3/3] 计算回测指标 + 绘图")
    metrics = compute_backtest_metrics(bt)
    save_backtest_metrics(metrics)
    plot_equity_curve(bt)
    logger.info("回测全部完成！")
    return bt, metrics


# ============================================================
# 增强回测：多策略对比
# ============================================================

def _compute_strategy_equity(market_ret, signal, transaction_cost=DEFAULT_TRANSACTION_COST):
    """
    计算单个策略的收益序列和权益曲线。
    signal: 当天信号（1=做多, 0=空仓）
    信号需要 shift(1) 以避免使用当天未知信息
    """
    position = signal.shift(1)
    position = position.fillna(0)

    # 计算仓位变化和交易成本
    position_change = position.diff().abs().fillna(0)
    cost = position_change * transaction_cost

    strategy_ret = position * market_ret - cost
    equity = (1 + strategy_ret.fillna(0)).cumprod()

    return strategy_ret, equity, position


def _compute_single_metrics(rets, equity, positions, label=""):
    """计算单个策略的完整指标"""
    total_return = equity.iloc[-1] - 1.0 if len(equity) > 0 else 0
    ann_return = rets.mean() * TRADING_DAYS
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    win_rate = (rets[rets != 0] > 0).mean() if (rets != 0).sum() > 0 else 0
    trade_count = int((positions.diff().abs() > 0).sum())

    return {
        "strategy": label,
        "cumulative_return": round(float(total_return), 6),
        "annual_return": round(float(ann_return), 6),
        "annual_volatility": round(float(ann_vol), 6),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 6),
        "calmar": round(float(calmar), 4),
        "win_rate": round(float(win_rate), 4),
        "trade_count": trade_count,
    }


@timer
def run_enhanced_backtest(transaction_cost=DEFAULT_TRANSACTION_COST):
    """
    运行增强回测，对比多个策略：
    1. BuyAndHold - 始终持有
    2. AlwaysLong - 与 BuyAndHold 类似
    3. LogisticRegression - LR 模型预测
    4. RandomForestClassifier - RF 模型预测
    5. DummyMostFrequent - 多数类基线
    6. DummyStratified - 分层随机基线
    """
    logger.info("=" * 50)
    logger.info("增强回测: 多策略对比")
    logger.info("=" * 50)

    # 加载数据
    ret_path = OUTPUT_TABLES / MARKET_RETURNS_FILE
    pred_path = OUTPUT_TABLES / PREDICTIONS_FILE
    if not ret_path.exists():
        raise FileNotFoundError(f"收益率文件不存在: {ret_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")

    ret = pd.read_csv(ret_path, index_col=0, parse_dates=True)
    pred = pd.read_csv(pred_path, index_col=0, parse_dates=True)

    common_dates = ret.index.intersection(pred.index)
    ret = ret.loc[common_dates]
    pred = pred.loc[common_dates]

    market_ret = ret["market_return_1d"]
    logger.info("数据对齐: %d 行, 交易成本=%.4f", len(common_dates), transaction_cost)

    all_metrics = []
    equity_dict = {}
    drawdown_dict = {}

    # 1. BuyAndHold
    signal_bh = pd.Series(1, index=common_dates)
    ret_bh, eq_bh, pos_bh = _compute_strategy_equity(market_ret, signal_bh, transaction_cost)
    m_bh = _compute_single_metrics(ret_bh, eq_bh, pos_bh, "BuyAndHold")
    all_metrics.append(m_bh)
    equity_dict["BuyAndHold"] = eq_bh
    drawdown_dict["BuyAndHold"] = (eq_bh - eq_bh.cummax()) / eq_bh.cummax()

    # 2. AlwaysLong（同 BuyAndHold）
    all_metrics.append(_compute_single_metrics(ret_bh, eq_bh, pos_bh, "AlwaysLong"))
    equity_dict["AlwaysLong"] = eq_bh
    drawdown_dict["AlwaysLong"] = drawdown_dict["BuyAndHold"]

    # 3-6. 模型策略
    model_strategies = [
        ("LogisticRegression", "LogisticRegression_pred"),
        ("RandomForestClassifier", "RandomForestClassifier_pred"),
        ("DummyMostFrequent", "DummyMostFrequent_pred"),
        ("DummyStratified", "DummyStratified_pred"),
    ]

    for label, col in model_strategies:
        if col in pred.columns:
            signal = pred[col].astype(float)
            ret_s, eq_s, pos_s = _compute_strategy_equity(market_ret, signal, transaction_cost)
            m_s = _compute_single_metrics(ret_s, eq_s, pos_s, label)
            all_metrics.append(m_s)
            equity_dict[label] = eq_s
            drawdown_dict[label] = (eq_s - eq_s.cummax()) / eq_s.cummax()
            logger.info("  %s: cum_ret=%.4f sharpe=%.4f max_dd=%.4f",
                        label, m_s["cumulative_return"], m_s["sharpe"], m_s["max_drawdown"])
        else:
            logger.warning("  预测列 %s 不存在，跳过", col)

    # 保存策略对比表
    metrics_df = pd.DataFrame(all_metrics)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_TABLES / ENHANCED_METRICS_FILE,
                      index=False, encoding="utf-8-sig")
    logger.info("策略对比表已保存: %s", OUTPUT_TABLES / ENHANCED_METRICS_FILE)

    # 保存权益曲线
    equity_df = pd.DataFrame(equity_dict, index=common_dates)
    equity_df.index.name = "Date"
    equity_df.to_csv(OUTPUT_TABLES / ENHANCED_EQUITY_FILE, encoding="utf-8-sig")
    logger.info("权益曲线已保存: %s", OUTPUT_TABLES / ENHANCED_EQUITY_FILE)

    # 绘制权益曲线图
    _plot_enhanced_equity(equity_dict, common_dates)
    _plot_enhanced_drawdown(drawdown_dict, common_dates)

    logger.info("增强回测完成！")
    return metrics_df, equity_df


def _plot_enhanced_equity(equity_dict, dates):
    """绘制多策略权益曲线"""
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_FIGURES / ENHANCED_EQUITY_PLOT

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#d62728", "#8c564b", "#1f77b4", "#2ca02c", "#ff7f0e", "#e377c2"]

    for i, (label, eq) in enumerate(equity_dict.items()):
        ax.plot(dates, eq, linewidth=1.2 if "BuyAndHold" not in label else 1.0,
                color=colors[i % len(colors)], label=label,
                alpha=0.9 if "BuyAndHold" not in label else 0.6)

    ax.set_title("SPY Multi-Strategy Equity Curves (Enhanced Backtest)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Equity", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fp, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("权益曲线图已保存: %s", fp)


def _plot_enhanced_drawdown(drawdown_dict, dates):
    """绘制多策略回撤曲线"""
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_FIGURES / ENHANCED_DRAWDOWN_PLOT

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#d62728", "#8c564b", "#1f77b4", "#2ca02c", "#ff7f0e", "#e377c2"]

    for i, (label, dd) in enumerate(drawdown_dict.items()):
        ax.fill_between(dates, dd, 0, alpha=0.3, color=colors[i % len(colors)], label=label)

    ax.set_title("SPY Multi-Strategy Drawdowns (Enhanced Backtest)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fp, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("回撤曲线图已保存: %s", fp)
