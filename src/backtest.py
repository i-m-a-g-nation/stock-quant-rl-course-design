"""
简易回测模块：基于阶段3预测信号的方向策略回测
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

MODEL_COL = "RandomForestClassifier_pred"
TRADING_DAYS = 252
PLOT_DPI = 150


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
