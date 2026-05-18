"""
数据概览可视化模块：生成三张基础图像
  1. 收盘价时间序列（含 20/60 日均线）
  2. 日收益率分布直方图（含正态拟合）
  3. 成交量时序柱状图（含 20 日均量线）
"""
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，适合命令行批处理

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from src.utils import OUTPUT_FIGURES, setup_logging, timer

logger = setup_logging()

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 图像尺寸与风格
FIG_SIZE = (14, 6)
DPI = 150


@timer
def plot_close_with_ma(df: pd.DataFrame, save_path: str = None):
    """
    图1: 收盘价 + 20/60 日移动均线
    参数:
        df       : 行情 DataFrame（需含 Close 列）
        save_path: 保存路径，默认 outputs/figures/SPY_close_price.png
    """
    if save_path is None:
        save_path = str(OUTPUT_FIGURES / "SPY_close_price.png")

    # 计算均线
    close = df["Close"]
    ma20 = close.rolling(window=20).mean()
    ma60 = close.rolling(window=60).mean()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(close.index, close.values, linewidth=1.0, color="#1f77b4", label="收盘价", alpha=0.9)
    ax.plot(ma20.index, ma20.values, linewidth=1.2, color="#ff7f0e", label="MA20 (20日均线)")
    ax.plot(ma60.index, ma60.values, linewidth=1.2, color="#2ca02c", label="MA60 (60日均线)")

    # 格式
    ax.set_title("SPY 收盘价 & 移动均线 (2015-2026)", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("价格 (USD)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("图1 已保存: %s", save_path)


@timer
def plot_daily_return_distribution(df: pd.DataFrame, save_path: str = None):
    """
    图2: 日收益率分布直方图 + 正态分布拟合曲线
    参数:
        df       : 行情 DataFrame（需含 Close 列）
        save_path: 保存路径，默认 outputs/figures/SPY_return_distribution.png
    """
    if save_path is None:
        save_path = str(OUTPUT_FIGURES / "SPY_return_distribution.png")

    # 计算日收益率（简单收益率）
    returns = df["Close"].pct_change().dropna() * 100  # 转为百分比

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 直方图
    n, bins, patches = ax.hist(
        returns.values, bins=80, density=True,
        color="#1f77b4", alpha=0.7, edgecolor="white", linewidth=0.3,
    )

    # 正态分布拟合（手动计算，避免 scipy 依赖）
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    y = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, y, linewidth=2.0, color="#d62728", label=f"正态拟合 N({mu:.3f}, {sigma**2:.4f})")

    # 统计标注
    ax.axvline(mu, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    skew_val = returns.skew()
    kurt_val = returns.kurtosis()

    textstr = (
        f"均值: {mu:.3f}%\n"
        f"标准差: {sigma:.3f}%\n"
        f"偏度: {skew_val:.3f}\n"
        f"峰度: {kurt_val:.3f}"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7)
    ax.text(0.97, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    ax.set_title("SPY 日收益率分布 (2015-2026)", fontsize=14, fontweight="bold")
    ax.set_xlabel("日收益率 (%)", fontsize=11)
    ax.set_ylabel("概率密度", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("图2 已保存: %s", save_path)


@timer
def plot_volume_bar(df: pd.DataFrame, save_path: str = None):
    """
    图3: 成交量时序柱状图 + 20 日均量线
    参数:
        df       : 行情 DataFrame（需含 Volume 列）
        save_path: 保存路径，默认 outputs/figures/SPY_volume.png
    """
    if save_path is None:
        save_path = str(OUTPUT_FIGURES / "SPY_volume.png")

    volume = df["Volume"]
    ma_vol = volume.rolling(window=20).mean()

    # 根据收盘价涨跌着色柱子
    close = df["Close"]
    price_change = close.diff()
    colors = np.where(price_change >= 0, "#d62728", "#1f77b4")  # 涨红跌绿
    # 第一个元素没有 diff 值，设为灰色
    colors[0] = "#7f7f7f"

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(volume.index, volume.values, color=colors, width=1.0, alpha=0.7, linewidth=0)
    ax.plot(ma_vol.index, ma_vol.values, linewidth=1.5, color="#ff7f0e",
            label="MA20 (20日均量)")

    ax.set_title("SPY 成交量 (2015-2026)", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("成交量", fontsize=11)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K")
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("图3 已保存: %s", save_path)


@timer
def generate_all_plots(df: pd.DataFrame):
    """
    一键生成三张数据概览图像并保存到 outputs/figures/
    参数:
        df: 行情 DataFrame（需含 Close, Volume 列）
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 50)
    logger.info("开始生成数据概览图像")
    logger.info("=" * 50)

    plot_close_with_ma(df)
    plot_daily_return_distribution(df)
    plot_volume_bar(df)

    logger.info("全部图像生成完毕，保存于 %s", OUTPUT_FIGURES)
