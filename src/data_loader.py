"""
数据加载模块：使用 yfinance 下载 SPY 历史日线行情数据
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils import DATA_RAW, setup_logging, timer

logger = setup_logging()

# 目标标的与时间范围
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"
RAW_FILENAME = "SPY_raw_2015_2025.csv"


@timer
def download_spy(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    从 Yahoo Finance 下载 SPY 历史日线数据
    参数:
        ticker: 股票代码
        start : 起始日期 (YYYY-MM-DD)
        end   : 结束日期 (YYYY-MM-DD)
    返回:
        pd.DataFrame，列为 [Open, High, Low, Close, Adj Close, Volume]
    """
    logger.info("正在下载 %s 数据: %s → %s", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if df.empty:
        raise ValueError(f"下载的 {ticker} 数据为空，请检查网络或日期范围")

    # yfinance 可能返回 MultiIndex 列名，展平为单层
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    logger.info("下载完成，共 %d 条日线记录，%d 列", len(df), len(df.columns))
    logger.info("日期范围: %s → %s", df.index.min().strftime("%Y-%m-%d"),
                df.index.max().strftime("%Y-%m-%d"))
    return df


@timer
def save_raw_data(df: pd.DataFrame, filename: str = RAW_FILENAME) -> Path:
    """
    将原始数据保存为 CSV
    参数:
        df      : 原始数据 DataFrame
        filename: 保存文件名
    返回:
        保存路径 (pathlib.Path)
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    filepath = DATA_RAW / filename
    df.to_csv(filepath)
    logger.info("原始数据已保存: %s", filepath)
    return filepath


@timer
def load_raw_data(filename: str = RAW_FILENAME) -> pd.DataFrame:
    """
    从本地 CSV 加载原始数据
    参数:
        filename: 文件名
    返回:
        pd.DataFrame，以日期为索引
    """
    filepath = DATA_RAW / filename
    if not filepath.exists():
        raise FileNotFoundError(f"原始数据文件不存在: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info("已加载本地原始数据: %s (%d 行)", filepath, len(df))
    return df


@timer
def load_data(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    数据加载主入口：优先读取本地缓存，否则从 Yahoo Finance 下载
    参数:
        ticker        : 股票代码
        start         : 起始日期
        end           : 结束日期
        force_download: 是否强制重新下载
    返回:
        pd.DataFrame
    """
    local_file = DATA_RAW / RAW_FILENAME
    if local_file.exists() and not force_download:
        logger.info("发现本地缓存，直接加载")
        return load_raw_data()

    logger.info("本地缓存不存在或强制下载，开始从 Yahoo Finance 获取")
    df = download_spy(ticker, start, end)
    save_raw_data(df)
    return df
