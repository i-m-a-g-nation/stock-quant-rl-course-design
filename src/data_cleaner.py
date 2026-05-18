"""
数据清洗模块：处理缺失值、重复值、异常值，输出标准化数据
"""
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils import DATA_PROCESSED, setup_logging, timer

logger = setup_logging()

# 保留的核心行情列（依优先级）
CORE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
CLEAN_FILENAME = "SPY_clean_2015_2025.csv"


def _check_negative_values(df: pd.DataFrame) -> int:
    """检查 OHLCV 中的负值（除 Volume 可为零）"""
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    available = [c for c in price_cols if c in df.columns]
    neg_mask = (df[available] < 0).any(axis=1)
    n_neg = neg_mask.sum()
    if n_neg > 0:
        logger.warning("发现 %d 行含负价格，将丢弃这些行", n_neg)
    return n_neg


@timer
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗行情数据，执行以下步骤：
      1. 筛选核心列
      2. 处理缺失值（前向填充 → 后向填充 → 丢弃残余）
      3. 删除重复索引行
      4. 丢弃负价格行（异常）
      5. 按日期排序
      6. 数据类型标准化
    参数:
        df: 原始 DataFrame
    返回:
        清洗后的 DataFrame
    """
    logger.info("=" * 50)
    logger.info("开始数据清洗流程")
    logger.info("=" * 50)

    n_initial = len(df)
    logger.info("初始: %d 行, %d 列", n_initial, len(df.columns))

    # Step 1: 筛选核心列
    available_cols = [c for c in CORE_COLUMNS if c in df.columns]
    missing_cols = set(CORE_COLUMNS) - set(available_cols)
    if missing_cols:
        logger.warning("以下核心列缺失: %s，仅保留可用列", missing_cols)
    df = df[available_cols].copy()
    logger.info("Step 1 列筛选: 保留 %d 列 → %s", len(available_cols), available_cols)

    # Step 2: 缺失值处理
    n_missing_before = df.isnull().sum().sum()
    df = df.ffill()          # 前向填充（用前一天值补当天）
    df = df.bfill()          # 后向填充（处理头部 NaN）
    df = df.dropna()         # 丢弃仍为 NaN 的行
    n_missing_after = df.isnull().sum().sum()
    logger.info("Step 2 缺失值: %d → %d", n_missing_before, n_missing_after)

    # Step 3: 删除重复索引
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        df = df[~df.index.duplicated(keep="first")]
        logger.info("Step 3 重复索引: 删除 %d 行", n_dup)
    else:
        logger.info("Step 3 重复索引: 无重复")

    # Step 4: 负价格异常处理
    n_neg = _check_negative_values(df)
    if n_neg > 0:
        df = df[~(df[[c for c in ["Open", "High", "Low", "Close", "Adj Close"]
                        if c in df.columns]] < 0).any(axis=1)]
        logger.info("Step 4 异常负值: 删除 %d 行", n_neg)
    else:
        logger.info("Step 4 异常负值: 未发现")

    # Step 5: 按日期排序
    df = df.sort_index()
    logger.info("Step 5 排序: 已按日期升序排列")

    # Step 6: 数据类型标准化
    price_cols_available = [c for c in ["Open", "High", "Low", "Close", "Adj Close"]
                            if c in df.columns]
    for col in price_cols_available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")
    logger.info("Step 6 数据类型: 价格→float, 成交量→int64")

    # 最终统计
    n_final = len(df)
    logger.info("=" * 50)
    logger.info("清洗完成: %d 行 → %d 行 (丢弃 %d 行)", n_initial, n_final, n_initial - n_final)
    logger.info("日期范围: %s → %s", df.index.min().strftime("%Y-%m-%d"),
                df.index.max().strftime("%Y-%m-%d"))
    logger.info("=" * 50)
    return df


@timer
def save_clean_data(df: pd.DataFrame, filename: str = CLEAN_FILENAME) -> Path:
    """
    保存清洗后数据到 data/processed/
    参数:
        df      : 清洗后 DataFrame
        filename: 文件名
    返回:
        保存路径
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    filepath = DATA_PROCESSED / filename
    df.to_csv(filepath)
    logger.info("清洗数据已保存: %s", filepath)
    return filepath


@timer
def load_clean_data(filename: str = CLEAN_FILENAME) -> pd.DataFrame:
    """
    从 data/processed/ 加载清洗后数据
    参数:
        filename: 文件名
    返回:
        pd.DataFrame
    """
    filepath = DATA_PROCESSED / filename
    if not filepath.exists():
        raise FileNotFoundError(f"清洗数据文件不存在: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info("已加载清洗数据: %s (%d 行)", filepath, len(df))
    return df
