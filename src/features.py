"""
特征工程模块：基于清洗数据构建基础技术特征 + 相对化特征
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import DATA_PROCESSED, DATA_FEATURES, OUTPUT_TABLES, setup_logging, timer

logger = setup_logging("feature_eng")

# 输入/输出
INPUT_FILE = "SPY_clean_2015_2025.csv"
OUTPUT_FILE = "SPY_features_2015_2025.csv"
SUMMARY_FILE = "SPY_feature_summary.csv"
FEATURE_COLUMNS_FILE = "SPY_feature_columns_enhanced.csv"

# 保留的原始字段
RAW_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# 特征列集合定义
FEATURE_COLS_ORIGINAL = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "return_1d", "return_5d", "ma_5", "ma_10", "ma_20", "ma_60",
    "volatility_20", "volume_ma_20", "close_ma20_ratio",
]

FEATURE_COLS_RELATIVE = [
    "open_close_ratio", "high_low_ratio",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio_v2", "close_ma60_ratio",
    "volume_ratio_20", "volatility_change_5",
    "return_1d", "return_5d", "volatility_20",
]

FEATURE_COLS_ALL = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "return_1d", "return_5d", "ma_5", "ma_10", "ma_20", "ma_60",
    "volatility_20", "volume_ma_20", "close_ma20_ratio",
    "open_close_ratio", "high_low_ratio",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio_v2", "close_ma60_ratio",
    "volume_ratio_20", "volatility_change_5",
]


@timer
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于清洗数据构建基础特征 + 相对化特征。
    参数:
        df: 清洗后 DataFrame（日期索引，含 OHLCV 列）
    返回:
        特征 DataFrame（含原始列 + 技术特征列 + 相对化特征列）
    """
    logger.info("=" * 50)
    logger.info("开始特征工程")
    logger.info("=" * 50)
    logger.info("输入: %d 行, %d 列", len(df), len(df.columns))

    # 确保原始列存在
    available_raw = [c for c in RAW_COLUMNS if c in df.columns]
    feat = df[available_raw].copy()

    close = feat["Close"]
    volume = feat["Volume"]

    # ---- 收益率特征 ----
    feat["return_1d"] = close.pct_change(1) * 100
    feat["return_5d"] = close.pct_change(5) * 100
    logger.info("  + return_1d, return_5d")

    # ---- 移动均线 ----
    feat["ma_5"] = close.rolling(window=5).mean()
    feat["ma_10"] = close.rolling(window=10).mean()
    feat["ma_20"] = close.rolling(window=20).mean()
    feat["ma_60"] = close.rolling(window=60).mean()
    logger.info("  + ma_5, ma_10, ma_20, ma_60")

    # ---- 波动率 ----
    feat["volatility_20"] = feat["return_1d"].rolling(window=20).std()
    logger.info("  + volatility_20")

    # ---- 成交量均线 ----
    feat["volume_ma_20"] = volume.rolling(window=20).mean()
    logger.info("  + volume_ma_20")

    # ---- 价格相对位置 ----
    feat["close_ma20_ratio"] = close / feat["ma_20"]
    logger.info("  + close_ma20_ratio")

    # ---- 相对化特征（新增） ----
    feat["open_close_ratio"] = feat["Open"] / feat["Close"] - 1
    feat["high_low_ratio"] = feat["High"] / feat["Low"] - 1
    feat["close_ma5_ratio"] = close / feat["ma_5"] - 1
    feat["close_ma10_ratio"] = close / feat["ma_10"] - 1
    feat["close_ma20_ratio_v2"] = close / feat["ma_20"] - 1
    feat["close_ma60_ratio"] = close / feat["ma_60"] - 1
    feat["volume_ratio_20"] = volume / feat["volume_ma_20"] - 1
    feat["volatility_change_5"] = feat["volatility_20"] / feat["volatility_20"].shift(5) - 1
    logger.info("  + 相对化特征 (8 个)")

    # ---- 清洗 NaN（rolling / pct_change 产生） ----
    n_before = len(feat)
    feat = feat.dropna()
    n_dropped = n_before - len(feat)
    logger.info("删除 NaN 行: %d 行 → %d 行 (丢弃 %d 行)", n_before, len(feat), n_dropped)

    # 按日期排序
    feat = feat.sort_index()

    logger.info("=" * 50)
    logger.info("特征工程完成: %d 行, %d 列", len(feat), len(feat.columns))
    logger.info("日期范围: %s → %s",
                feat.index.min().strftime("%Y-%m-%d"),
                feat.index.max().strftime("%Y-%m-%d"))
    logger.info("=" * 50)
    return feat


@timer
def save_features(df: pd.DataFrame, filename: str = OUTPUT_FILE) -> Path:
    """保存特征数据到 data/features/"""
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    filepath = DATA_FEATURES / filename
    df.to_csv(filepath)
    logger.info("特征数据已保存: %s", filepath)
    return filepath


@timer
def load_features(filename: str = OUTPUT_FILE) -> pd.DataFrame:
    """从 data/features/ 加载特征数据"""
    filepath = DATA_FEATURES / filename
    if not filepath.exists():
        raise FileNotFoundError(f"特征文件不存在: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info("已加载特征数据: %s (%d 行)", filepath, len(df))
    return df


@timer
def generate_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成特征摘要报告。
    包含: 基本信息 + 每个特征的均值、标准差、最小值、最大值
    """
    logger.info("生成特征摘要...")
    rows = []

    # 基本信息
    rows.append({"指标": "总行数", "值": len(df)})
    rows.append({"指标": "总列数", "值": len(df.columns)})
    rows.append({"指标": "起始日期", "值": df.index.min().strftime("%Y-%m-%d")})
    rows.append({"指标": "结束日期", "值": df.index.max().strftime("%Y-%m-%d")})
    rows.append({"指标": "缺失值总数", "值": int(df.isnull().sum().sum())})

    # 每列描述
    desc = df.describe()
    for col in df.columns:
        for stat in ["mean", "std", "min", "max"]:
            if stat in desc.index:
                val = desc.loc[stat, col]
                rows.append({
                    "指标": f"{stat}_{col}",
                    "值": round(float(val), 4),
                })

    report = pd.DataFrame(rows)
    logger.info("特征摘要生成完成，共 %d 项", len(report))
    return report


@timer
def save_feature_summary(report: pd.DataFrame, filename: str = SUMMARY_FILE) -> Path:
    """保存特征摘要到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    report.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("特征摘要已保存: %s", filepath)
    return filepath


@timer
def save_feature_columns_description(filename: str = FEATURE_COLUMNS_FILE) -> Path:
    """保存特征字段说明到 outputs/tables/"""
    rows = []
    for col in FEATURE_COLS_ORIGINAL:
        rows.append({"特征列": col, "集合": "ORIGINAL", "说明": _get_feature_description(col)})
    for col in FEATURE_COLS_RELATIVE:
        if col not in FEATURE_COLS_ORIGINAL:
            rows.append({"特征列": col, "集合": "RELATIVE", "说明": _get_feature_description(col)})
        else:
            rows.append({"特征列": col, "集合": "BOTH", "说明": _get_feature_description(col)})
    for col in FEATURE_COLS_ALL:
        if col not in [r["特征列"] for r in rows]:
            rows.append({"特征列": col, "集合": "ALL", "说明": _get_feature_description(col)})

    df = pd.DataFrame(rows)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("特征字段说明已保存: %s (%d 行)", filepath, len(df))
    return filepath


def _get_feature_description(col: str) -> str:
    """获取特征列的中文说明"""
    descriptions = {
        "Open": "开盘价",
        "High": "最高价",
        "Low": "最低价",
        "Close": "收盘价",
        "Adj Close": "调整后收盘价",
        "Volume": "成交量",
        "return_1d": "1日收益率 (%)",
        "return_5d": "5日收益率 (%)",
        "ma_5": "5日移动均线",
        "ma_10": "10日移动均线",
        "ma_20": "20日移动均线",
        "ma_60": "60日移动均线",
        "volatility_20": "20日波动率",
        "volume_ma_20": "20日成交量均线",
        "close_ma20_ratio": "收盘价/20日均线",
        "open_close_ratio": "开盘/收盘价比 - 1",
        "high_low_ratio": "最高/最低价比 - 1",
        "close_ma5_ratio": "收盘价/5日均线 - 1",
        "close_ma10_ratio": "收盘价/10日均线 - 1",
        "close_ma20_ratio_v2": "收盘价/20日均线 - 1",
        "close_ma60_ratio": "收盘价/60日均线 - 1",
        "volume_ratio_20": "成交量/20日均量 - 1",
        "volatility_change_5": "波动率5日变化率",
    }
    return descriptions.get(col, "")


@timer
def run_pipeline():
    """一键运行特征工程全流程"""
    # 1. 加载清洗数据
    logger.info("[Step 1/4] 加载清洗数据")
    clean_path = DATA_PROCESSED / INPUT_FILE
    if not clean_path.exists():
        raise FileNotFoundError(f"清洗数据不存在: {clean_path}\n请先运行 python run_stage1.py")
    df_clean = pd.read_csv(clean_path, index_col=0, parse_dates=True)

    # 2. 构建特征
    logger.info("[Step 2/4] 构建特征")
    df_feat = build_features(df_clean)
    save_features(df_feat)

    # 3. 生成摘要
    logger.info("[Step 3/4] 生成特征摘要")
    summary = generate_feature_summary(df_feat)
    save_feature_summary(summary)

    # 4. 保存特征字段说明
    logger.info("[Step 4/4] 保存特征字段说明")
    save_feature_columns_description()

    # 打印关键信息
    logger.info("--- 特征摘要 ---")
    for _, row in summary.iterrows():
        name = str(row["指标"])
        if name in ["总行数", "总列数", "起始日期", "结束日期", "缺失值总数"]:
            logger.info("  %s: %s", name, row["值"])

    logger.info("特征工程全部完成！")
    return df_feat, summary
