"""
数据质量报告模块
=================
从 data/processed/SPY_clean_2015_2025.csv 读取清洗后的行情数据，
生成数据质量报告并保存到 outputs/tables/SPY_data_quality_report.csv。

作为模块调用:
    from src.data_quality import generate_quality_report, save_quality_report

作为脚本运行:
    python src/data_quality.py
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# 确保项目根目录在 sys.path 中（支持直接运行）
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import DATA_PROCESSED, OUTPUT_TABLES, setup_logging, timer

logger = setup_logging("data_quality")

# 默认路径
DEFAULT_INPUT = "SPY_clean_2015_2025.csv"
DEFAULT_OUTPUT = "SPY_data_quality_report.csv"
DEFAULT_TICKER = "SPY"

# 描述性统计指标名称
STAT_NAMES = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]


def _extract_ticker(filename: str) -> str:
    """从文件名提取 ticker，如 'SPY_clean_2015_2025.csv' → 'SPY'"""
    stem = Path(filename).stem
    return stem.split("_")[0].upper()


def _compute_return_1d(df: pd.DataFrame) -> pd.Series:
    """
    计算日收益率（百分比形式），丢弃首行 NaN
    返回:
        pd.Series，值为百分比收益率（如 1.5 表示 1.5%）
    """
    if "Close" not in df.columns:
        raise KeyError("缺少 Close 列，无法计算 return_1d")
    return (df["Close"].pct_change() * 100).dropna()


@timer
def generate_quality_report(
    df: pd.DataFrame,
    ticker: str = DEFAULT_TICKER,
) -> pd.DataFrame:
    """
    生成数据质量报告。

    参数:
        df    : 清洗后的行情 DataFrame（日期为索引，含 OHLCV 列）
        ticker: 股票代码

    返回:
        pd.DataFrame (列: ["指标", "值"])
    """
    logger.info("=" * 50)
    logger.info("生成数据质量报告 [ticker=%s]", ticker)
    logger.info("=" * 50)

    if df.empty:
        raise ValueError("输入 DataFrame 为空")

    rows = []
    cols_available = list(df.columns)

    # 计算日收益率
    return_1d = None
    if "Close" in df.columns:
        return_1d = _compute_return_1d(df)
        logger.info("return_1d 已计算: %d 个有效值", len(return_1d))

    # ================================================================
    # Section 1: 基本信息
    # ================================================================
    date_min = df.index.min().strftime("%Y-%m-%d")
    date_max = df.index.max().strftime("%Y-%m-%d")
    is_monotonic = bool(df.index.is_monotonic_increasing)
    n_dup_dates = int(df.index.duplicated().sum())

    rows.append({"指标": "ticker",                 "值": ticker})
    rows.append({"指标": "数据总行数",              "值": len(df)})
    rows.append({"指标": "数据总列数",              "值": len(df.columns)})
    rows.append({"指标": "起始日期",                "值": date_min})
    rows.append({"指标": "结束日期",                "值": date_max})
    rows.append({"指标": "交易天数",                "值": len(df)})
    rows.append({
        "指标": "日期是否单调递增",
        "值": "是" if is_monotonic else f"否（非单调，检查索引）",
    })
    rows.append({
        "指标": "是否存在重复日期",
        "值": "否" if n_dup_dates == 0 else f"是（{n_dup_dates} 处重复）",
    })

    logger.info(
        "基本信息: ticker=%s | %d行 %d列 | %s → %s | 单调=%s | 重复=%d处",
        ticker, len(df), len(df.columns),
        date_min, date_max, is_monotonic, n_dup_dates,
    )

    # ================================================================
    # Section 2: 缺失值统计（每列一条）
    # ================================================================
    missing = df.isnull().sum()
    for col in cols_available:
        rows.append({"指标": f"缺失值_{col}", "值": int(missing[col])})
    if return_1d is not None:
        rows.append({"指标": "缺失值_return_1d", "值": int(return_1d.isnull().sum())})

    logger.info("缺失值总计: %d", int(missing.sum()))

    # ================================================================
    # Section 3: 描述性统计
    # ================================================================
    # 3a. OHLCV 列的 describe
    desc = df.describe()
    for stat in STAT_NAMES:
        if stat in desc.index:
            for col in cols_available:
                raw_val = desc.loc[stat, col]
                val = round(float(raw_val), 4) if pd.notna(raw_val) else "N/A"
                rows.append({"指标": f"{stat}_{col}", "值": val})

    # 3b. return_1d 的 describe
    if return_1d is not None:
        ret_desc = return_1d.describe()
        for stat in STAT_NAMES:
            if stat in ret_desc.index:
                raw_val = ret_desc[stat]
                val = round(float(raw_val), 4) if pd.notna(raw_val) else "N/A"
                rows.append({"指标": f"{stat}_return_1d", "值": val})

    logger.info("描述性统计: 已完成")

    # ================================================================
    # Section 4: 关键指标摘要
    # ================================================================
    if return_1d is not None:
        rows.append({"指标": "return_1d_均值",   "值": round(float(return_1d.mean()), 4)})
        rows.append({"指标": "return_1d_标准差", "值": round(float(return_1d.std()), 4)})
        rows.append({"指标": "return_1d_最小值", "值": round(float(return_1d.min()), 4)})
        rows.append({"指标": "return_1d_最大值", "值": round(float(return_1d.max()), 4)})

    if "Close" in df.columns:
        rows.append({"指标": "Close_最小值", "值": round(float(df["Close"].min()), 4)})
        rows.append({"指标": "Close_最大值", "值": round(float(df["Close"].max()), 4)})

    if "Volume" in df.columns:
        rows.append({"指标": "Volume_最小值", "值": int(df["Volume"].min())})
        rows.append({"指标": "Volume_最大值", "值": int(df["Volume"].max())})

    logger.info("关键指标: 已完成")

    # 汇总
    report_df = pd.DataFrame(rows)
    logger.info("报告生成完成，共 %d 项指标", len(report_df))
    return report_df


@timer
def save_quality_report(
    report_df: pd.DataFrame,
    filename: str = DEFAULT_OUTPUT,
):
    """
    保存质量报告到 outputs/tables/
    参数:
        report_df: 报告 DataFrame
        filename : 文件名
    返回:
        保存路径 (pathlib.Path)
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    report_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("质量报告已保存: %s", filepath)
    return filepath


# ================================================================
# 终端摘要打印
# ================================================================

def _print_summary(report_df: pd.DataFrame, ticker: str):
    """在终端打印结构化的质量报告摘要"""
    lookup = dict(zip(report_df["指标"].astype(str), report_df["值"].astype(str)))

    def _get(key, default="N/A"):
        return lookup.get(key, default)

    lines = [
        "",
        "=" * 55,
        f"  {ticker} 数据质量报告 · 终端摘要",
        "=" * 55,
        f"  总行数           : {_get('数据总行数')}",
        f"  起始日期         : {_get('起始日期')}",
        f"  结束日期         : {_get('结束日期')}",
        f"  日期单调递增     : {_get('日期是否单调递增')}",
        f"  存在重复日期     : {_get('是否存在重复日期')}",
        "",
        f"  --- 缺失值 ---",
    ]
    # 动态找出所有缺失值行
    for idx, row in report_df.iterrows():
        name = str(row["指标"])
        if name.startswith("缺失值_"):
            lines.append(f"  {name}          : {row['值']}")

    lines += [
        "",
        f"  Close  最小值     : {_get('Close_最小值')}",
        f"  Close  最大值     : {_get('Close_最大值')}",
        f"  Volume 最小值     : {_get('Volume_最小值')}",
        f"  Volume 最大值     : {_get('Volume_最大值')}",
        "",
        f"  return_1d 均值    : {_get('return_1d_均值')}",
        f"  return_1d 标准差  : {_get('return_1d_标准差')}",
        f"  return_1d 最小值  : {_get('return_1d_最小值')}",
        f"  return_1d 最大值  : {_get('return_1d_最大值')}",
        "=" * 55,
    ]
    for line in lines:
        logger.info(line)


# ================================================================
# main 函数：支持 python src/data_quality.py
# ================================================================

def main(
    input_filename: str = DEFAULT_INPUT,
    output_filename: str = DEFAULT_OUTPUT,
):
    """
    数据质量报告主入口 —— 从 CSV 读取、生成报告、保存、打印摘要。

    用法:
        python src/data_quality.py
        python src/data_quality.py  <自定义输入>  <自定义输出>
    """
    input_path = DATA_PROCESSED / input_filename

    # 检查文件是否存在
    if not input_path.exists():
        logger.error("输入文件不存在: %s", input_path)
        # 尝试 fallback 到 SPY_clean_2015_2025.csv
        fallback_path = DATA_PROCESSED / "SPY_clean_2015_2025.csv"
        if fallback_path.exists():
            logger.warning("自动回退到 SPY_clean_2015_2025.csv")
            input_path = fallback_path
            input_filename = fallback_path.name
        else:
            logger.error("请先运行 run_stage1.py 下载并清洗数据")
            sys.exit(1)

    # 读取数据
    logger.info("读取数据: %s", input_path)
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    if df.empty:
        logger.error("数据文件为空: %s", input_path)
        sys.exit(1)

    # 从文件名提取 ticker
    ticker = _extract_ticker(input_filename)
    logger.info("ticker 识别为: %s", ticker)

    # 生成报告
    report = generate_quality_report(df, ticker=ticker)

    # 保存到 CSV
    save_quality_report(report, filename=output_filename)

    # 终端摘要
    _print_summary(report, ticker)

    logger.info("数据质量报告流程全部完成！")
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("用户中断")
        sys.exit(1)
    except Exception:
        logger.exception("运行失败")
        sys.exit(1)
