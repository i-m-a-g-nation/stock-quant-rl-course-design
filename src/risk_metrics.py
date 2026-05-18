"""
风险指标模块：市场收益统计、风险指标计算、方向命中率分析
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import DATA_PROCESSED, OUTPUT_TABLES, setup_logging, timer

logger = setup_logging("risk_metrics")

# 输入
CLEAN_FILE = "SPY_clean_2015_2025.csv"
PREDICTIONS_FILE = "SPY_ml_test_predictions.csv"

# 输出
RETURNS_FILE = "SPY_market_returns.csv"
RISK_METRICS_FILE = "SPY_risk_metrics.csv"
DIRECTION_ANALYSIS_FILE = "SPY_prediction_direction_analysis.csv"

TRADING_DAYS = 252


@timer
def load_clean_data(filename: str = CLEAN_FILE) -> pd.DataFrame:
    """加载清洗数据"""
    filepath = DATA_PROCESSED / filename
    if not filepath.exists():
        raise FileNotFoundError(f"清洗数据不存在: {filepath}\n请先运行 python run_stage1.py")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info("加载清洗数据: %d 行", len(df))
    return df


@timer
def compute_market_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算市场日收益率、累计收益、回撤。
    返回:
        DataFrame (Date, Close, market_return_1d, cumulative_return, drawdown)
    """
    ret = df[["Close"]].copy()
    ret["market_return_1d"] = ret["Close"].pct_change()
    ret["cumulative_return"] = (1 + ret["market_return_1d"]).cumprod()
    cummax = ret["cumulative_return"].cummax()
    ret["drawdown"] = (ret["cumulative_return"] - cummax) / cummax
    # 删除首行 NaN
    ret = ret.dropna(subset=["market_return_1d"])
    logger.info("收益率序列计算完成: %d 行", len(ret))
    return ret


@timer
def save_market_returns(ret: pd.DataFrame, filename: str = RETURNS_FILE) -> Path:
    """保存收益率序列到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    ret.to_csv(filepath, encoding="utf-8-sig")
    logger.info("收益率序列已保存: %s", filepath)
    return filepath


@timer
def compute_risk_metrics(ret: pd.DataFrame) -> pd.DataFrame:
    """
    计算基础风险/统计指标。
    返回:
        指标 DataFrame (指标, 值)
    """
    returns = ret["market_return_1d"].dropna()

    annual_return = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

    # 最大回撤
    cumret = ret["cumulative_return"]
    cummax = cumret.cummax()
    dd = (cumret - cummax) / cummax
    max_drawdown = dd.min()

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    positive_day_ratio = (returns > 0).mean()

    rows = [
        {"指标": "annual_return",        "值": round(annual_return, 6)},
        {"指标": "annual_volatility",    "值": round(annual_volatility, 6)},
        {"指标": "sharpe_ratio",         "值": round(sharpe_ratio, 4)},
        {"指标": "max_drawdown",         "值": round(max_drawdown, 6)},
        {"指标": "calmar_ratio",         "值": round(calmar_ratio, 4)},
        {"指标": "skewness",            "值": round(skewness, 4)},
        {"指标": "kurtosis",            "值": round(kurtosis, 4)},
        {"指标": "var_95",              "值": round(var_95, 6)},
        {"指标": "cvar_95",             "值": round(cvar_95, 6)},
        {"指标": "positive_day_ratio",  "值": round(positive_day_ratio, 4)},
    ]

    report = pd.DataFrame(rows)
    logger.info("风险指标计算完成，共 %d 项", len(report))
    return report


@timer
def save_risk_metrics(report: pd.DataFrame, filename: str = RISK_METRICS_FILE) -> Path:
    """保存风险指标到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    report.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("风险指标已保存: %s", filepath)
    return filepath


@timer
def compute_direction_analysis(predictions_file: str = None) -> pd.DataFrame:
    """
    基于阶段 3 预测结果计算方向命中率统计。
    仅做分类统计，不生成交易信号、不计算策略收益。

    返回:
        DataFrame 或 None（如果预测文件不存在）
    """
    fname = predictions_file or PREDICTIONS_FILE
    filepath = OUTPUT_TABLES / fname
    if not filepath.exists():
        logger.info("预测文件不存在，跳过方向分析: %s", filepath)
        return None

    pred = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if "RandomForestClassifier_pred" not in pred.columns:
        logger.warning("预测文件中未找到 RandomForestClassifier_pred 列，跳过方向分析")
        return None

    total = len(pred)
    correct = int((pred["RandomForestClassifier_pred"] == pred["target_direction"]).sum())
    accuracy = correct / total

    pos_pred_ratio = pred["RandomForestClassifier_pred"].mean()
    actual_pos_ratio = pred["target_direction"].mean()

    rows = [
        {"指标": "total_test_days",         "值": total},
        {"指标": "correct_direction_days",  "值": correct},
        {"指标": "direction_accuracy",      "值": round(accuracy, 4)},
        {"指标": "positive_prediction_ratio", "值": round(pos_pred_ratio, 4)},
        {"指标": "actual_positive_ratio",   "值": round(actual_pos_ratio, 4)},
    ]

    analysis = pd.DataFrame(rows)
    logger.info("方向命中率分析完成: accuracy=%.4f", accuracy)
    return analysis


@timer
def save_direction_analysis(analysis: pd.DataFrame, filename: str = DIRECTION_ANALYSIS_FILE) -> Path:
    """保存方向命中率统计"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    analysis.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("方向命中率分析已保存: %s", filepath)
    return filepath


@timer
def run_pipeline():
    """一键运行风险指标评估全流程"""
    logger.info("[Step 1/3] 加载数据 + 计算收益率序列")
    df = load_clean_data()
    ret = compute_market_returns(df)
    save_market_returns(ret)

    logger.info("[Step 2/3] 计算风险指标")
    risk_report = compute_risk_metrics(ret)
    save_risk_metrics(risk_report)

    logger.info("[Step 3/3] 预测方向命中率分析")
    direction_analysis = compute_direction_analysis()
    if direction_analysis is not None:
        save_direction_analysis(direction_analysis)

    # 打印摘要
    logger.info("--- 风险指标摘要 ---")
    for _, row in risk_report.iterrows():
        logger.info("  %s = %s", row["指标"], row["值"])

    if direction_analysis is not None:
        logger.info("--- 预测方向分析摘要 ---")
        for _, row in direction_analysis.iterrows():
            logger.info("  %s = %s", row["指标"], row["值"])

    logger.info("风险指标评估全部完成！")
    return risk_report
