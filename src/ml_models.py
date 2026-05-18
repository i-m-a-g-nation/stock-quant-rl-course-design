"""
机器学习基线模型模块：LogisticRegression + RandomForestClassifier 方向预测
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import DATA_FEATURES, OUTPUT_TABLES, OUTPUT_MODELS, setup_logging, timer

logger = setup_logging("ml_models")

# 输入/输出
INPUT_FILE = "SPY_features_2015_2025.csv"
METRICS_FILE = "SPY_ml_baseline_metrics.csv"
PREDICTIONS_FILE = "SPY_ml_test_predictions.csv"
FEATURE_IMPORTANCE_FILE = "SPY_ml_feature_importance.csv"
LR_MODEL_FILE = "SPY_logistic_regression.joblib"
RF_MODEL_FILE = "SPY_random_forest_classifier.joblib"

# 特征列（所有可用特征）
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "return_1d", "return_5d", "ma_5", "ma_10", "ma_20", "ma_60",
    "volatility_20", "volume_ma_20", "close_ma20_ratio",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42


@timer
def load_feature_data(filename: str = INPUT_FILE) -> pd.DataFrame:
    """加载特征数据集"""
    filepath = DATA_FEATURES / filename
    if not filepath.exists():
        raise FileNotFoundError(f"特征文件不存在: {filepath}\n请先运行 python run_stage2.py")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info("加载特征数据: %d 行, %d 列", len(df), len(df.columns))
    return df


@timer
def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造监督学习标签：
      target_return_1d = 下一交易日 return_1d
      target_direction  = target_return_1d > 0 ? 1 : 0
    """
    df["target_return_1d"] = df["return_1d"].shift(-1)
    df["target_direction"] = (df["target_return_1d"] > 0).astype(int)

    n_before = len(df)
    df = df.dropna(subset=["target_return_1d", "target_direction"])
    n_dropped = n_before - len(df)
    logger.info("标签构造完成，删除 NaN: %d 行 → %d 行 (丢弃 %d 行)",
                n_before, len(df), n_dropped)
    logger.info("方向分布: 上涨=%.1f%%, 下跌=%.1f%%",
                100 * df["target_direction"].mean(),
                100 * (1 - df["target_direction"].mean()))
    return df


@timer
def train_test_split_time_series(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple:
    """
    按时间顺序划分训练集和测试集（不打乱）。
    返回:
        (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    logger.info("时间序列划分: 训练 %d 行 (%.0f%%), 测试 %d 行 (%.0f%%)",
                len(train), 100 * (1 - test_size), len(test), 100 * test_size)
    logger.info("训练集: %s → %s", train.index.min().strftime("%Y-%m-%d"),
                train.index.max().strftime("%Y-%m-%d"))
    logger.info("测试集: %s → %s", test.index.min().strftime("%Y-%m-%d"),
                test.index.max().strftime("%Y-%m-%d"))
    return train, test


@timer
def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame):
    """
    训练两个基线模型并评估。
    返回:
        (metrics_df, fitted_models dict, predictions_df)
    """
    X_train = train[FEATURE_COLS].values
    y_train = train["target_direction"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["target_direction"].values

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    save_paths = {
        "LogisticRegression": OUTPUT_MODELS / LR_MODEL_FILE,
        "RandomForestClassifier": OUTPUT_MODELS / RF_MODEL_FILE,
    }

    rows = []
    fitted_models = {}
    pred_data = {
        "Date": test.index,
        "target_direction": y_test,
    }

    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logger.info("训练 %s ...", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        rows.append({"模型": name, "指标": "accuracy", "值": round(acc, 4)})
        rows.append({"模型": name, "指标": "precision", "值": round(prec, 4)})
        rows.append({"模型": name, "指标": "recall", "值": round(rec, 4)})
        rows.append({"模型": name, "指标": "f1", "值": round(f1, 4)})

        pred_data[f"{name}_pred"] = y_pred
        pred_data[f"{name}_prob"] = y_prob.round(4)

        # 保存模型
        filepath = save_paths[name]
        joblib.dump(model, filepath)
        logger.info("  %s 已保存: %s", name, filepath.name)
        logger.info("  accuracy=%.4f  precision=%.4f  recall=%.4f  f1=%.4f",
                    acc, prec, rec, f1)

    metrics_df = pd.DataFrame(rows)
    predictions_df = pd.DataFrame(pred_data)
    predictions_df = predictions_df.set_index("Date")

    logger.info("评估完成，共 %d 行指标", len(metrics_df))
    return metrics_df, fitted_models, predictions_df


@timer
def save_metrics(metrics_df: pd.DataFrame, filename: str = METRICS_FILE) -> Path:
    """保存评估指标到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    metrics_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("评估指标已保存: %s", filepath)
    return filepath


@timer
def save_predictions(predictions_df: pd.DataFrame, filename: str = PREDICTIONS_FILE) -> Path:
    """保存测试集预测结果到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    predictions_df.to_csv(filepath, encoding="utf-8-sig")
    logger.info("测试集预测结果已保存: %s", filepath)
    return filepath


@timer
def save_feature_importance(
    fitted_models: dict,
    filename: str = FEATURE_IMPORTANCE_FILE,
) -> Path:
    """
    提取并保存特征重要性。
    RandomForest: feature_importances_
    LogisticRegression: abs(coef_[0]) 作为重要性
    """
    rows = []
    for name, model in fitted_models.items():
        if name == "RandomForestClassifier":
            importance = model.feature_importances_
        elif name == "LogisticRegression":
            importance = np.abs(model.coef_[0])
        else:
            continue

        for feat, imp in zip(FEATURE_COLS, importance):
            rows.append({
                "model": name,
                "feature": feat,
                "importance": round(float(imp), 6),
            })

    imp_df = pd.DataFrame(rows)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    imp_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("特征重要性已保存: %s (%d 行)", filepath, len(imp_df))
    return filepath


@timer
def run_pipeline():
    """一键运行 ML 基线模型全流程"""
    logger.info("[Step 1/4] 加载特征数据")
    df = load_feature_data()

    logger.info("[Step 2/4] 构造标签 + 划分训练/测试集")
    df = build_labels(df)
    train, test = train_test_split_time_series(df)

    logger.info("[Step 3/4] 训练模型 + 评估")
    metrics, fitted_models, predictions = train_and_evaluate(train, test)
    save_metrics(metrics)
    save_predictions(predictions)

    logger.info("[Step 4/4] 特征重要性分析")
    save_feature_importance(fitted_models)

    logger.info("ML 基线模型训练全部完成！")
    return metrics
