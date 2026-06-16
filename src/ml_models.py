"""
机器学习基线模型模块：LogisticRegression + RandomForestClassifier + DummyClassifier 方向预测
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import DATA_FEATURES, OUTPUT_TABLES, OUTPUT_MODELS, OUTPUT_FIGURES, setup_logging, timer

logger = setup_logging("ml_models")

# 输入/输出
INPUT_FILE = "SPY_features_2015_2025.csv"
METRICS_FILE = "SPY_ml_baseline_metrics.csv"
PREDICTIONS_FILE = "SPY_ml_test_predictions.csv"
FEATURE_IMPORTANCE_FILE = "SPY_ml_feature_importance.csv"
LR_MODEL_FILE = "SPY_logistic_regression.joblib"
RF_MODEL_FILE = "SPY_random_forest_classifier.joblib"

# 增强实验输出
ENHANCED_METRICS_FILE = "SPY_ml_model_metrics_enhanced.csv"
ENHANCED_PREDICTIONS_FILE = "SPY_ml_test_predictions_enhanced.csv"
CONFUSION_MATRIX_FILE = "SPY_ml_confusion_matrices.csv"
ENHANCED_PLOT_FILE = "SPY_ml_metrics_enhanced.png"
PERMUTATION_IMPORTANCE_FILE = "SPY_permutation_importance.csv"
PERMUTATION_IMPORTANCE_PLOT = "SPY_permutation_importance_top15.png"

# 特征列 - 原始版本
FEATURE_COLS_ORIGINAL = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "return_1d", "return_5d", "ma_5", "ma_10", "ma_20", "ma_60",
    "volatility_20", "volume_ma_20", "close_ma20_ratio",
]

# 特征列 - 相对化特征版本
FEATURE_COLS_RELATIVE = [
    "open_close_ratio", "high_low_ratio",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio_v2", "close_ma60_ratio",
    "volume_ratio_20", "volatility_change_5",
    "return_1d", "return_5d", "volatility_20",
]

# 特征列 - 全部特征
FEATURE_COLS_ALL = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "return_1d", "return_5d", "ma_5", "ma_10", "ma_20", "ma_60",
    "volatility_20", "volume_ma_20", "close_ma20_ratio",
    "open_close_ratio", "high_low_ratio",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio_v2", "close_ma60_ratio",
    "volume_ratio_20", "volatility_change_5",
]

# 默认使用全部特征
FEATURE_COLS = FEATURE_COLS_ALL

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
def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, feature_cols=None):
    """
    训练基线模型（含 DummyClassifier）并评估。
    返回:
        (metrics_df, fitted_models dict, predictions_df)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # 确保只使用数据中存在的特征列
    available_cols = [c for c in feature_cols if c in train.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        logger.warning("部分特征列不存在，已跳过: %s", missing)

    X_train = train[available_cols].values
    y_train = train["target_direction"].values
    X_test = test[available_cols].values
    y_test = test["target_direction"].values

    models = {
        "DummyMostFrequent": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "DummyStratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
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
    confusion_rows = []
    fitted_models = {}
    pred_data = {
        "Date": test.index,
        "target_direction": y_test,
    }

    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

    positive_rate_true = round(float(y_test.mean()), 4)

    for name, model in models.items():
        logger.info("训练 %s ...", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        y_pred = model.predict(X_test)

        # 尝试获取概率
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None
                has_proba = False
        else:
            y_prob = None

        # 基础指标
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # AUC
        auc_val = np.nan
        if has_proba and y_prob is not None:
            try:
                auc_val = roc_auc_score(y_test, y_prob)
            except Exception:
                auc_val = np.nan

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, len(y_test)

        positive_rate_pred = round(float(y_pred.mean()), 4)

        rows.append({"模型": name, "指标": "accuracy", "值": round(acc, 4)})
        rows.append({"模型": name, "指标": "balanced_accuracy", "值": round(bal_acc, 4)})
        rows.append({"模型": name, "指标": "precision", "值": round(prec, 4)})
        rows.append({"模型": name, "指标": "recall", "值": round(rec, 4)})
        rows.append({"模型": name, "指标": "f1", "值": round(f1, 4)})
        rows.append({"模型": name, "指标": "roc_auc", "值": round(auc_val, 4) if not np.isnan(auc_val) else "NaN"})
        rows.append({"模型": name, "指标": "positive_rate_pred", "值": positive_rate_pred})
        rows.append({"模型": name, "指标": "positive_rate_true", "值": positive_rate_true})

        # 混淆矩阵行
        confusion_rows.append({
            "模型": name, "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        })

        # 预测数据
        pred_data[f"{name}_pred"] = y_pred
        if has_proba and y_prob is not None:
            pred_data[f"{name}_prob"] = y_prob.round(4)

        # 保存模型（仅 LR 和 RF）
        if name in save_paths:
            filepath = save_paths[name]
            joblib.dump(model, filepath)
            logger.info("  %s 已保存: %s", name, filepath.name)

        logger.info("  %s: acc=%.4f bal_acc=%.4f f1=%.4f auc=%s pred_pos=%.2f%%",
                    name, acc, bal_acc, f1,
                    f"{auc_val:.4f}" if not np.isnan(auc_val) else "NaN",
                    positive_rate_pred * 100)

    metrics_df = pd.DataFrame(rows)
    confusion_df = pd.DataFrame(confusion_rows)
    predictions_df = pd.DataFrame(pred_data)
    predictions_df = predictions_df.set_index("Date")

    logger.info("评估完成，共 %d 行指标", len(metrics_df))
    return metrics_df, fitted_models, predictions_df, confusion_df


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
    feature_cols=None,
) -> Path:
    """
    提取并保存特征重要性。
    RandomForest: feature_importances_
    LogisticRegression: abs(coef_[0]) 作为重要性
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    rows = []
    for name, model in fitted_models.items():
        if name == "RandomForestClassifier":
            importance = model.feature_importances_
        elif name == "LogisticRegression":
            importance = np.abs(model.coef_[0])
        else:
            continue

        for feat, imp in zip(feature_cols, importance):
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
def save_enhanced_metrics(metrics_df: pd.DataFrame, filename: str = ENHANCED_METRICS_FILE) -> Path:
    """保存增强评估指标到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    metrics_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("增强评估指标已保存: %s", filepath)
    return filepath


@timer
def save_enhanced_predictions(predictions_df: pd.DataFrame, filename: str = ENHANCED_PREDICTIONS_FILE) -> Path:
    """保存增强测试集预测结果到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    predictions_df.to_csv(filepath, encoding="utf-8-sig")
    logger.info("增强测试集预测结果已保存: %s", filepath)
    return filepath


@timer
def save_confusion_matrices(confusion_df: pd.DataFrame, filename: str = CONFUSION_MATRIX_FILE) -> Path:
    """保存混淆矩阵到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    confusion_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("混淆矩阵已保存: %s", filepath)
    return filepath


@timer
def plot_enhanced_metrics(metrics_df: pd.DataFrame, filename: str = ENHANCED_PLOT_FILE) -> Path:
    """绘制模型指标对比图"""
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_FIGURES / filename

    # 筛选要绘制的指标
    plot_metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    models = metrics_df["模型"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(plot_metrics))
    width = 0.18
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, model in enumerate(models):
        model_data = metrics_df[metrics_df["模型"] == model]
        values = []
        for m in plot_metrics:
            row = model_data[model_data["指标"] == m]
            if len(row) > 0:
                val = row["值"].values[0]
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    values.append(0)
            else:
                values.append(0)
        ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])

    ax.set_xlabel("Metrics", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("SPY ML Model Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(plot_metrics)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("模型指标对比图已保存: %s", filepath)
    return filepath


@timer
def compute_permutation_importance(
    fitted_models: dict,
    test: pd.DataFrame,
    feature_cols=None,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """计算 permutation importance"""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    available_cols = [c for c in feature_cols if c in test.columns]
    X_test = test[available_cols].values
    y_test = test["target_direction"].values

    rows = []
    for name, model in fitted_models.items():
        if name in ("DummyMostFrequent", "DummyStratified"):
            continue

        logger.info("计算 %s 的 permutation importance ...", name)

        # 尝试用 roc_auc，失败则用 balanced_accuracy
        scoring = "roc_auc"
        try:
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=n_repeats, random_state=RANDOM_STATE,
                scoring=scoring,
            )
        except Exception:
            scoring = "balanced_accuracy"
            try:
                result = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=n_repeats, random_state=RANDOM_STATE,
                    scoring=scoring,
                )
            except Exception:
                logger.warning("  %s permutation importance 计算失败，跳过", name)
                continue

        for feat, imp_mean, imp_std in zip(available_cols, result.importances_mean, result.importances_std):
            rows.append({
                "model": name,
                "feature": feat,
                "importance_mean": round(float(imp_mean), 6),
                "importance_std": round(float(imp_std), 6),
                "scoring": scoring,
            })

        logger.info("  %s 完成 (scoring=%s)", name, scoring)

    imp_df = pd.DataFrame(rows)
    return imp_df


@timer
def save_permutation_importance(imp_df: pd.DataFrame, filename: str = PERMUTATION_IMPORTANCE_FILE) -> Path:
    """保存 permutation importance 到 outputs/tables/"""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_TABLES / filename
    imp_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info("Permutation importance 已保存: %s (%d 行)", filepath, len(imp_df))
    return filepath


@timer
def plot_permutation_importance_top15(imp_df: pd.DataFrame, filename: str = PERMUTATION_IMPORTANCE_PLOT) -> Path:
    """绘制前 15 个重要特征的 permutation importance"""
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_FIGURES / filename

    models = imp_df["model"].unique()
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_data = imp_df[imp_df["model"] == model].copy()
        model_data = model_data.sort_values("importance_mean", ascending=True).tail(15)

        ax.barh(model_data["feature"], model_data["importance_mean"],
                xerr=model_data["importance_std"], color="#1f77b4", alpha=0.8)
        ax.set_xlabel("Importance (mean ± std)", fontsize=10)
        ax.set_title(f"Permutation Importance — {model}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Top 15 Permutation Importance Features", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Permutation importance 图已保存: %s", filepath)
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
    metrics, fitted_models, predictions, confusion_df = train_and_evaluate(train, test)
    save_metrics(metrics)
    save_predictions(predictions)

    # 保存增强输出
    save_enhanced_metrics(metrics)
    save_enhanced_predictions(predictions)
    save_confusion_matrices(confusion_df)
    plot_enhanced_metrics(metrics)

    logger.info("[Step 4/4] 特征重要性分析")
    save_feature_importance(fitted_models)

    logger.info("[Step 5/5] Permutation importance 分析")
    perm_imp = compute_permutation_importance(fitted_models, test)
    if len(perm_imp) > 0:
        save_permutation_importance(perm_imp)
        plot_permutation_importance_top15(perm_imp)

    logger.info("ML 基线模型训练全部完成！")
    return metrics
