"""
多周期预测与动态阈值验证模块（V2 Stage 10）

支持 1/5/10/20 日预测周期，二分类/三分类任务，
固定阈值/波动率动态阈值。
所有验证严格按时间顺序，不做随机打乱。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 数据加载
# ============================================================

def load_feature_data(path):
    """读取特征数据，解析日期，按日期升序排序。"""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")
    return df


# ============================================================
# 2. 多周期数据集准备
# ============================================================

def prepare_multi_horizon_dataset(
    df, horizon=1, task_type="binary",
    threshold_mode="fixed", fixed_threshold=None, vol_k=None,
):
    """
    生成多周期预测数据集。

    参数:
        df: 特征 DataFrame
        horizon: 预测周期 (1, 5, 10, 20)
        task_type: "binary" 或 "three_class"
        threshold_mode: "fixed" 或 "volatility_adjusted"
        fixed_threshold: 固定阈值（three_class + fixed 模式）
        vol_k: 波动率系数（three_class + volatility_adjusted 模式）
    返回:
        X, y, future_return, dates, feature_columns, threshold_series
    """
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    # 生成未来收益率
    future_return = df[price_col].shift(-horizon) / df[price_col] - 1

    # 获取波动率（用于动态阈值）
    if "volatility_20" in df.columns:
        rolling_vol = df["volatility_20"] / 100  # 原始值为百分比
    else:
        rolling_vol = df["return_1d"].rolling(window=20).std() / 100

    # 计算阈值序列
    if task_type == "three_class":
        if threshold_mode == "fixed":
            if fixed_threshold is None:
                fixed_threshold = _default_fixed_threshold(horizon)
            threshold_series = pd.Series(fixed_threshold, index=df.index)
        elif threshold_mode == "volatility_adjusted":
            if vol_k is None:
                vol_k = 0.5
            threshold_series = vol_k * rolling_vol * np.sqrt(horizon)
        else:
            threshold_series = pd.Series(0.001, index=df.index)
    else:
        threshold_series = pd.Series(0.0, index=df.index)

    # 生成标签
    if task_type == "binary":
        y = (future_return > 0).astype(int)
    else:  # three_class
        y = pd.Series(0, index=future_return.index, dtype=int)
        y[future_return > threshold_series] = 1
        y[future_return < -threshold_series] = -1

    # 排除的列
    exclude_cols = {
        "Date", "date", "target", "label", "three_class_label",
        "next_return", "future_return",
        "future_1d_return", "future_5d_return",
        "future_10d_return", "future_20d_return",
        "Close_next", "Adj Close_next",
        "Close", "Adj Close", "Open", "High", "Low", "Volume",
    }

    feature_columns = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 合并并删除缺失
    combined = pd.DataFrame({
        "y": y,
        "future_return": future_return,
        "threshold": threshold_series,
    })
    for col in feature_columns:
        combined[col] = df[col]
    combined = combined.dropna()

    X = combined[feature_columns]
    y = combined["y"]
    future_return = combined["future_return"]
    threshold_series = combined["threshold"]
    dates = combined.index

    return X, y, future_return, dates, feature_columns, threshold_series


def _default_fixed_threshold(horizon):
    """返回默认固定阈值。"""
    defaults = {1: 0.001, 5: 0.003, 10: 0.005, 20: 0.010}
    return defaults.get(horizon, 0.001)


# ============================================================
# 3. Walk-forward 切分
# ============================================================

def make_walk_forward_splits(n_samples, initial_train_size=500, test_size=126):
    """生成 expanding window walk-forward 切分索引。"""
    splits = []
    fold_id = 0
    train_end = initial_train_size

    while train_end + test_size <= n_samples:
        test_start = train_end
        test_end = min(train_end + test_size, n_samples)
        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, test_end))
        splits.append((train_indices, test_indices, fold_id))
        fold_id += 1
        train_end += test_size

    return splits


# ============================================================
# 4. 分类器评估
# ============================================================

def evaluate_classifier(
    model, X_train, y_train, X_test, y_test,
    fold_id, horizon, task_type, threshold_mode,
    threshold_value, vol_k, train_dates, test_dates,
):
    """评估分类器，自动区分 binary / three_class。"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    result = {
        "fold_id": fold_id,
        "horizon": horizon,
        "task_type": task_type,
        "threshold_mode": threshold_mode,
        "threshold_value": round(float(np.mean(threshold_value)), 6) if threshold_mode == "fixed" else np.nan,
        "vol_k": vol_k if threshold_mode == "volatility_adjusted" else np.nan,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_start": str(train_dates[0].date()) if len(train_dates) > 0 else "",
        "train_end": str(train_dates[-1].date()) if len(train_dates) > 0 else "",
        "test_start": str(test_dates[0].date()) if len(test_dates) > 0 else "",
        "test_end": str(test_dates[-1].date()) if len(test_dates) > 0 else "",
    }

    if task_type == "binary":
        has_proba = hasattr(model, "predict_proba")
        y_prob = None
        if has_proba:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                has_proba = False

        result["precision"] = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
        result["recall"] = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
        result["f1"] = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

        if has_proba and y_prob is not None and len(np.unique(y_test)) >= 2:
            try:
                result["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            except Exception:
                result["roc_auc"] = np.nan
        else:
            result["roc_auc"] = np.nan

        result["positive_rate_true"] = round(float(np.mean(y_test > 0)), 4)
        result["positive_rate_pred"] = round(float(np.mean(y_pred > 0)), 4)

    else:  # three_class
        labels_order = [-1, 0, 1]
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        f1_per = f1_score(y_test, y_pred, labels=labels_order, average=None, zero_division=0)
        pred_counts = pd.Series(y_pred).value_counts()
        n = len(y_pred)

        result["macro_precision"] = round(float(macro_prec), 4)
        result["macro_recall"] = round(float(macro_rec), 4)
        result["macro_f1"] = round(float(macro_f1), 4)
        result["weighted_f1"] = round(float(weighted_f1), 4)
        result["class_-1_f1"] = round(float(f1_per[0]), 4)
        result["class_0_f1"] = round(float(f1_per[1]), 4)
        result["class_1_f1"] = round(float(f1_per[2]), 4)
        result["pred_down_rate"] = round(float(pred_counts.get(-1, 0) / n), 4)
        result["pred_neutral_rate"] = round(float(pred_counts.get(0, 0) / n), 4)
        result["pred_up_rate"] = round(float(pred_counts.get(1, 0) / n), 4)

    return result


# ============================================================
# 5. 简化回测
# ============================================================

def simple_multi_horizon_backtest(
    y_pred, y_true, future_return, task_type="binary",
    transaction_cost=0.0005, horizon=1,
):
    """
    多周期信号简化回测。只做 long/cash。
    注意: h 日收益回测为简化近似，不代表真实逐日执行。
    长 horizon 下存在重叠收益和复利放大问题，total_return 仅用于同口径模型比较。
    """
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)
    future_return = np.array(future_return, dtype=float)

    if task_type == "binary":
        signal = (y_pred == 1).astype(float)
    else:
        signal = (y_pred == 1).astype(float)

    pos_change = np.abs(np.diff(np.concatenate([[0], signal])))
    cost = pos_change * transaction_cost

    strategy_return = signal * future_return - cost

    equity = np.cumprod(1 + strategy_return)
    total_return = equity[-1] - 1.0 if len(equity) > 0 else 0.0

    n_days = len(strategy_return)
    ann_factor = 252 / max(horizon, 1)
    ann_return = np.mean(strategy_return) * ann_factor
    ann_vol = np.std(strategy_return) * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    trades = int(np.sum(pos_change > 0))
    turnover = trades / n_days if n_days > 0 else 0.0
    avg_position = float(np.mean(signal))
    exposure_rate = float(np.mean(signal))
    no_trade_rate = 1.0 - exposure_rate

    return {
        "total_return": round(float(total_return), 4),
        "annual_return": round(float(ann_return), 4),
        "annual_volatility": round(float(ann_vol), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "trades": trades,
        "turnover": round(float(turnover), 4),
        "avg_position": round(float(avg_position), 4),
        "exposure_rate": round(float(exposure_rate), 4),
        "no_trade_rate": round(float(no_trade_rate), 4),
    }


def non_overlap_backtest(
    y_pred, y_true, future_return, task_type="binary",
    transaction_cost=0.0005, horizon=1,
):
    """
    Non-overlap 回测：每 horizon 天取一个样本，避免重叠收益复利放大。
    适用于长 horizon（h>1）场景，输出更保守的收益估计。
    """
    y_pred = np.array(y_pred, dtype=float)
    future_return = np.array(future_return, dtype=float)

    if task_type == "binary":
        signal = (y_pred == 1).astype(float)
    else:
        signal = (y_pred == 1).astype(float)

    # 每 horizon 天取一个样本
    step = max(horizon, 1)
    indices = list(range(0, len(signal), step))

    signal_sub = signal[indices]
    return_sub = future_return[indices]

    pos_change = np.abs(np.diff(np.concatenate([[0], signal_sub])))
    cost = pos_change * transaction_cost

    strategy_return = signal_sub * return_sub - cost

    equity = np.cumprod(1 + strategy_return)
    total_return = equity[-1] - 1.0 if len(equity) > 0 else 0.0

    n_periods = len(strategy_return)
    ann_factor = 252 / max(horizon, 1)
    ann_return = np.mean(strategy_return) * ann_factor
    ann_vol = np.std(strategy_return) * np.sqrt(ann_factor) if n_periods > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    trades = int(np.sum(pos_change > 0))

    return {
        "non_overlap_total_return": round(float(total_return), 4),
        "non_overlap_annual_return": round(float(ann_return), 4),
        "non_overlap_sharpe": round(float(sharpe), 4),
        "non_overlap_max_drawdown": round(float(max_dd), 4),
        "non_overlap_trades": trades,
        "non_overlap_n_periods": n_periods,
    }


# ============================================================
# 6. 汇总
# ============================================================

def summarize_results(results_list, group_cols=("model", "horizon", "task_type", "threshold_mode")):
    """按 model + horizon + task_type + threshold_mode 汇总。"""
    df = pd.DataFrame(results_list)

    # 确定指标列
    numeric_cols = [
        "accuracy", "balanced_accuracy",
        "precision", "recall", "f1", "roc_auc",
        "positive_rate_true", "positive_rate_pred",
        "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
        "class_-1_f1", "class_0_f1", "class_1_f1",
        "pred_down_rate", "pred_neutral_rate", "pred_up_rate",
    ]
    backtest_cols = [
        "total_return", "annual_return", "annual_volatility",
        "sharpe", "max_drawdown", "trades", "turnover",
        "avg_position", "exposure_rate", "no_trade_rate",
        "non_overlap_total_return", "non_overlap_annual_return",
        "non_overlap_sharpe", "non_overlap_max_drawdown",
        "non_overlap_trades", "non_overlap_n_periods",
    ]
    all_metrics = numeric_cols + [c for c in backtest_cols if c in df.columns]

    rows = []
    for keys, group in df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_folds"] = len(group)
        for col in all_metrics:
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(vals) > 0:
                    row[f"{col}_mean"] = round(float(vals.mean()), 4)
                    row[f"{col}_std"] = round(float(vals.std()), 4)
                else:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 7. 绘图
# ============================================================

def plot_metric_by_horizon(summary_df, metric, ylabel, title, filepath,
                           task_filter=None, threshold_filter=None):
    """按 horizon 绘制指标对比图。"""
    df = summary_df.copy()
    if task_filter:
        df = df[df["task_type"] == task_filter]
    if threshold_filter:
        df = df[df["threshold_mode"] == threshold_filter]

    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    models = df["model"].unique()
    horizons = sorted(df["horizon"].unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x = np.arange(len(horizons))
    width = 0.15

    for i, model in enumerate(models):
        model_data = df[df["model"] == model]
        vals = []
        for h in horizons:
            row = model_data[model_data["horizon"] == h]
            if len(row) > 0 and f"{metric}_mean" in row.columns:
                v = row[f"{metric}_mean"].values[0]
                vals.append(float(v) if not pd.isna(v) else 0)
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=model, color=colors[i % len(colors)])

    ax.set_xlabel("Horizon (days)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(label_dist_df, filepath):
    """绘制各 horizon 标签分布。"""
    fig, axes = plt.subplots(1, len(label_dist_df["horizon"].unique()),
                             figsize=(4 * len(label_dist_df["horizon"].unique()), 4))
    horizons = sorted(label_dist_df["horizon"].unique())
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        h_data = label_dist_df[label_dist_df["horizon"] == h]
        task = h_data["task_type"].iloc[0]
        if task == "binary":
            labels_list = ["Down (0)", "Up (1)"]
            values = [h_data[h_data["label"] == 0]["count"].sum(),
                      h_data[h_data["label"] == 1]["count"].sum()]
            colors = ["#d62728", "#2ca02c"]
        else:
            labels_list = ["Down (-1)", "Neutral (0)", "Up (1)"]
            values = [h_data[h_data["label"] == -1]["count"].sum(),
                      h_data[h_data["label"] == 0]["count"].sum(),
                      h_data[h_data["label"] == 1]["count"].sum()]
            colors = ["#d62728", "#ffbb78", "#2ca02c"]

        ax.pie(values, labels=labels_list, autopct="%1.1f%%", colors=colors, startangle=90)
        ax.set_title(f"{h}d ({task})", fontsize=11, fontweight="bold")

    fig.suptitle("Label Distribution by Horizon", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_by_model(summary_df, filepath):
    """绘制每个模型在不同 horizon 下的最佳表现。"""
    df = summary_df.copy()
    models = df["model"].unique()
    horizons = sorted(df["horizon"].unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        model_data = df[df["model"] == model]
        best_bal_acc = []
        for h in horizons:
            h_data = model_data[model_data["horizon"] == h]
            if len(h_data) > 0 and "balanced_accuracy_mean" in h_data.columns:
                best_bal_acc.append(float(h_data["balanced_accuracy_mean"].max()))
            else:
                best_bal_acc.append(0)
        ax.plot(horizons, best_bal_acc, marker="o", linewidth=1.5, markersize=6,
                color=colors[i % len(colors)], label=model)

    ax.set_xlabel("Horizon (days)", fontsize=11)
    ax.set_ylabel("Best Balanced Accuracy", fontsize=11)
    ax.set_title("Best Balanced Accuracy by Model and Horizon", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. 完整多周期 walk-forward 流程
# ============================================================

def run_multi_horizon_walk_forward(
    feature_path,
    horizons=(1, 5, 10, 20),
    task_types=("binary", "three_class"),
    threshold_mode="fixed",
    fixed_thresholds=None,
    vol_k_values=None,
    models_config=None,
    initial_train_size=500,
    test_size=126,
    transaction_cost=0.0005,
    output_dir=None,
):
    """完整的多周期 walk-forward 验证流程。"""
    import logging
    logger = logging.getLogger("stage10")

    if fixed_thresholds is None:
        fixed_thresholds = {1: 0.001, 5: 0.003, 10: 0.005, 20: 0.010}
    if vol_k_values is None:
        vol_k_values = [0.25, 0.50]

    if models_config is None:
        models_config = {
            "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
            "DummyStratified": DummyClassifier(strategy="stratified", random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5, class_weight="balanced"),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        }

    df = load_feature_data(feature_path)
    logger.info("数据维度: %d 行 × %d 列", len(df), len(df.columns))

    all_results = []
    all_label_dist = []

    for horizon in horizons:
        for task_type in task_types:
            logger.info("\n=== Horizon=%d, Task=%s ===", horizon, task_type)

            # 确定 threshold 配置
            if task_type == "binary":
                configs = [("fixed", None, None)]
            else:
                configs = []
                ft = fixed_thresholds.get(horizon, 0.001)
                configs.append(("fixed", ft, None))
                for vk in vol_k_values:
                    configs.append(("volatility_adjusted", None, vk))

            for thresh_mode, fixed_t, vol_k in configs:
                logger.info("  threshold_mode=%s, fixed=%.4f, vol_k=%s",
                            thresh_mode, fixed_t if fixed_t else 0, vol_k)

                X, y, future_return, dates, feature_columns, threshold_series = \
                    prepare_multi_horizon_dataset(
                        df, horizon=horizon, task_type=task_type,
                        threshold_mode=thresh_mode,
                        fixed_threshold=fixed_t, vol_k=vol_k,
                    )

                logger.info("  样本数: %d", len(X))

                # 记录标签分布
                label_counts = y.value_counts().sort_index()
                for lbl, cnt in label_counts.items():
                    all_label_dist.append({
                        "horizon": horizon,
                        "task_type": task_type,
                        "threshold_mode": thresh_mode,
                        "label": int(lbl),
                        "count": int(cnt),
                        "ratio": round(float(cnt / len(y)), 4),
                    })

                splits = make_walk_forward_splits(
                    len(X), initial_train_size=initial_train_size, test_size=test_size,
                )
                logger.info("  Walk-forward: %d folds", len(splits))

                for train_idx, test_idx, fold_id in splits:
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                    y_test = y.iloc[test_idx]
                    future_return_test = future_return.iloc[test_idx]
                    threshold_test = threshold_series.iloc[test_idx]
                    train_dates = dates[train_idx]
                    test_dates = dates[test_idx]

                    for model_name, model_template in models_config.items():
                        from sklearn.base import clone
                        model = clone(model_template)

                        # 评估
                        metrics = evaluate_classifier(
                            model, X_train.values, y_train.values,
                            X_test.values, y_test.values,
                            fold_id, horizon, task_type, thresh_mode,
                            threshold_test.values, vol_k,
                            train_dates, test_dates,
                        )
                        metrics["model"] = model_name

                        # 回测
                        model.fit(X_train.values, y_train.values)
                        y_pred = model.predict(X_test.values)
                        bt_metrics = simple_multi_horizon_backtest(
                            y_pred, y_test.values, future_return_test.values,
                            task_type=task_type, transaction_cost=transaction_cost,
                            horizon=horizon,
                        )
                        # non-overlap 回测（长 horizon 更保守的估计）
                        no_bt = non_overlap_backtest(
                            y_pred, y_test.values, future_return_test.values,
                            task_type=task_type, transaction_cost=transaction_cost,
                            horizon=horizon,
                        )
                        bt_metrics.update(no_bt)
                        metrics.update(bt_metrics)
                        all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    label_dist_df = pd.DataFrame(all_label_dist)
    summary_df = summarize_results(all_results)

    # 保存
    if output_dir:
        out = Path(output_dir)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)

        results_df.to_csv(out / "tables" / "SPY_multi_horizon_metrics.csv",
                          index=False, encoding="utf-8-sig")
        summary_df.to_csv(out / "tables" / "SPY_multi_horizon_summary.csv",
                          index=False, encoding="utf-8-sig")
        label_dist_df.to_csv(out / "tables" / "SPY_multi_horizon_label_distribution.csv",
                             index=False, encoding="utf-8-sig")

        # 绘图
        plot_metric_by_horizon(
            summary_df, "balanced_accuracy", "Balanced Accuracy",
            "Multi-Horizon Balanced Accuracy (binary)",
            out / "figures" / "SPY_multi_horizon_balanced_accuracy.png",
            task_filter="binary",
        )
        plot_metric_by_horizon(
            summary_df, "macro_f1", "Macro F1",
            "Multi-Horizon Macro F1 (three_class, fixed)",
            out / "figures" / "SPY_multi_horizon_macro_f1.png",
            task_filter="three_class", threshold_filter="fixed",
        )
        plot_metric_by_horizon(
            summary_df, "total_return", "Total Return",
            "Multi-Horizon Total Return (binary)",
            out / "figures" / "SPY_multi_horizon_total_return.png",
            task_filter="binary",
        )
        plot_label_distribution(label_dist_df, out / "figures" / "SPY_multi_horizon_label_distribution.png")
        plot_best_by_model(summary_df, out / "figures" / "SPY_multi_horizon_best_by_model.png")

        # non-overlap 回测图
        plot_metric_by_horizon(
            summary_df, "non_overlap_total_return", "Non-Overlap Total Return",
            "Multi-Horizon Non-Overlap Total Return (binary)",
            out / "figures" / "SPY_multi_horizon_non_overlap_total_return.png",
            task_filter="binary",
        )

        logger.info("所有结果已保存到 %s", output_dir)

    return summary_df, results_df, label_dist_df
