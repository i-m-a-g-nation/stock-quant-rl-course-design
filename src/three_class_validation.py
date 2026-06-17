"""
三分类交易信号验证模块（V2 Stage 9）

实现三分类方向预测 walk-forward 验证框架。
标签：1=明显上涨（可持仓），0=震荡（不交易），-1=明显下跌（空仓）
所有验证严格按时间顺序，不做随机打乱。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. 数据加载与准备
# ============================================================

def load_feature_data(path):
    """读取特征数据，解析日期，按日期升序排序。"""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")
    return df


def prepare_three_class_dataset(df, threshold=0.001):
    """
    生成三分类数据集。

    标签:
        next_return > threshold  → 1（上涨/可持仓）
        next_return < -threshold → -1（下跌/空仓）
        否则                     → 0（震荡/不交易）

    参数:
        df: 含有 Adj Close 或 Close 列的 DataFrame
        threshold: no-trade zone 阈值
    返回:
        X, y, next_return, dates, feature_columns
    """
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    next_return = df[price_col].shift(-1) / df[price_col] - 1

    # 三分类标签
    y = pd.Series(0, index=next_return.index, dtype=int)
    y[next_return > threshold] = 1
    y[next_return < -threshold] = -1

    exclude_cols = {
        "Date", "date", "target", "label", "three_class_label",
        "next_return", "future_return", "future_5d_return",
        "Close_next", "Adj Close_next",
        "Close", "Adj Close", "Open", "High", "Low", "Volume",
    }

    feature_columns = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    combined = pd.DataFrame({
        "y": y,
        "next_return": next_return,
    })
    for col in feature_columns:
        combined[col] = df[col]
    combined = combined.dropna()

    X = combined[feature_columns]
    y = combined["y"]
    next_return = combined["next_return"]
    dates = combined.index

    return X, y, next_return, dates, feature_columns


# ============================================================
# 2. Walk-forward 切分
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
# 3. 三分类评估
# ============================================================

def evaluate_three_class_classifier(model, X_train, y_train, X_test, y_test,
                                    fold_id, threshold, train_dates, test_dates):
    """评估三分类分类器。"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # macro / weighted 指标
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # 每类指标
    labels_order = [-1, 0, 1]
    prec_per_class = precision_score(y_test, y_pred, labels=labels_order, average=None, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, labels=labels_order, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, labels=labels_order, average=None, zero_division=0)

    # 类别分布
    true_counts = pd.Series(y_test).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()
    n = len(y_test)

    return {
        "fold_id": fold_id,
        "threshold": threshold,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "class_-1_precision": round(float(prec_per_class[0]), 4),
        "class_-1_recall": round(float(rec_per_class[0]), 4),
        "class_-1_f1": round(float(f1_per_class[0]), 4),
        "class_0_precision": round(float(prec_per_class[1]), 4),
        "class_0_recall": round(float(rec_per_class[1]), 4),
        "class_0_f1": round(float(f1_per_class[1]), 4),
        "class_1_precision": round(float(prec_per_class[2]), 4),
        "class_1_recall": round(float(rec_per_class[2]), 4),
        "class_1_f1": round(float(f1_per_class[2]), 4),
        "true_down_rate": round(float(true_counts.get(-1, 0) / n), 4),
        "true_neutral_rate": round(float(true_counts.get(0, 0) / n), 4),
        "true_up_rate": round(float(true_counts.get(1, 0) / n), 4),
        "pred_down_rate": round(float(pred_counts.get(-1, 0) / n), 4),
        "pred_neutral_rate": round(float(pred_counts.get(0, 0) / n), 4),
        "pred_up_rate": round(float(pred_counts.get(1, 0) / n), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_start": str(train_dates[0].date()) if len(train_dates) > 0 else "",
        "train_end": str(train_dates[-1].date()) if len(train_dates) > 0 else "",
        "test_start": str(test_dates[0].date()) if len(test_dates) > 0 else "",
        "test_end": str(test_dates[-1].date()) if len(test_dates) > 0 else "",
    }


def compute_confusion_matrix(y_test, y_pred, model_name, fold_id, threshold):
    """计算混淆矩阵并返回明细行。"""
    labels_order = [-1, 0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    rows = []
    for i, true_label in enumerate(labels_order):
        for j, pred_label in enumerate(labels_order):
            rows.append({
                "model": model_name,
                "fold_id": fold_id,
                "threshold": threshold,
                "true_label": true_label,
                "pred_label": pred_label,
                "count": int(cm[i][j]),
            })
    return rows


# ============================================================
# 4. 三分类回测
# ============================================================

def simple_three_class_backtest(y_pred, y_true, transaction_cost=0.0005):
    """
    三分类信号回测。只做 long/cash，不做做空。

    信号规则:
        prediction == 1 → 持仓 (position=1)
        prediction == 0 或 -1 → 空仓 (position=0)
    """
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    signal = (y_pred == 1).astype(float)

    pos_change = np.abs(np.diff(np.concatenate([[0], signal])))
    cost = pos_change * transaction_cost

    strategy_return = signal * y_true - cost

    equity = np.cumprod(1 + strategy_return)
    total_return = equity[-1] - 1.0 if len(equity) > 0 else 0.0

    n_days = len(strategy_return)
    ann_return = np.mean(strategy_return) * 252
    ann_vol = np.std(strategy_return) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    trades = int(np.sum(pos_change > 0))
    turnover = trades / n_days if n_days > 0 else 0.0
    avg_position = float(np.mean(signal))
    exposure_rate = float(np.mean(signal))
    no_trade_rate = float(np.mean(y_pred == 0)) + float(np.mean(y_pred == -1))

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


# ============================================================
# 5. 汇总
# ============================================================

def summarize_three_class_results(results_list, group_cols=("model", "threshold")):
    """按 model + threshold 汇总跨 fold 均值和标准差。"""
    df = pd.DataFrame(results_list)

    numeric_cols = [
        "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
        "class_-1_precision", "class_-1_recall", "class_-1_f1",
        "class_0_precision", "class_0_recall", "class_0_f1",
        "class_1_precision", "class_1_recall", "class_1_f1",
        "true_down_rate", "true_neutral_rate", "true_up_rate",
        "pred_down_rate", "pred_neutral_rate", "pred_up_rate",
    ]

    backtest_cols = [
        "total_return", "annual_return", "annual_volatility",
        "sharpe", "max_drawdown", "trades", "turnover",
        "avg_position", "exposure_rate", "no_trade_rate",
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
                vals = group[col].dropna()
                if len(vals) > 0:
                    row[f"{col}_mean"] = round(float(vals.mean()), 4)
                    row[f"{col}_std"] = round(float(vals.std()), 4)
                else:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. 绘图
# ============================================================

def plot_metric_by_fold(results_df, metric, ylabel, title, filepath,
                        group_col="model"):
    """绘制某指标按 fold 的变化图。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    models = results_df[group_col].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, model in enumerate(models):
        model_data = results_df[results_df[group_col] == model]
        ax.plot(
            model_data["fold_id"], model_data[metric],
            marker="o", linewidth=1.5, markersize=4,
            color=colors[i % len(colors)], label=model,
        )

    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(y_all, threshold, filepath):
    """绘制标签分布饼图。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    counts = y_all.value_counts().sort_index()
    labels = ["Down (-1)", "Neutral (0)", "Up (1)"]
    values = [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)]
    colors = ["#d62728", "#ffbb78", "#2ca02c"]

    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title(f"Label Distribution (threshold={threshold})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_comparison(summary_df, metric, ylabel, title, filepath):
    """比较不同 threshold 下的指标。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = summary_df["model"].unique()
    thresholds = sorted(summary_df["threshold"].unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x = np.arange(len(thresholds))
    width = 0.15

    for i, model in enumerate(models):
        model_data = summary_df[summary_df["model"] == model]
        vals = []
        for t in thresholds:
            row = model_data[model_data["threshold"] == t]
            if len(row) > 0 and metric in row.columns:
                vals.append(float(row[metric].values[0]))
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=model, color=colors[i % len(colors)])

    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{t:.4f}" for t in thresholds])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. 完整三分类 walk-forward 流程
# ============================================================

def run_three_class_walk_forward(
    feature_path,
    thresholds=(0.0005, 0.0010, 0.0020),
    models_config=None,
    initial_train_size=500,
    test_size=126,
    transaction_cost=0.0005,
    output_dir=None,
):
    """完整的三分类 walk-forward 验证流程。"""
    import logging
    logger = logging.getLogger("stage9")

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
    all_cm_rows = []
    all_predictions = []

    for threshold in thresholds:
        logger.info("\n=== Threshold = %.4f ===", threshold)

        X, y, next_return, dates, feature_columns = prepare_three_class_dataset(df, threshold=threshold)
        logger.info("样本数: %d, 特征数: %d", len(X), len(feature_columns))
        label_counts = y.value_counts().sort_index()
        logger.info("类别分布: down=%d (%.1f%%), neutral=%d (%.1f%%), up=%d (%.1f%%)",
                     label_counts.get(-1, 0), 100 * label_counts.get(-1, 0) / len(y),
                     label_counts.get(0, 0), 100 * label_counts.get(0, 0) / len(y),
                     label_counts.get(1, 0), 100 * label_counts.get(1, 0) / len(y))

        # 保存标签分布图
        if output_dir:
            from pathlib import Path
            out = Path(output_dir)
            (out / "figures").mkdir(parents=True, exist_ok=True)
            plot_label_distribution(
                y, threshold,
                out / "figures" / f"SPY_three_class_label_distribution_t{threshold:.4f}.png",
            )

        splits = make_walk_forward_splits(
            len(X), initial_train_size=initial_train_size, test_size=test_size,
        )
        logger.info("Walk-forward: %d folds", len(splits))

        for train_idx, test_idx, fold_id in splits:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            next_return_test = next_return.iloc[test_idx]
            train_dates = dates[train_idx]
            test_dates = dates[test_idx]

            for model_name, model_template in models_config.items():
                from sklearn.base import clone
                model = clone(model_template)

                # 评估
                metrics = evaluate_three_class_classifier(
                    model, X_train.values, y_train.values,
                    X_test.values, y_test.values,
                    fold_id, threshold, train_dates, test_dates,
                )
                metrics["model"] = model_name

                # 混淆矩阵
                model.fit(X_train.values, y_train.values)
                y_pred = model.predict(X_test.values)
                cm_rows = compute_confusion_matrix(y_test.values, y_pred, model_name, fold_id, threshold)
                all_cm_rows.extend(cm_rows)

                # 回测
                bt_metrics = simple_three_class_backtest(
                    y_pred, next_return_test.values, transaction_cost=transaction_cost,
                )
                metrics.update(bt_metrics)
                all_results.append(metrics)

                # 预测明细
                for i, date in enumerate(test_dates):
                    all_predictions.append({
                        "date": date,
                        "fold_id": fold_id,
                        "model": model_name,
                        "threshold": threshold,
                        "y_true": int(y_test.iloc[i]),
                        "y_pred": int(y_pred[i]),
                    })

    results_df = pd.DataFrame(all_results)
    cm_df = pd.DataFrame(all_cm_rows)
    predictions_df = pd.DataFrame(all_predictions)
    summary_df = summarize_three_class_results(all_results)

    # 保存
    if output_dir:
        out = Path(output_dir)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)

        results_df.to_csv(out / "tables" / "SPY_three_class_metrics.csv", index=False, encoding="utf-8-sig")

        bt_cols = ["model", "fold_id", "threshold", "total_return", "annual_return",
                    "annual_volatility", "sharpe", "max_drawdown", "trades",
                    "turnover", "avg_position", "exposure_rate", "no_trade_rate"]
        available_bt = [c for c in bt_cols if c in results_df.columns]
        results_df[available_bt].to_csv(out / "tables" / "SPY_three_class_backtest_metrics.csv", index=False, encoding="utf-8-sig")

        cm_df.to_csv(out / "tables" / "SPY_three_class_confusion_matrices.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(out / "tables" / "SPY_three_class_threshold_summary.csv", index=False, encoding="utf-8-sig")
        predictions_df.to_csv(out / "tables" / "SPY_three_class_predictions.csv", index=False, encoding="utf-8-sig")

        # 绘图（使用 threshold=0.0010 的结果）
        t_default = 0.0010
        df_t = results_df[results_df["threshold"] == t_default]
        if len(df_t) > 0:
            plot_metric_by_fold(df_t, "macro_f1", "Macro F1",
                                f"Three-Class Macro F1 by Fold (threshold={t_default})",
                                out / "figures" / "SPY_three_class_macro_f1_by_fold.png")
            plot_metric_by_fold(df_t, "balanced_accuracy", "Balanced Accuracy",
                                f"Three-Class Balanced Accuracy by Fold (threshold={t_default})",
                                out / "figures" / "SPY_three_class_balanced_accuracy_by_fold.png")
            plot_metric_by_fold(df_t, "total_return", "Total Return",
                                f"Three-Class Total Return by Fold (threshold={t_default})",
                                out / "figures" / "SPY_three_class_total_return_by_fold.png")

        # 标签分布图（合并）
        plot_label_distribution(
            y, t_default,
            out / "figures" / "SPY_three_class_label_distribution.png",
        )

        # threshold 比较图
        if len(summary_df) > 0:
            plot_threshold_comparison(summary_df, "macro_f1_mean", "Macro F1 (mean)",
                                      "Three-Class Macro F1 by Threshold",
                                      out / "figures" / "SPY_three_class_threshold_comparison.png")

        logger.info("所有结果已保存到 %s", output_dir)

    return summary_df, results_df
