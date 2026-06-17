"""
Walk-forward 时间序列验证模块（V2）

实现 expanding window 验证框架，用于检查模型在不同时间段是否稳定。
所有验证严格按时间顺序进行，不做随机打乱，避免未来信息泄漏。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. 数据加载与准备
# ============================================================

def load_feature_data(path):
    """
    读取特征数据，解析日期，按日期升序排序。
    参数:
        path: CSV 文件路径
    返回:
        DataFrame，以 Date 为索引
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")
    return df


def prepare_direction_dataset(df):
    """
    生成方向分类数据集。
    目标: next_return > 0 → 1, 否则 0。

    参数:
        df: 含有 Close 列的 DataFrame
    返回:
        X: 特征 DataFrame
        y: 目标 Series
        next_return: 下一交易日收益率 Series
        dates: 日期索引
        feature_columns: 使用的特征列名列表
    """
    # 计算下一交易日收益率
    next_return = df["Close"].pct_change().shift(-1) * 100

    # 构造目标
    target = (next_return > 0).astype(int)

    # 排除的列
    exclude_cols = {
        "Date", "date", "target", "label", "next_return",
        "future_return", "Close_next", "Adj Close_next",
        "Close", "Adj Close", "Open", "High", "Low", "Volume",
    }

    # 选择数值型特征列
    feature_columns = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 合并并删除缺失
    combined = pd.DataFrame({
        "target": target,
        "next_return": next_return,
    })
    for col in feature_columns:
        combined[col] = df[col]

    combined = combined.dropna()

    X = combined[feature_columns]
    y = combined["target"]
    next_return = combined["next_return"]
    dates = combined.index

    return X, y, next_return, dates, feature_columns


# ============================================================
# 2. Walk-forward 切分
# ============================================================

def make_walk_forward_splits(
    n_samples,
    initial_train_size=500,
    test_size=126,
    min_train_size=None,
):
    """
    生成 expanding window 的 walk-forward 切分索引。

    参数:
        n_samples: 总样本数
        initial_train_size: 初始训练集大小
        test_size: 每个测试窗口大小
        min_train_size: 最小训练集大小（默认等于 initial_train_size）
    返回:
        list of (train_indices, test_indices, fold_id)
    """
    if min_train_size is None:
        min_train_size = initial_train_size

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
# 3. 安全 AUC 计算
# ============================================================

def safe_roc_auc(y_true, y_prob):
    """
    安全计算 ROC-AUC。如果测试集只有一个类别，返回 NaN。
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


# ============================================================
# 4. 分类器评估
# ============================================================

def evaluate_classifier(model, X_train, y_train, X_test, y_test, fold_id,
                        train_dates, test_dates):
    """
    评估单个分类器在单个 fold 上的表现。

    返回:
        dict，包含各项指标
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 概率预测
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
            has_proba = False
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = safe_roc_auc(y_test, y_prob) if has_proba else np.nan

    positive_rate_true = float(y_test.mean())
    positive_rate_pred = float(y_pred.mean())

    return {
        "fold_id": fold_id,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else np.nan,
        "positive_rate_true": round(positive_rate_true, 4),
        "positive_rate_pred": round(positive_rate_pred, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_start": str(train_dates[0].date()) if len(train_dates) > 0 else "",
        "train_end": str(train_dates[-1].date()) if len(train_dates) > 0 else "",
        "test_start": str(test_dates[0].date()) if len(test_dates) > 0 else "",
        "test_end": str(test_dates[-1].date()) if len(test_dates) > 0 else "",
    }


# ============================================================
# 5. 简化信号回测
# ============================================================

def simple_signal_backtest(y_pred, next_return_test, transaction_cost=0.0005):
    """
    对预测信号做简化回测。
    position = y_pred（1=持有，0=空仓）
    策略收益 = position * next_return - 交易成本

    参数:
        y_pred: 预测信号（0 或 1）
        next_return_test: 下一交易日收益率（百分比）
        transaction_cost: 交易成本（仅仓位变化时扣除）
    返回:
        dict，包含回测指标
    """
    position = np.array(y_pred, dtype=float)
    returns = np.array(next_return_test, dtype=float) / 100.0  # 转为小数

    # 计算仓位变化和交易成本
    pos_change = np.abs(np.diff(np.concatenate([[0], position])))
    cost = pos_change * transaction_cost

    strategy_return = position * returns - cost

    # 累计收益
    equity = np.cumprod(1 + strategy_return)
    total_return = equity[-1] - 1.0 if len(equity) > 0 else 0.0

    # 年化收益
    n_days = len(strategy_return)
    ann_return = np.mean(strategy_return) * 252

    # 年化波动率
    ann_vol = np.std(strategy_return) * np.sqrt(252)

    # 夏普比率
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # 最大回撤
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = np.min(drawdown)

    # 交易次数
    trades = int(np.sum(pos_change > 0))

    # 换手率
    turnover = trades / n_days if n_days > 0 else 0.0

    return {
        "total_return": round(float(total_return), 4),
        "annual_return": round(float(ann_return), 4),
        "annual_volatility": round(float(ann_vol), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "trades": trades,
        "turnover": round(float(turnover), 4),
    }


# ============================================================
# 6. 汇总跨 fold 结果
# ============================================================

def summarize_walk_forward(results_list, group_col="model"):
    """
    按模型汇总跨 fold 的均值和标准差。

    参数:
        results_list: list of dict
        group_col: 分组列名
    返回:
        DataFrame
    """
    df = pd.DataFrame(results_list)

    # 数值列
    numeric_cols = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "roc_auc", "positive_rate_true", "positive_rate_pred",
    ]

    # 如果有回测列
    backtest_cols = [
        "total_return", "annual_return", "annual_volatility",
        "sharpe", "max_drawdown", "trades", "turnover",
    ]

    all_metrics = numeric_cols + [c for c in backtest_cols if c in df.columns]

    rows = []
    for model_name, group in df.groupby(group_col):
        row = {"model": model_name, "n_folds": len(group)}
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
# 7. 绘图函数
# ============================================================

def plot_metric_by_fold(results_df, metric, ylabel, title, filepath):
    """
    绘制某指标按 fold 的变化图。
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    models = results_df["model"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, model in enumerate(models):
        model_data = results_df[results_df["model"] == model]
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


def plot_backtest_metric_by_fold(results_df, metric, ylabel, title, filepath):
    """
    绘制回测指标按 fold 的变化图。
    """
    if metric not in results_df.columns:
        return
    plot_metric_by_fold(results_df, metric, ylabel, title, filepath)


# ============================================================
# 8. 完整 walk-forward 流程
# ============================================================

def run_walk_forward(
    feature_path,
    models_config=None,
    initial_train_size=500,
    test_size=126,
    transaction_cost=0.0005,
    output_dir=None,
):
    """
    完整的 walk-forward 验证流程。

    参数:
        feature_path: 特征 CSV 路径
        models_config: 模型配置字典
        initial_train_size: 初始训练集大小
        test_size: 每个测试窗口大小
        transaction_cost: 交易成本
        output_dir: 输出目录
    返回:
        summary_df, all_results_df
    """
    import logging
    logger = logging.getLogger("stage7")

    if models_config is None:
        models_config = {
            "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
            "DummyStratified": DummyClassifier(strategy="stratified", random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=200, random_state=42, max_depth=5,
            ),
        }

    # 加载数据
    logger.info("加载特征数据: %s", feature_path)
    df = load_feature_data(feature_path)
    logger.info("数据维度: %d 行 × %d 列", len(df), len(df.columns))

    # 准备数据集
    X, y, next_return, dates, feature_columns = prepare_direction_dataset(df)
    logger.info("样本数: %d, 特征数: %d", len(X), len(feature_columns))
    logger.info("目标分布: 上涨=%.1f%%, 下跌=%.1f%%", 100 * y.mean(), 100 * (1 - y.mean()))

    # 生成 walk-forward 切分
    splits = make_walk_forward_splits(
        len(X), initial_train_size=initial_train_size, test_size=test_size,
    )
    logger.info("Walk-forward 切分: %d 个 fold", len(splits))

    if len(splits) == 0:
        logger.error("无法生成足够的 fold，请检查参数")
        return pd.DataFrame(), pd.DataFrame()

    # 运行所有 fold
    all_results = []

    for train_idx, test_idx, fold_id in splits:
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        next_return_test = next_return.iloc[test_idx]
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]

        logger.info("Fold %d: train=%s~%s (%d), test=%s~%s (%d)",
                     fold_id,
                     train_dates[0].date(), train_dates[-1].date(), len(train_idx),
                     test_dates[0].date(), test_dates[-1].date(), len(test_idx))

        for model_name, model_template in models_config.items():
            # 克隆模型
            from sklearn.base import clone
            model = clone(model_template)

            # 分类评估
            metrics = evaluate_classifier(
                model, X_train.values, y_train.values,
                X_test.values, y_test.values,
                fold_id, train_dates, test_dates,
            )
            metrics["model"] = model_name

            # 简化回测
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict(X_test.values)
            bt_metrics = simple_signal_backtest(
                y_pred, next_return_test.values, transaction_cost=transaction_cost,
            )
            metrics.update(bt_metrics)

            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)

    # 汇总
    summary_df = summarize_walk_forward(results_df)
    logger.info("Walk-forward 验证完成")

    # 保存结果
    if output_dir is not None:
        output_dir = Path(output_dir)
        (output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(parents=True, exist_ok=True)

        # 保存明细
        results_df.to_csv(
            output_dir / "tables" / "SPY_walk_forward_classification_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("分类指标明细已保存")

        # 保存回测指标
        bt_cols = [
            "model", "fold_id", "total_return", "annual_return",
            "annual_volatility", "sharpe", "max_drawdown", "trades", "turnover",
        ]
        available_bt_cols = [c for c in bt_cols if c in results_df.columns]
        results_df[available_bt_cols].to_csv(
            output_dir / "tables" / "SPY_walk_forward_backtest_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("回测指标明细已保存")

        # 保存预测（包含日期和真实值）
        pred_rows = []
        for train_idx, test_idx, fold_id in splits:
            test_dates_fold = dates[test_idx]
            y_test_fold = y.iloc[test_idx]
            for model_name in models_config.keys():
                model_results = results_df[
                    (results_df["model"] == model_name) & (results_df["fold_id"] == fold_id)
                ]
                if len(model_results) > 0:
                    for i, date in enumerate(test_dates_fold):
                        pred_rows.append({
                            "date": date,
                            "fold_id": fold_id,
                            "model": model_name,
                            "y_true": int(y_test_fold.iloc[i]),
                        })
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(
            output_dir / "tables" / "SPY_walk_forward_predictions.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("预测明细已保存")

        # 保存汇总
        summary_df.to_csv(
            output_dir / "tables" / "SPY_walk_forward_summary.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("汇总指标已保存")

        # 绘图
        plot_metric_by_fold(
            results_df, "roc_auc", "ROC-AUC",
            "Walk-Forward ROC-AUC by Fold",
            output_dir / "figures" / "SPY_walk_forward_roc_auc_by_fold.png",
        )
        plot_metric_by_fold(
            results_df, "balanced_accuracy", "Balanced Accuracy",
            "Walk-Forward Balanced Accuracy by Fold",
            output_dir / "figures" / "SPY_walk_forward_balanced_accuracy_by_fold.png",
        )
        plot_backtest_metric_by_fold(
            results_df, "total_return", "Total Return",
            "Walk-Forward Total Return by Fold",
            output_dir / "figures" / "SPY_walk_forward_total_return_by_fold.png",
        )
        logger.info("图表已保存")

    return summary_df, results_df
