"""
收益率回归验证模块（V2 Stage 8）

实现 walk-forward 时间序列回归验证框架。
预测目标为下一交易日收益率 next_return，不是价格。
所有验证严格按时间顺序进行，不做随机打乱。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. 数据加载与准备
# ============================================================

def load_feature_data(path):
    """
    读取特征数据，解析日期，按日期升序排序。
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")
    return df


def prepare_return_regression_dataset(df, horizon=1):
    """
    生成收益率回归数据集。

    目标:
        horizon=1: next_return = Adj Close.shift(-1) / Adj Close - 1
        horizon=5: future_5d_return = Adj Close.shift(-5) / Adj Close - 1

    参数:
        df: 含有 Adj Close 或 Close 列的 DataFrame
        horizon: 预测周期（1 或 5）
    返回:
        X: 特征 DataFrame
        y_return: 目标收益率 Series
        dates: 日期索引
        feature_columns: 使用的特征列名列表
    """
    # 选择价格列
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    # 生成目标收益率
    if horizon == 1:
        y_return = df[price_col].shift(-1) / df[price_col] - 1
    elif horizon == 5:
        y_return = df[price_col].shift(-5) / df[price_col] - 1
    else:
        y_return = df[price_col].shift(-horizon) / df[price_col] - 1

    # 排除的列
    exclude_cols = {
        "Date", "date", "target", "label", "next_return",
        "future_return", "future_5d_return", "Close_next", "Adj Close_next",
        "Close", "Adj Close", "Open", "High", "Low", "Volume",
    }

    # 选择数值型特征列
    feature_columns = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 合并并删除缺失
    combined = pd.DataFrame({"y_return": y_return})
    for col in feature_columns:
        combined[col] = df[col]

    combined = combined.dropna()

    X = combined[feature_columns]
    y_return = combined["y_return"]
    dates = combined.index

    return X, y_return, dates, feature_columns


# ============================================================
# 2. Walk-forward 切分（与 Stage 7 相同逻辑，不修改 validation.py）
# ============================================================

def make_walk_forward_splits(
    n_samples,
    initial_train_size=500,
    test_size=126,
    min_train_size=None,
):
    """
    生成 expanding window 的 walk-forward 切分索引。
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
# 3. 回归器评估
# ============================================================

def evaluate_regressor(model, X_train, y_train, X_test, y_test, fold_id,
                       train_dates, test_dates):
    """
    评估单个回归器在单个 fold 上的表现。
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 相关系数
    pearson_corr = np.nan
    spearman_corr = np.nan
    if len(y_pred) > 2:
        try:
            pearson_corr, _ = stats.pearsonr(y_test, y_pred)
        except Exception:
            pearson_corr = np.nan
        try:
            spearman_corr, _ = stats.spearmanr(y_test, y_pred)
        except Exception:
            spearman_corr = np.nan

    # 方向命中率
    direction_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_test)))

    # 预测分布
    positive_rate_true = float(np.mean(y_test > 0))
    positive_rate_pred = float(np.mean(y_pred > 0))

    return {
        "fold_id": fold_id,
        "mae": round(float(mae), 6),
        "mse": round(float(mse), 8),
        "rmse": round(float(rmse), 6),
        "r2": round(float(r2), 4),
        "pearson_corr": round(float(pearson_corr), 4) if not np.isnan(pearson_corr) else np.nan,
        "spearman_corr": round(float(spearman_corr), 4) if not np.isnan(spearman_corr) else np.nan,
        "direction_accuracy": round(float(direction_accuracy), 4),
        "positive_rate_true": round(float(positive_rate_true), 4),
        "positive_rate_pred": round(float(positive_rate_pred), 4),
        "pred_mean": round(float(np.mean(y_pred)), 6),
        "pred_std": round(float(np.std(y_pred)), 6),
        "true_mean": round(float(np.mean(y_test)), 6),
        "true_std": round(float(np.std(y_test)), 6),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_start": str(train_dates[0].date()) if len(train_dates) > 0 else "",
        "train_end": str(train_dates[-1].date()) if len(train_dates) > 0 else "",
        "test_start": str(test_dates[0].date()) if len(test_dates) > 0 else "",
        "test_end": str(test_dates[-1].date()) if len(test_dates) > 0 else "",
    }


# ============================================================
# 4. 基于预测收益率的简化回测
# ============================================================

def simple_return_signal_backtest(
    y_pred, y_true, threshold=0.0005, transaction_cost=0.0005,
):
    """
    根据预测收益率构造交易信号并回测。

    信号规则:
        pred_return > threshold → 持仓 (position=1)
        pred_return <= threshold → 空仓 (position=0)

    参数:
        y_pred: 预测收益率
        y_true: 实际收益率
        threshold: 信号阈值（用于覆盖交易成本）
        transaction_cost: 交易成本
    返回:
        dict，包含回测指标
    """
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    # 信号
    signal = (y_pred > threshold).astype(float)

    # 仓位变化和交易成本
    pos_change = np.abs(np.diff(np.concatenate([[0], signal])))
    cost = pos_change * transaction_cost

    # 策略收益
    strategy_return = signal * y_true - cost

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
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # 交易次数
    trades = int(np.sum(pos_change > 0))

    # 换手率
    turnover = trades / n_days if n_days > 0 else 0.0

    # 平均仓位
    avg_position = float(np.mean(signal))

    return {
        "total_return": round(float(total_return), 4),
        "annual_return": round(float(ann_return), 4),
        "annual_volatility": round(float(ann_vol), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "trades": trades,
        "turnover": round(float(turnover), 4),
        "avg_position": round(float(avg_position), 4),
    }


# ============================================================
# 5. 汇总跨 fold 结果
# ============================================================

def summarize_regression_results(results_list, group_col="model"):
    """
    按模型汇总跨 fold 的均值和标准差。
    """
    df = pd.DataFrame(results_list)

    numeric_cols = [
        "mae", "mse", "rmse", "r2",
        "pearson_corr", "spearman_corr",
        "direction_accuracy",
        "positive_rate_true", "positive_rate_pred",
        "pred_mean", "pred_std", "true_mean", "true_std",
    ]

    backtest_cols = [
        "total_return", "annual_return", "annual_volatility",
        "sharpe", "max_drawdown", "trades", "turnover", "avg_position",
    ]

    all_metrics = numeric_cols + [c for c in backtest_cols if c in df.columns]

    rows = []
    for model_name, group in df.groupby(group_col):
        row = {"model": model_name, "n_folds": len(group)}
        for col in all_metrics:
            if col in group.columns:
                vals = group[col].dropna()
                if len(vals) > 0:
                    row[f"{col}_mean"] = round(float(vals.mean()), 6)
                    row[f"{col}_std"] = round(float(vals.std()), 6)
                else:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. 绘图函数
# ============================================================

def plot_metric_by_fold(results_df, metric, ylabel, title, filepath):
    """
    绘制某指标按 fold 的变化图。
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    models = results_df["model"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

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


def plot_pred_vs_true(y_true_all, y_pred_all, model_name, filepath):
    """
    绘制预测值 vs 真实值散点图。
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true_all, y_pred_all, alpha=0.3, s=10, color="#1f77b4")

    # 对角线
    lim = max(abs(y_true_all).max(), abs(y_pred_all).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y=x")

    ax.set_xlabel("True Return", fontsize=11)
    ax.set_ylabel("Predicted Return", fontsize=11)
    ax.set_title(f"Predicted vs True Return — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. 完整收益率回归 walk-forward 流程
# ============================================================

def run_return_regression_walk_forward(
    feature_path,
    models_config=None,
    initial_train_size=500,
    test_size=126,
    threshold=0.0005,
    transaction_cost=0.0005,
    horizon=1,
    output_dir=None,
):
    """
    完整的收益率回归 walk-forward 验证流程。
    """
    import logging
    logger = logging.getLogger("stage8")

    if models_config is None:
        models_config = {
            "DummyRegressor_mean": DummyRegressor(strategy="mean"),
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=200, random_state=42, max_depth=5,
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                random_state=42,
            ),
        }

    # 加载数据
    logger.info("加载特征数据: %s", feature_path)
    df = load_feature_data(feature_path)
    logger.info("数据维度: %d 行 × %d 列", len(df), len(df.columns))

    # 准备数据集
    X, y_return, dates, feature_columns = prepare_return_regression_dataset(
        df, horizon=horizon,
    )
    logger.info("样本数: %d, 特征数: %d", len(X), len(feature_columns))
    logger.info("目标收益率: mean=%.6f, std=%.6f", y_return.mean(), y_return.std())
    logger.info("上涨比例: %.1f%%", 100 * (y_return > 0).mean())

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
    all_predictions = []

    for train_idx, test_idx, fold_id in splits:
        X_train = X.iloc[train_idx]
        y_train = y_return.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y_return.iloc[test_idx]
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]

        logger.info("Fold %d: train=%s~%s (%d), test=%s~%s (%d)",
                     fold_id,
                     train_dates[0].date(), train_dates[-1].date(), len(train_idx),
                     test_dates[0].date(), test_dates[-1].date(), len(test_idx))

        for model_name, model_template in models_config.items():
            from sklearn.base import clone
            model = clone(model_template)

            # 回归评估
            metrics = evaluate_regressor(
                model, X_train.values, y_train.values,
                X_test.values, y_test.values,
                fold_id, train_dates, test_dates,
            )
            metrics["model"] = model_name

            # 简化回测
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict(X_test.values)
            bt_metrics = simple_return_signal_backtest(
                y_pred, y_test.values,
                threshold=threshold, transaction_cost=transaction_cost,
            )
            metrics.update(bt_metrics)

            all_results.append(metrics)

            # 保存预测明细
            for i, date in enumerate(test_dates):
                all_predictions.append({
                    "date": date,
                    "fold_id": fold_id,
                    "model": model_name,
                    "y_true": round(float(y_test.iloc[i]), 6),
                    "y_pred": round(float(y_pred[i]), 6),
                })

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)

    # 汇总
    summary_df = summarize_regression_results(results_df)
    logger.info("收益率回归 walk-forward 验证完成")

    # 保存结果
    if output_dir is not None:
        output_dir = Path(output_dir)
        (output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(parents=True, exist_ok=True)

        # 保存明细
        results_df.to_csv(
            output_dir / "tables" / "SPY_return_regression_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("回归指标明细已保存")

        # 保存回测指标
        bt_cols = [
            "model", "fold_id", "total_return", "annual_return",
            "annual_volatility", "sharpe", "max_drawdown",
            "trades", "turnover", "avg_position",
        ]
        available_bt_cols = [c for c in bt_cols if c in results_df.columns]
        results_df[available_bt_cols].to_csv(
            output_dir / "tables" / "SPY_return_regression_backtest_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("回测指标明细已保存")

        # 保存预测明细
        predictions_df.to_csv(
            output_dir / "tables" / "SPY_return_regression_predictions.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("预测明细已保存")

        # 保存汇总
        summary_df.to_csv(
            output_dir / "tables" / "SPY_return_regression_summary.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("汇总指标已保存")

        # 绘图
        plot_metric_by_fold(
            results_df, "rmse", "RMSE",
            "Walk-Forward RMSE by Fold (Return Regression)",
            output_dir / "figures" / "SPY_return_regression_rmse_by_fold.png",
        )
        plot_metric_by_fold(
            results_df, "direction_accuracy", "Direction Accuracy",
            "Walk-Forward Direction Accuracy by Fold (Return Regression)",
            output_dir / "figures" / "SPY_return_regression_direction_accuracy_by_fold.png",
        )
        plot_metric_by_fold(
            results_df, "total_return", "Total Return",
            "Walk-Forward Total Return by Fold (Return Regression)",
            output_dir / "figures" / "SPY_return_regression_total_return_by_fold.png",
        )

        # 预测 vs 真实散点图（使用最佳模型或第一个非 dummy 模型）
        plot_model = "GradientBoostingRegressor"
        if plot_model not in predictions_df["model"].unique():
            plot_model = list(models_config.keys())[-1]
        pred_subset = predictions_df[predictions_df["model"] == plot_model]
        if len(pred_subset) > 0:
            plot_pred_vs_true(
                pred_subset["y_true"].values,
                pred_subset["y_pred"].values,
                plot_model,
                output_dir / "figures" / "SPY_return_regression_pred_vs_true.png",
            )
        logger.info("图表已保存")

    return summary_df, results_df
