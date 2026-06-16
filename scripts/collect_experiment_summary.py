"""
=============================================================================
实验总摘要生成脚本
=============================================================================
读取各阶段输出的 CSV 文件，生成 EXPERIMENT_SUMMARY.md。

用法:  python scripts/collect_experiment_summary.py
=============================================================================
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils import OUTPUT_TABLES, OUTPUT_FIGURES, setup_logging

logger = setup_logging("summary")


def _read_csv_safe(filepath):
    """安全读取 CSV，不存在则返回 None"""
    if filepath.exists():
        return pd.read_csv(filepath, encoding="utf-8-sig")
    return None


def collect_summary():
    """收集实验结果并生成摘要"""
    lines = []
    lines.append("# 实验总摘要 — Enhanced Experiments")
    lines.append("")
    lines.append("> 自动生成于实验完成后。本项目仍为课程实验，不构成投资建议。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ============================================================
    # 1. 实验增强内容
    # ============================================================
    lines.append("## 1. 本次实验增强内容")
    lines.append("")
    lines.append("1. **监督学习增强**：新增 DummyClassifier 基线（MostFrequent / Stratified），")
    lines.append("   增加 balanced_accuracy、roc_auc、混淆矩阵等指标，验证模型是否优于随机基线。")
    lines.append("2. **特征工程增强**：新增 8 个相对化特征（open_close_ratio 等），")
    lines.append("   定义 FEATURE_COLS_ORIGINAL / RELATIVE / ALL 三组特征集。")
    lines.append("3. **回测增强**：同时回测 BuyAndHold、AlwaysLong、LogisticRegression、")
    lines.append("   RandomForestClassifier、DummyMostFrequent、DummyStratified 六个策略。")
    lines.append("4. **Q-learning 增强**：将全样本训练/评估改为训练集(80%)训练、测试集(20%)评估，")
    lines.append("   测试集使用 epsilon=0 贪心策略，同时对比 RandomPolicy 和 BuyAndHold。")
    lines.append("5. **Permutation Importance**：对 LR 和 RF 在测试集上计算 permutation importance。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ============================================================
    # 2. 监督学习模型指标摘要
    # ============================================================
    lines.append("## 2. 监督学习模型指标摘要")
    lines.append("")

    ml_metrics = _read_csv_safe(OUTPUT_TABLES / "SPY_ml_model_metrics_enhanced.csv")
    if ml_metrics is not None:
        lines.append("| 模型 | accuracy | balanced_accuracy | f1 | roc_auc | positive_rate_pred |")
        lines.append("|------|----------|-------------------|-----|---------|-------------------|")

        models = ml_metrics["模型"].unique()
        for model in models:
            m = ml_metrics[ml_metrics["模型"] == model]
            vals = {}
            for _, row in m.iterrows():
                vals[row["指标"]] = row["值"]
            lines.append(f"| {model} | {vals.get('accuracy', 'N/A')} | "
                        f"{vals.get('balanced_accuracy', 'N/A')} | "
                        f"{vals.get('f1', 'N/A')} | "
                        f"{vals.get('roc_auc', 'N/A')} | "
                        f"{vals.get('positive_rate_pred', 'N/A')} |")
        lines.append("")

        # 混淆矩阵
        cm = _read_csv_safe(OUTPUT_TABLES / "SPY_ml_confusion_matrices.csv")
        if cm is not None:
            lines.append("### 混淆矩阵")
            lines.append("")
            lines.append("| 模型 | TN | FP | FN | TP |")
            lines.append("|------|-----|-----|-----|-----|")
            for _, row in cm.iterrows():
                lines.append(f"| {row['模型']} | {row['TN']} | {row['FP']} | {row['FN']} | {row['TP']} |")
            lines.append("")
    else:
        lines.append("*（未找到增强指标文件，请先运行 python run_stage3.py）*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ============================================================
    # 3. 多策略回测结果摘要
    # ============================================================
    lines.append("## 3. 多策略回测结果摘要")
    lines.append("")

    bt_metrics = _read_csv_safe(OUTPUT_TABLES / "SPY_strategy_backtest_enhanced_metrics.csv")
    if bt_metrics is not None:
        lines.append("| 策略 | cumulative_return | annual_return | sharpe | max_drawdown | calmar | win_rate | trade_count |")
        lines.append("|------|-------------------|---------------|--------|--------------|--------|----------|-------------|")
        for _, row in bt_metrics.iterrows():
            lines.append(f"| {row['strategy']} | {row['cumulative_return']} | "
                        f"{row['annual_return']} | {row['sharpe']} | "
                        f"{row['max_drawdown']} | {row['calmar']} | "
                        f"{row['win_rate']} | {row['trade_count']} |")
        lines.append("")
    else:
        lines.append("*（未找到增强回测文件，请先运行 python run_stage4.py）*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ============================================================
    # 4. Q-learning 样本外测试摘要
    # ============================================================
    lines.append("## 4. Q-learning 样本外测试摘要")
    lines.append("")

    rl_summary = _read_csv_safe(OUTPUT_TABLES / "SPY_rl_train_test_summary.csv")
    if rl_summary is not None:
        lines.append("### 训练/测试集信息")
        lines.append("")
        for _, row in rl_summary.iterrows():
            lines.append(f"- **{row['指标']}**: {row['值']}")
        lines.append("")

    rl_metrics = _read_csv_safe(OUTPUT_TABLES / "SPY_rl_test_strategy_metrics.csv")
    if rl_metrics is not None:
        lines.append("### 测试集策略对比")
        lines.append("")
        lines.append("| 策略 | total_return | sharpe_ratio | max_drawdown | number_of_trades |")
        lines.append("|------|--------------|--------------|--------------|------------------|")
        for _, row in rl_metrics.iterrows():
            lines.append(f"| {row['strategy']} | {row['total_return']} | "
                        f"{row['sharpe_ratio']} | {row['max_drawdown']} | "
                        f"{row['number_of_trades']} |")
        lines.append("")
    else:
        lines.append("*（未找到 Q-learning 增强文件，请先运行 python run_stage6.py）*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ============================================================
    # 5. 主要发现
    # ============================================================
    lines.append("## 5. 主要发现")
    lines.append("")

    # 自动分析
    if ml_metrics is not None:
        lines.append("### 5.1 逻辑回归 vs 多数类基线")
        lines.append("")
        lr_acc = ml_metrics[(ml_metrics["模型"] == "LogisticRegression") &
                           (ml_metrics["指标"] == "accuracy")]["值"].values
        dummy_acc = ml_metrics[(ml_metrics["模型"] == "DummyMostFrequent") &
                              (ml_metrics["指标"] == "accuracy")]["值"].values
        if len(lr_acc) > 0 and len(dummy_acc) > 0:
            try:
                lr_val = float(lr_acc[0])
                dummy_val = float(dummy_acc[0])
                if lr_val > dummy_val:
                    lines.append(f"- 逻辑回归 accuracy ({lr_val:.4f}) **优于**多数类基线 ({dummy_val:.4f})。")
                else:
                    lines.append(f"- 逻辑回归 accuracy ({lr_val:.4f}) **未优于**多数类基线 ({dummy_val:.4f})，"
                                "需关注模型是否仅学到类别偏置。")
            except (ValueError, TypeError):
                lines.append("- 指标值解析异常，无法比较。")
        lines.append("")

        lines.append("### 5.2 模型是否偏向预测上涨")
        lines.append("")
        for model in ["LogisticRegression", "RandomForestClassifier"]:
            pred_rate = ml_metrics[(ml_metrics["模型"] == model) &
                                  (ml_metrics["指标"] == "positive_rate_pred")]["值"].values
            true_rate = ml_metrics[(ml_metrics["模型"] == model) &
                                  (ml_metrics["指标"] == "positive_rate_true")]["值"].values
            if len(pred_rate) > 0 and len(true_rate) > 0:
                try:
                    p = float(pred_rate[0])
                    t = float(true_rate[0])
                    diff = p - t
                    if abs(diff) > 0.1:
                        lines.append(f"- {model}: 预测上涨比例 ({p:.2%}) vs 真实上涨比例 ({t:.2%})，"
                                    f"偏差 {diff:+.2%}，**存在偏向**。")
                    else:
                        lines.append(f"- {model}: 预测上涨比例 ({p:.2%}) vs 真实上涨比例 ({t:.2%})，"
                                    f"偏差较小 ({diff:+.2%})。")
                except (ValueError, TypeError):
                    pass
        lines.append("")

    if rl_metrics is not None:
        lines.append("### 5.3 Q-learning 是否跑赢 BuyAndHold")
        lines.append("")
        ql_ret = rl_metrics[rl_metrics["strategy"] == "QLearning"]["total_return"].values
        if len(ql_ret) > 0:
            try:
                ql_val = float(ql_ret[0])
                # 尝试找 BuyAndHold
                bh_rows = rl_metrics[rl_metrics["strategy"] == "BuyAndHold"]
                if len(bh_rows) > 0:
                    bh_val = float(bh_rows["total_return"].values[0])
                    if ql_val > bh_val:
                        lines.append(f"- Q-learning ({ql_val:.4f}) **跑赢** BuyAndHold ({bh_val:.4f})。")
                    else:
                        lines.append(f"- Q-learning ({ql_val:.4f}) **未跑赢** BuyAndHold ({bh_val:.4f})。")
                else:
                    lines.append(f"- Q-learning total_return = {ql_val:.4f}。")
            except (ValueError, TypeError):
                pass
        lines.append("")

        lines.append("### 5.4 Q-learning 是否降低最大回撤")
        lines.append("")
        ql_dd = rl_metrics[rl_metrics["strategy"] == "QLearning"]["max_drawdown"].values
        if len(ql_dd) > 0:
            try:
                ql_dd_val = float(ql_dd[0])
                bh_dd_rows = rl_metrics[rl_metrics["strategy"] == "BuyAndHold"]
                if len(bh_dd_rows) > 0:
                    bh_dd_val = float(bh_dd_rows["max_drawdown"].values[0])
                    if abs(ql_dd_val) < abs(bh_dd_val):
                        lines.append(f"- Q-learning 最大回撤 ({ql_dd_val:.4f}) **小于** BuyAndHold ({bh_dd_val:.4f})，"
                                    "风险有所降低。")
                    else:
                        lines.append(f"- Q-learning 最大回撤 ({ql_dd_val:.4f}) **不小于** BuyAndHold ({bh_dd_val:.4f})。")
                else:
                    lines.append(f"- Q-learning max_drawdown = {ql_dd_val:.4f}。")
            except (ValueError, TypeError):
                pass
        lines.append("")

    lines.append("### 5.5 LSTM 过拟合分析")
    lines.append("")
    lines.append("请参考 `outputs/figures/SPY_lstm_training_curve.png` 中的训练/测试损失曲线。")
    lines.append("若训练损失持续下降而测试损失上升，则存在过拟合迹象。")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ============================================================
    # 6. 后续报告需要更新的位置
    # ============================================================
    lines.append("## 6. 后续报告需要更新的位置")
    lines.append("")
    lines.append("以下章节需要根据增强实验结果更新：")
    lines.append("")
    lines.append("1. **摘要**：更新主要实验结论和指标数据。")
    lines.append("2. **4.2 机器学习预测结果**：")
    lines.append("   - 添加 DummyClassifier 基线对比。")
    lines.append("   - 添加 balanced_accuracy、roc_auc、混淆矩阵分析。")
    lines.append("   - 讨论 positive_rate_pred vs positive_rate_true。")
    lines.append("   - 引用 permutation importance 结果。")
    lines.append("3. **4.4 回测与风险评估结果**：")
    lines.append("   - 替换为六策略对比表。")
    lines.append("   - 添加权益曲线和回撤曲线图。")
    lines.append("4. **4.5 强化学习实验结果**：")
    lines.append("   - 添加训练集/测试集分离说明。")
    lines.append("   - 添加测试集上 Q-learning vs BuyAndHold vs Random 对比。")
    lines.append("5. **4.6 综合分析**：综合各部分结果进行分析。")
    lines.append("6. **4.7 结论与展望**：更新结论。")
    lines.append("7. **附件代码片段**：更新为最新代码。")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ============================================================
    # 7. 生成文件清单
    # ============================================================
    lines.append("## 7. 新增/更新文件清单")
    lines.append("")
    lines.append("### 新增输出文件")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| `outputs/tables/SPY_ml_model_metrics_enhanced.csv` | 增强 ML 指标表 |")
    lines.append("| `outputs/tables/SPY_ml_test_predictions_enhanced.csv` | 增强测试预测 |")
    lines.append("| `outputs/tables/SPY_ml_confusion_matrices.csv` | 混淆矩阵 |")
    lines.append("| `outputs/tables/SPY_feature_columns_enhanced.csv` | 特征字段说明 |")
    lines.append("| `outputs/tables/SPY_strategy_backtest_enhanced_metrics.csv` | 多策略回测指标 |")
    lines.append("| `outputs/tables/SPY_strategy_equity_curves_enhanced.csv` | 多策略权益曲线 |")
    lines.append("| `outputs/tables/SPY_permutation_importance.csv` | Permutation importance |")
    lines.append("| `outputs/tables/SPY_rl_train_test_summary.csv` | RL 训练/测试摘要 |")
    lines.append("| `outputs/tables/SPY_rl_test_strategy_metrics.csv` | RL 测试集策略指标 |")
    lines.append("| `outputs/tables/SPY_rl_test_equity_curves.csv` | RL 测试集权益曲线 |")
    lines.append("| `outputs/figures/SPY_ml_metrics_enhanced.png` | ML 指标对比图 |")
    lines.append("| `outputs/figures/SPY_permutation_importance_top15.png` | Perm Imp 图 |")
    lines.append("| `outputs/figures/SPY_strategy_equity_curves_enhanced.png` | 多策略权益图 |")
    lines.append("| `outputs/figures/SPY_strategy_drawdowns_enhanced.png` | 多策略回撤图 |")
    lines.append("| `outputs/figures/SPY_rl_test_equity_curves.png` | RL 测试集权益图 |")
    lines.append("| `outputs/figures/SPY_rl_test_drawdowns.png` | RL 测试集回撤图 |")
    lines.append("")

    lines.append("### 更新的源代码文件")
    lines.append("")
    lines.append("| 文件 | 变更说明 |")
    lines.append("|------|----------|")
    lines.append("| `src/ml_models.py` | 新增 DummyClassifier、增强指标、permutation importance |")
    lines.append("| `src/features.py` | 新增相对化特征、特征列集合定义 |")
    lines.append("| `src/backtest.py` | 新增多策略回测函数 |")
    lines.append("| `src/rl_agent.py` | 新增训练集/测试集分离评估 |")
    lines.append("| `run_stage2.py` | 增加特征字段说明保存 |")
    lines.append("| `run_stage3.py` | 增加增强指标输出 |")
    lines.append("| `run_stage4.py` | 增加多策略回测 |")
    lines.append("| `run_stage6.py` | 增加训练集/测试集分离 Q-learning |")
    lines.append("| `README.md` | 增加 Enhanced Experiments 说明 |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**免责声明**：本项目仍为课程实验，不构成投资建议。")
    lines.append("")

    # 写入文件
    summary_path = OUTPUT_TABLES.parent / "EXPERIMENT_SUMMARY.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("实验总摘要已保存: %s", summary_path)
    return summary_path


if __name__ == "__main__":
    collect_summary()
