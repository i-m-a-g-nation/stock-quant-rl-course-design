#!/usr/bin/env python3
"""
检查 public 版与 full 版的一致性。
public 必需文件缺失时退出码为 1。
local full 可选文件缺失只 warning，不失败。
"""

import os
import sys
from pathlib import Path

# 定义 public 必需文件列表
PUBLIC_REQUIRED_FILES = [
    "README.md",
    "FINAL_PROJECT_CHECK.md",
    "VERSION_GUIDE.md",
    "LOCAL_FULL_MANIFEST.example.md",
    ".gitignore",
    "docs/大数据课设报告_最终版.md",
    "outputs/EXPERIMENT_SUMMARY.md",
    "outputs/METRIC_DEFINITION_NOTES.md",
    "outputs/CODE_FINAL_CHECK.md",
    "scripts/check_public_full_consistency.py",
]

# 定义 local full 可选文件列表（示例）
LOCAL_FULL_OPTIONAL_FILES = [
    "data/raw/SPY_raw_2015_2025.csv",
    "data/processed/SPY_clean_2015_2025.csv",
    "data/features/SPY_features_2015_2025.csv",
    "outputs/models/SPY_logistic_regression.joblib",
    "outputs/models/SPY_random_forest_classifier.joblib",
    "outputs/models/SPY_lstm_model.pt",
    "outputs/tables/SPY_data_quality_report.csv",
    "outputs/tables/SPY_feature_summary.csv",
    "outputs/tables/SPY_ml_baseline_metrics.csv",
    "outputs/tables/SPY_ml_test_predictions.csv",
    "outputs/tables/SPY_ml_feature_importance.csv",
    "outputs/tables/SPY_market_returns.csv",
    "outputs/tables/SPY_risk_metrics.csv",
    "outputs/tables/SPY_prediction_direction_analysis.csv",
    "outputs/tables/SPY_strategy_backtest.csv",
    "outputs/tables/SPY_strategy_metrics.csv",
    "outputs/tables/SPY_lstm_metrics.csv",
    "outputs/tables/SPY_lstm_test_predictions.csv",
    "outputs/tables/SPY_rl_env_baseline_metrics.csv",
    "outputs/tables/SPY_rl_env_baseline_history.csv",
    "outputs/tables/SPY_rl_qlearning_metrics.csv",
    "outputs/tables/SPY_rl_qlearning_history.csv",
    "outputs/tables/SPY_rl_qlearning_training_log.csv",
    "outputs/figures/SPY_close_price.png",
    "outputs/figures/SPY_volume.png",
    "outputs/figures/SPY_return_distribution.png",
    "outputs/figures/SPY_strategy_equity_curve.png",
    "outputs/figures/SPY_lstm_training_curve.png",
    "outputs/figures/SPY_rl_env_baseline_equity.png",
    "outputs/figures/SPY_rl_qlearning_equity.png",
]

def check_public_required_files():
    """检查 public 必需文件是否存在"""
    missing_files = []
    for file_path in PUBLIC_REQUIRED_FILES:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("[FAIL] 缺少 public 必需文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("[OK] 所有 public 必需文件都存在")
        return True

def check_local_full_optional_files():
    """检查 local full 可选文件是否存在（仅警告）"""
    missing_files = []
    for file_path in LOCAL_FULL_OPTIONAL_FILES:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("[WARN] 以下 local full 可选文件缺失（不影响 public 版）:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("[OK] 所有 local full 可选文件都存在")
        return True

def main():
    print("=" * 60)
    print("检查 public 版与 full 版的一致性")
    print("=" * 60)
    
    # 检查 public 必需文件
    public_ok = check_public_required_files()
    
    print("-" * 60)
    
    # 检查 local full 可选文件
    local_ok = check_local_full_optional_files()
    
    print("=" * 60)
    
    if not public_ok:
        print("[FAIL] 检查失败：缺少 public 必需文件")
        sys.exit(1)
    else:
        print("[OK] 检查通过：public 版文件完整")
        sys.exit(0)

if __name__ == "__main__":
    main()