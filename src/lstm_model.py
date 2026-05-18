"""
LSTM 方向预测模块：使用 PyTorch LSTM 预测下一交易日涨跌方向
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from src.utils import DATA_FEATURES, OUTPUT_TABLES, OUTPUT_MODELS, OUTPUT_FIGURES, setup_logging, timer

logger = setup_logging("lstm_model")

# 输入
INPUT_FILE = "SPY_features_2015_2025.csv"

# 技术特征列（9 个）
FEATURE_COLS = [
    "return_1d", "return_5d", "ma_5", "ma_10", "ma_20", "ma_60",
    "volatility_20", "volume_ma_20", "close_ma20_ratio",
]

# 输出
METRICS_FILE = "SPY_lstm_metrics.csv"
PREDICTIONS_FILE = "SPY_lstm_test_predictions.csv"
MODEL_FILE = "SPY_lstm_model.pt"
PLOT_FILE = "SPY_lstm_training_curve.png"

# 超参数
LOOKBACK = 20
HIDDEN_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_STATE):
    """固定随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("随机种子已固定: seed=%d, device=%s", seed, DEVICE)


class LSTMModel(nn.Module):
    """单层 LSTM + Dense(sigmoid) 二分类"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # 取最后时间步
        out = self.fc(out)
        return torch.sigmoid(out).squeeze(-1)


def _create_sequences(features, labels, lookback):
    """从特征和标签构建滑动窗口序列"""
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i : i + lookback])
        y.append(labels[i + lookback])
    return np.array(X), np.array(y)


@timer
def load_and_prepare():
    """加载特征数据，构造标签，标准化，构建序列"""
    filepath = DATA_FEATURES / INPUT_FILE
    if not filepath.exists():
        raise FileNotFoundError(f"特征文件不存在: {filepath}\n请先运行 python run_stage2.py")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info("加载特征数据: %d 行", len(df))

    # 构造标签：先用 shift 得到下一期收益，dropna 后再构造方向标签，
    # 避免最后一行 target_return_1d=NaN 被误标为下跌 (NaN>0=False→0)
    df["target_return_1d"] = df["return_1d"].shift(-1)
    df = df.dropna(subset=["target_return_1d"])
    df["target_direction"] = (df["target_return_1d"] > 0).astype(int)
    logger.info("标签构造完成: %d 行, 上涨比例=%.1f%%",
                len(df), 100 * df["target_direction"].mean())

    # 特征与标签
    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = df["target_direction"].values.astype(np.float32)

    # 时间序列划分（不打乱）
    split_idx = int(len(X_all) * (1 - TEST_SIZE))
    X_train_raw, X_test_raw = X_all[:split_idx], X_all[split_idx:]
    y_train_raw, y_test_raw = y_all[:split_idx], y_all[split_idx:]

    # 标准化（只在训练集 fit）
    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(X_train_raw)
    X_test_raw = scaler.transform(X_test_raw)

    # 构建滑动窗口序列
    X_train, y_train = _create_sequences(X_train_raw, y_train_raw, LOOKBACK)
    X_test, y_test = _create_sequences(X_test_raw, y_test_raw, LOOKBACK)

    logger.info("序列构造: 训练 %d, 测试 %d", len(X_train), len(X_test))
    logger.info("训练集日期: %s → %s",
                df.index[:split_idx].min().strftime("%Y-%m-%d"),
                df.index[:split_idx].max().strftime("%Y-%m-%d"))
    logger.info("测试集日期: %s → %s",
                df.index[split_idx:].min().strftime("%Y-%m-%d"),
                df.index[split_idx:].max().strftime("%Y-%m-%d"))

    return (X_train, y_train), (X_test, y_test), df.index[split_idx:]


@timer
def train_model(X_train, y_train, X_test, y_test):
    """训练 LSTM 模型，返回模型和训练历史"""
    model = LSTMModel(len(FEATURE_COLS), HIDDEN_SIZE).to(DEVICE)
    logger.info("模型: LSTM(input=%d, hidden=%d) on %s", len(FEATURE_COLS), HIDDEN_SIZE, DEVICE)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += ((pred > 0.5).float() == yb).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # 测试集评估
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            y_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)
            pred_t = model(x_t)
            test_loss = criterion(pred_t, y_t).item()
            test_acc = ((pred_t > 0.5).float() == y_t).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch %2d | train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f",
                        epoch + 1, train_loss, train_acc, test_loss, test_acc)

    logger.info("训练完成")
    return model, history


@timer
def evaluate_model(model, X_test, y_test):
    """评估模型并返回指标和预测结果"""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_prob = model(x_t).cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    n = len(y_test)

    rows = [
        {"模型": "LSTM", "指标": "accuracy", "值": round(acc, 4)},
        {"模型": "LSTM", "指标": "precision", "值": round(prec, 4)},
        {"模型": "LSTM", "指标": "recall", "值": round(rec, 4)},
        {"模型": "LSTM", "指标": "f1", "值": round(f1, 4)},
        {"模型": "LSTM", "指标": "roc_auc", "值": round(auc, 4)},
        {"模型": "LSTM", "指标": "test_samples", "值": n},
    ]

    metrics = pd.DataFrame(rows)
    pred_df = pd.DataFrame({
        "y_true": y_test.astype(int),
        "LSTM_pred": y_pred.flatten(),
        "LSTM_prob": y_prob.round(4),
    })

    logger.info("accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
                acc, prec, rec, f1, auc)
    return metrics, pred_df


@timer
def save_metrics(metrics):
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_TABLES / METRICS_FILE
    metrics.to_csv(fp, index=False, encoding="utf-8-sig")
    logger.info("指标: %s", fp)
    return fp


@timer
def save_predictions(pred_df, test_dates):
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_TABLES / PREDICTIONS_FILE
    pred_df.index = test_dates[-len(pred_df):]
    pred_df.to_csv(fp, encoding="utf-8-sig")
    logger.info("预测: %s", fp)
    return fp


@timer
def save_model(model):
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_MODELS / MODEL_FILE
    torch.save(model.state_dict(), fp)
    logger.info("模型: %s", fp)
    return fp


@timer
def plot_training_curve(history):
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_FIGURES / PLOT_FILE

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", color="#1f77b4")
    ax1.plot(history["test_loss"], label="Test Loss", color="#d62728")
    ax1.set_title("LSTM Training — Loss Curve", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Accuracy", color="#1f77b4")
    ax2.plot(history["test_acc"], label="Test Accuracy", color="#d62728")
    ax2.set_title("LSTM Training — Accuracy Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("训练曲线: %s", fp)
    return fp


@timer
def run_pipeline():
    set_seed(RANDOM_STATE)
    logger.info("[Step 1/4] 数据加载与序列构造")
    (X_train, y_train), (X_test, y_test), test_dates = load_and_prepare()

    logger.info("[Step 2/4] 训练 LSTM 模型")
    model, history = train_model(X_train, y_train, X_test, y_test)

    logger.info("[Step 3/4] 评估模型")
    metrics, pred_df = evaluate_model(model, X_test, y_test)

    logger.info("[Step 4/4] 保存结果")
    save_metrics(metrics)
    save_predictions(pred_df, test_dates)
    save_model(model)
    plot_training_curve(history)

    logger.info("LSTM 模型训练全部完成！")
    return metrics
