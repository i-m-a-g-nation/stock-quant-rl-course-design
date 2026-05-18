"""
简易股票交易环境：不依赖 gymnasium，纯 Python 实现
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import DATA_FEATURES, setup_logging, timer

logger = setup_logging("trading_env")

FEATURE_COLS = [
    "return_1d", "return_5d", "ma_5", "ma_10", "ma_20", "ma_60",
    "volatility_20", "volume_ma_20", "close_ma20_ratio",
]

INPUT_FILE = "SPY_features_2015_2025.csv"


class StockTradingEnv:
    """
    简易股票交易环境。

    状态: 当前步的 9 个技术特征（无未来数据）
    动作: 0=空仓, 1=持有 SPY
    奖励: position * 下一期 market_return - transaction_cost

    时间对齐规则（无未来函数）:
      - 步 i 观察特征 features[i]，决策 action
      - action 在下一天（步 i+1）生效
      - reward = action * returns[i+1] / 100 - cost
    """

    def __init__(self, df, transaction_cost=0.001, window_size=1):
        self.df = df.reset_index(drop=True)
        self.features = self.df[FEATURE_COLS].values.astype(np.float32)
        self.returns = self.df["return_1d"].values.astype(np.float32)
        self.n = len(self.df)

        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.action_space_n = 2
        self.obs_dim = len(FEATURE_COLS)

    def reset(self):
        self.current_step = self.window_size - 1
        self.position = 0
        self.equity = 1.0
        self.prev_position = 0

        self.history = {
            "step": [], "action": [], "position": [],
            "reward": [], "equity": [], "market_return": [],
        }
        return self._get_observation()

    def _get_observation(self):
        return self.features[self.current_step].copy()

    def step(self, action):
        """
        action 决定下一期仓位，用下一期收益计算 reward（无未来函数）。
        返回: (observation, reward, done, info)
        """
        self.position = int(action)

        cost = 0.0
        if self.position != self.prev_position:
            cost = self.transaction_cost

        # 使用下一期收益（action 在下一天生效，避免未来函数）
        next_idx = self.current_step + 1
        market_ret = self.returns[next_idx] / 100.0
        reward = self.position * market_ret - cost

        self.equity *= (1.0 + reward)
        self.prev_position = self.position

        self.history["step"].append(self.current_step)
        self.history["action"].append(action)
        self.history["position"].append(self.position)
        self.history["reward"].append(reward)
        self.history["equity"].append(self.equity)
        self.history["market_return"].append(market_ret)

        self.current_step += 1
        done = self.current_step >= self.n - 1

        obs = self._get_observation() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        info = {"step": self.current_step - 1, "equity": self.equity}
        return obs, reward, done, info

    def get_history_df(self):
        return pd.DataFrame(self.history)


def run_random_baseline(env, seed=42):
    """运行随机策略基线，返回带 strategy 标签的 DataFrame"""
    np.random.seed(seed)
    env.reset()
    done = False
    while not done:
        action = np.random.randint(0, 2)
        _, _, done, _ = env.step(action)
    df = env.get_history_df()
    df.insert(0, "strategy", "Random")
    return df


def run_buy_and_hold(env):
    """运行买入持有基线，返回带 strategy 标签的 DataFrame"""
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(1)
    df = env.get_history_df()
    df.insert(0, "strategy", "BuyAndHold")
    return df


def compute_metrics(history: pd.DataFrame, label="") -> dict:
    """
    从历史记录计算评估指标。
    使用 history["equity"] 作为权益曲线（复利累积），
    确保 final_equity 与 history 最后一行一致。
    """
    equity = history["equity"].values
    rewards = history["reward"].values

    total_return = equity[-1] - 1.0
    ann_return = np.mean(rewards) * 252
    ann_vol = np.std(rewards) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cummax = np.maximum.accumulate(equity)
    drawdowns = (equity - cummax) / cummax
    max_dd = np.min(drawdowns)

    positions = history["position"].values
    trades = np.sum(np.abs(np.diff(positions)) > 0)

    return {
        "strategy": label,
        "total_return": round(total_return, 6),
        "annual_return": round(ann_return, 6),
        "annual_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "number_of_trades": int(trades),
        "final_equity": round(equity[-1], 6),
    }
