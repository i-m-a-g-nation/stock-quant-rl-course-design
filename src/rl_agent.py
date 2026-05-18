"""
Tabular Q-learning 强化学习策略：课程设计级别
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.utils import OUTPUT_TABLES, OUTPUT_FIGURES, setup_logging, timer
from src.trading_env import StockTradingEnv, compute_metrics, FEATURE_COLS

logger = setup_logging("rl_agent")

RANDOM_STATE = 42
EPISODES = 50
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 0.30
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


def discretize(obs):
    """
    将连续特征离散化为有限状态索引。
    基于 return_1d, close_ma20_ratio, volatility_20 三个特征，
    每个特征分 3 个桶 → 共 27 个离散状态。
    """
    ret_1d = obs[0]        # return_1d (百分比)
    close_ma20 = obs[8]     # close_ma20_ratio
    vol_20 = obs[6]        # volatility_20

    # return_1d: neg / near zero / pos
    if ret_1d < -0.5:
        r_idx = 0
    elif ret_1d <= 0.5:
        r_idx = 1
    else:
        r_idx = 2

    # close_ma20_ratio: below / near / above
    if close_ma20 < 0.98:
        c_idx = 0
    elif close_ma20 <= 1.02:
        c_idx = 1
    else:
        c_idx = 2

    # volatility_20: low / mid / high
    if vol_20 < 0.8:
        v_idx = 0
    elif vol_20 <= 1.5:
        v_idx = 1
    else:
        v_idx = 2

    return (r_idx, c_idx, v_idx)


@timer
def train_qlearning(env, episodes=EPISODES):
    """
    训练 tabular Q-learning agent。
    返回: Q-table, training_log
    """
    np.random.seed(RANDOM_STATE)
    Q = defaultdict(lambda: np.zeros(2))  # 2 个动作
    epsilon = EPSILON_START
    log = []

    for ep in range(episodes):
        obs = env.reset()
        state = discretize(obs)
        done = False
        ep_reward = 0.0

        while not done:
            # epsilon-greedy 选择动作
            if np.random.random() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = int(np.argmax(Q[state]))

            next_obs, reward, done, _ = env.step(action)
            next_state = discretize(next_obs) if not done else None

            # Q-learning 更新
            best_next = 0.0 if done else np.max(Q[next_state])
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

            state = next_state
            ep_reward += reward

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        log.append({"episode": ep + 1, "total_reward": round(ep_reward, 6),
                    "epsilon": round(epsilon, 4)})

        if (ep + 1) % 10 == 0:
            logger.info("  Episode %3d | total_reward=%.4f epsilon=%.4f",
                        ep + 1, ep_reward, epsilon)

    states_visited = len(Q)
    logger.info("Q-learning 训练完成: %d 个状态被访问", states_visited)
    return dict(Q), pd.DataFrame(log)


@timer
def evaluate_qlearning(env, Q):
    """用贪心策略评估 Q-learning agent"""
    np.random.seed(RANDOM_STATE)
    obs = env.reset()
    state = discretize(obs)
    done = False
    while not done:
        action = int(np.argmax(Q.get(state, np.zeros(2))))
        obs, _, done, _ = env.step(action)
        state = discretize(obs) if not done else None
    df = env.get_history_df()
    df.insert(0, "strategy", "QLearning")
    return df
