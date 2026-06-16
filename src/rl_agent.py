"""
Tabular Q-learning 强化学习策略：课程设计级别
支持训练集/测试集分离评估
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

# 增强实验输出
TRAIN_TEST_SUMMARY_FILE = "SPY_rl_train_test_summary.csv"
TEST_STRATEGY_METRICS_FILE = "SPY_rl_test_strategy_metrics.csv"
TEST_EQUITY_CURVES_FILE = "SPY_rl_test_equity_curves.csv"
TEST_EQUITY_PLOT = "SPY_rl_test_equity_curves.png"
TEST_DRAWDOWN_PLOT = "SPY_rl_test_drawdowns.png"


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


# ============================================================
# 增强 Q-learning：训练集/测试集分离评估
# ============================================================

def _split_data_by_time(df, train_ratio=0.8):
    """按时间顺序切分数据"""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


@timer
def train_qlearning_on_train_set(train_df, episodes=EPISODES, transaction_cost=0.001):
    """
    在训练集上训练 Q-learning agent。
    返回: Q-table, training_log, train_summary
    """
    env = StockTradingEnv(train_df, transaction_cost=transaction_cost)
    np.random.seed(RANDOM_STATE)
    Q = defaultdict(lambda: np.zeros(2))
    epsilon = EPSILON_START
    log = []

    for ep in range(episodes):
        obs = env.reset()
        state = discretize(obs)
        done = False
        ep_reward = 0.0

        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = int(np.argmax(Q[state]))

            next_obs, reward, done, _ = env.step(action)
            next_state = discretize(next_obs) if not done else None

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

    train_summary = {
        "train_start": train_df.index.min().strftime("%Y-%m-%d"),
        "train_end": train_df.index.max().strftime("%Y-%m-%d"),
        "train_samples": len(train_df),
        "q_table_states": states_visited,
        "action_space": 2,
        "episodes": episodes,
        "final_epsilon": round(epsilon, 4),
    }

    logger.info("Q-learning 训练完成: %d 个状态被访问", states_visited)
    return dict(Q), pd.DataFrame(log), train_summary


@timer
def evaluate_on_test_set(test_df, Q, transaction_cost=0.001):
    """
    在测试集上评估 Q-learning（epsilon=0，贪心策略）以及基线策略。
    返回: test_summary, strategy_metrics, equity_curves
    """
    test_summary = {
        "test_start": test_df.index.min().strftime("%Y-%m-%d"),
        "test_end": test_df.index.max().strftime("%Y-%m-%d"),
        "test_samples": len(test_df),
    }

    strategy_metrics = []
    equity_curves = {}

    # 1. RandomPolicy
    env_random = StockTradingEnv(test_df, transaction_cost=transaction_cost)
    np.random.seed(RANDOM_STATE)
    obs = env_random.reset()
    done = False
    while not done:
        action = np.random.randint(0, 2)
        _, _, done, _ = env_random.step(action)
    hist_random = env_random.get_history_df()
    m_random = compute_metrics(hist_random, "RandomPolicy")
    strategy_metrics.append(m_random)
    equity_curves["RandomPolicy"] = hist_random["equity"].values

    # 2. BuyAndHold
    env_bh = StockTradingEnv(test_df, transaction_cost=transaction_cost)
    obs = env_bh.reset()
    done = False
    while not done:
        _, _, done, _ = env_bh.step(1)
    hist_bh = env_bh.get_history_df()
    m_bh = compute_metrics(hist_bh, "BuyAndHold")
    strategy_metrics.append(m_bh)
    equity_curves["BuyAndHold"] = hist_bh["equity"].values

    # 3. QLearning（epsilon=0，贪心）
    env_ql = StockTradingEnv(test_df, transaction_cost=transaction_cost)
    obs = env_ql.reset()
    state = discretize(obs)
    done = False
    while not done:
        action = int(np.argmax(Q.get(state, np.zeros(2))))
        obs, _, done, _ = env_ql.step(action)
        state = discretize(obs) if not done else None
    hist_ql = env_ql.get_history_df()
    m_ql = compute_metrics(hist_ql, "QLearning")
    strategy_metrics.append(m_ql)
    equity_curves["QLearning"] = hist_ql["equity"].values

    return test_summary, strategy_metrics, equity_curves


@timer
def save_train_test_summary(train_summary, test_summary, filename=TRAIN_TEST_SUMMARY_FILE):
    """保存训练集/测试集摘要"""
    rows = []
    for k, v in train_summary.items():
        rows.append({"指标": f"train_{k}", "值": v})
    for k, v in test_summary.items():
        rows.append({"指标": f"test_{k}", "值": v})

    df = pd.DataFrame(rows)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_TABLES / filename
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    logger.info("训练/测试集摘要已保存: %s", fp)
    return fp


@timer
def save_test_strategy_metrics(strategy_metrics, filename=TEST_STRATEGY_METRICS_FILE):
    """保存测试集策略指标"""
    df = pd.DataFrame(strategy_metrics)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_TABLES / filename
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    logger.info("测试集策略指标已保存: %s", fp)
    return fp


@timer
def save_test_equity_curves(equity_curves, filename=TEST_EQUITY_CURVES_FILE):
    """保存测试集权益曲线"""
    max_len = max(len(v) for v in equity_curves.values())
    padded = {}
    for label, curve in equity_curves.items():
        if len(curve) < max_len:
            padded[label] = np.pad(curve, (0, max_len - len(curve)), constant_values=np.nan)
        else:
            padded[label] = curve

    df = pd.DataFrame(padded)
    df.index.name = "step"
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_TABLES / filename
    df.to_csv(fp, encoding="utf-8-sig")
    logger.info("测试集权益曲线已保存: %s", fp)
    return fp


@timer
def plot_test_equity_curves(equity_curves, filename=TEST_EQUITY_PLOT):
    """绘制测试集权益曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_FIGURES / filename

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for i, (label, curve) in enumerate(equity_curves.items()):
        ax.plot(curve, linewidth=1.5 if label == "QLearning" else 1.0,
                color=colors[i % len(colors)], label=label,
                alpha=0.9 if label == "QLearning" else 0.6)

    ax.set_title("SPY RL Test Set — Strategy Equity Curves",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Equity", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("测试集权益曲线图已保存: %s", fp)
    return fp


@timer
def plot_test_drawdowns(equity_curves, filename=TEST_DRAWDOWN_PLOT):
    """绘制测试集回撤曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_FIGURES / filename

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for i, (label, curve) in enumerate(equity_curves.items()):
        eq = pd.Series(curve)
        cummax = eq.cummax()
        dd = (eq - cummax) / cummax
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.3,
                        color=colors[i % len(colors)], label=label)

    ax.set_title("SPY RL Test Set — Strategy Drawdowns",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("测试集回撤曲线图已保存: %s", fp)
    return fp


@timer
def run_enhanced_qlearning(df, train_ratio=0.8, episodes=EPISODES, transaction_cost=0.001):
    """
    运行增强 Q-learning 实验：训练集训练，测试集评估。
    """
    logger.info("=" * 50)
    logger.info("增强 Q-learning: 训练集/测试集分离")
    logger.info("=" * 50)

    # 切分数据
    train_df, test_df = _split_data_by_time(df, train_ratio)
    logger.info("训练集: %s → %s (%d 行)",
                train_df.index.min().strftime("%Y-%m-%d"),
                train_df.index.max().strftime("%Y-%m-%d"),
                len(train_df))
    logger.info("测试集: %s → %s (%d 行)",
                test_df.index.min().strftime("%Y-%m-%d"),
                test_df.index.max().strftime("%Y-%m-%d"),
                len(test_df))

    # 训练
    logger.info("\n>>> [训练阶段] Q-learning 在训练集上训练 (%d episodes)", episodes)
    Q, train_log, train_summary = train_qlearning_on_train_set(
        train_df, episodes=episodes, transaction_cost=transaction_cost
    )
    train_log.to_csv(OUTPUT_TABLES / "SPY_rl_qlearning_train_log.csv",
                     index=False, encoding="utf-8-sig")

    # 测试集评估
    logger.info("\n>>> [测试阶段] 在测试集上评估 (epsilon=0)")
    test_summary, strategy_metrics, equity_curves = evaluate_on_test_set(
        test_df, Q, transaction_cost=transaction_cost
    )

    # 保存结果
    save_train_test_summary(train_summary, test_summary)
    save_test_strategy_metrics(strategy_metrics)
    save_test_equity_curves(equity_curves)
    plot_test_equity_curves(equity_curves)
    plot_test_drawdowns(equity_curves)

    # 打印摘要
    logger.info("\n--- 测试集策略对比 ---")
    for m in strategy_metrics:
        logger.info("  [%s] cum_ret=%.4f sharpe=%.4f max_dd=%.4f trades=%d",
                    m["strategy"], m["total_return"], m["sharpe_ratio"],
                    m["max_drawdown"], m["number_of_trades"])

    logger.info("\n增强 Q-learning 实验完成！")
    return train_summary, test_summary, strategy_metrics
