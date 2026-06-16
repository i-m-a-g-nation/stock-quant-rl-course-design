# 术语表

| 术语 | 英文 | 通俗解释 | 本项目中的含义 | 答辩一句话 |
|---|---|---|---|---|
| SPY | SPDR S&P 500 ETF Trust | 一只跟踪 S&P 500 的 ETF | 唯一研究标的 | SPY 是有代表性的可交易 ETF，适合课程实验。 |
| 股票 | Stock | 一家公司的部分所有权凭证 | 用于解释金融背景 | 股票代表上市公司的部分权益。 |
| ETF | Exchange-Traded Fund | 可以在交易所交易的一篮子资产基金 | SPY 的资产类别 | ETF 兼具基金分散性与交易便利性。 |
| 指数 | Index | 描述一篮子证券整体表现的统计指标 | S&P 500 是指数 | 指数是统计指标，不等同于单只股票。 |
| S&P 500 | Standard & Poor's 500 Index | 美国大型上市公司代表性指数 | SPY 跟踪对象 | S&P 500 用于代表美国大型公司整体表现。 |
| 日线 | Daily Bar | 每个交易日一条行情 | 项目数据频率 | 本项目使用 SPY 日线数据。 |
| OHLCV | Open High Low Close Volume | 开高低收和成交量 | 原始行情核心字段 | OHLCV 是技术特征的基础。 |
| 开盘价 | Open | 当日开始交易附近的价格 | 保留为 ML 输入 | Open 描述交易日开盘价格。 |
| 最高价 | High | 当日最高成交价格 | 保留为 ML 输入 | High 描述日内上沿。 |
| 最低价 | Low | 当日最低成交价格 | 保留为 ML 输入 | Low 描述日内下沿。 |
| 收盘价 | Close | 当日结束交易附近的价格 | 收益率、均线、回测主要口径 | 项目主要用 Close 构造收益与趋势。 |
| 复权收盘价 | Adjusted Close | 考虑拆分和分红等因素后的价格 | 保留为输入，但未用于收益计算 | Adj Close 被保留，但当前收益口径使用 Close。 |
| 成交量 | Volume | 当日成交活跃程度 | 构造 `volume_ma_20` | Volume 用于描述市场活跃度。 |
| 收益率 | Return | 价格相对变化比例 | 风险、特征、回测、奖励基础 | 收益率比绝对价格更适合跨时间比较。 |
| 日收益率 | Daily Return | 一天的价格相对变化 | `return_1d` | `return_1d` 描述单日涨跌幅。 |
| 五日收益率 | Five-Day Return | 最近五天累计价格变化 | `return_5d` | `return_5d` 描述短周期累计变化。 |
| 累计收益率 | Cumulative Return | 多日收益复利累积后的总变化 | 权益曲线末值减 1 | 累计收益必须与风险一起阅读。 |
| 移动均线 | Moving Average | 最近若干天价格平均 | `ma_5`、`ma_10`、`ma_20`、`ma_60` | 移动均线平滑噪声并描述趋势。 |
| 滚动窗口 | Rolling Window | 随时间向后滑动的一段历史区间 | 均线、波动率和 LSTM 都使用 | 滚动窗口只使用当前及过去数据。 |
| 波动率 | Volatility | 收益变化有多剧烈 | `volatility_20`、年化波动率 | 波动率是风险的重要描述。 |
| 价格均线比 | Close-to-MA20 Ratio | 收盘价相对 20 日均线的位置 | `close_ma20_ratio` | 大于 1 通常表示价格在 20 日均线上方。 |
| 特征 | Feature | 提供给模型的输入变量 | OHLCV 加 9 项技术特征 | 特征是模型用于判断的输入信息。 |
| 特征工程 | Feature Engineering | 从原始数据加工输入变量 | Stage 2 | 特征工程把原始行情转成趋势、收益和风险表达。 |
| 标签 | Label | 模型要学习的答案 | 下一交易日涨跌方向 | 标签 1 表示下一交易日上涨。 |
| 分类 | Classification | 从有限类别中选择答案 | 上涨或不上涨二分类 | 本项目预测方向，不直接预测价格。 |
| 二分类 | Binary Classification | 只有两种类别的分类 | `0` 或 `1` | 项目把下一日方向写成二分类任务。 |
| 训练集 | Train Set | 用来拟合模型的数据 | 前 80% 时间段 | 训练集用于学习参数。 |
| 测试集 | Test Set | 用来评价泛化的数据 | 后 20% 时间段 | 测试集用于评价未参与训练的数据。 |
| 验证集 | Validation Set | 用来调参的数据 | 当前项目未单独设置 | 严格实验应将验证集和最终测试集分开。 |
| 时间序列切分 | Time Series Split | 按时间先后划分数据 | Stage 3、Stage 5 | 金融数据不能随机混入未来样本。 |
| 数据泄漏 | Data Leakage | 模型提前使用了不该知道的信息 | 需要持续防范 | 数据泄漏会让评估虚高。 |
| 未来函数 | Look-Ahead Bias | 今天使用了明天才能知道的信息 | Stage 4 `shift(1)`、Stage 6 下一期奖励 | 未来函数会破坏回测可信度。 |
| 逻辑回归 | Logistic Regression | 通过 sigmoid 输出分类概率的线性模型 | Stage 3 基线 | 逻辑回归简单可解释，但当前全猜上涨。 |
| sigmoid | Sigmoid Function | 把任意实数压到 0 到 1 | LR 与 LSTM 输出 | sigmoid 常用于输出二分类概率。 |
| 权重 | Weight | 特征影响程度的参数 | LR、LSTM 都有 | 权重由训练过程学习。 |
| 偏置 | Bias | 模型整体基准倾向 | LR、LSTM 都有 | 偏置类似模型的基础起点。 |
| 随机森林 | Random Forest | 多棵决策树集成投票 | Stage 3 非线性对照 | 随机森林复杂，但当前泛化较弱。 |
| 决策树 | Decision Tree | 逐层做特征阈值判断 | RF 的基本单元 | 决策树通过条件切分样本。 |
| Bagging | Bootstrap Aggregating | 多次抽样训练再汇总 | RF 的重要思想 | Bagging 用多模型汇总降低单模型不稳定。 |
| 特征重要性 | Feature Importance | 特征对模型判断的重要程度 | Stage 3 输出表 | 特征重要性用于解释模型，但不是因果证明。 |
| RNN | Recurrent Neural Network | 能传递历史状态的神经网络 | LSTM 的基础背景 | RNN 用隐藏状态处理序列。 |
| LSTM | Long Short-Term Memory | 带门控的循环神经网络 | Stage 5 | LSTM 适合序列，但不保证金融预测更准。 |
| 隐藏状态 | Hidden State | 当前时间步对外传递的表达 | LSTM 内部状态 | 隐藏状态概括当前序列信息。 |
| 细胞状态 | Cell State | 跨时间传递的长期记忆通道 | LSTM 内部状态 | 细胞状态帮助保留长期信息。 |
| 遗忘门 | Forget Gate | 决定旧记忆保留多少 | LSTM 门控 | 遗忘门控制哪些历史信息应被丢弃。 |
| 输入门 | Input Gate | 决定新信息写入多少 | LSTM 门控 | 输入门控制新信息写入记忆。 |
| 输出门 | Output Gate | 决定当前输出多少 | LSTM 门控 | 输出门控制细胞状态对外呈现程度。 |
| 回看窗口 | Lookback Window | 一个序列样本包含多少天 | LSTM 使用 20 天 | 每个 LSTM 输入样本含 20 天历史。 |
| 标准化 | Standardization | 将变量调整到相近尺度 | LSTM 训练集 fit 后作用于测试集 | 标准化应只在训练集拟合，避免泄漏。 |
| Accuracy | Accuracy | 总体预测正确比例 | LR、RF、LSTM 指标 | Accuracy 要结合类别分布阅读。 |
| Precision | Precision | 预测上涨中真的上涨比例 | 分类指标 | Precision 关注预测为正时有多可靠。 |
| Recall | Recall | 真实上涨中被找到的比例 | 分类指标 | Recall 过高或过低都要结合预测分布解释。 |
| F1 | F1 Score | Precision 与 Recall 的调和平均 | 分类指标 | F1 用于平衡精确率和召回率。 |
| AUC | Area Under ROC Curve | 分类排序区分能力 | 当前只输出 LSTM AUC | LSTM AUC 0.4916 接近随机水平。 |
| 回测 | Backtest | 在历史数据上模拟规则 | Stage 4、Stage 6 | 回测只说明历史，不保证未来。 |
| 策略收益 | Strategy Return | 按仓位规则得到的收益 | Stage 4 与 Stage 6 | 策略收益取决于仓位、市场收益和成本。 |
| 买入持有 | Buy and Hold | 长期保持持仓 | Stage 6 基线 | B&H 是判断复杂策略是否增值的基础对照。 |
| 交易成本 | Transaction Cost | 买卖带来的费用损耗 | Stage 6 变仓扣 0.1% | 交易成本会削弱频繁交易。 |
| 滑点 | Slippage | 理想价格与实际成交价格差异 | 当前未实现 | 当前回测未充分考虑滑点。 |
| 年化收益率 | Annual Return | 把日均收益换算成年尺度 | 风险表和策略表 | 年化收益要与风险指标一起看。 |
| 年化波动率 | Annual Volatility | 把日波动换算成年尺度 | 风险表和策略表 | 年化波动越高，历史不确定性通常越强。 |
| Sharpe ratio | Sharpe Ratio | 单位波动对应收益 | 风险表和策略表 | 本项目 Sharpe 未扣无风险利率。 |
| 最大回撤 | Maximum Drawdown | 历史权益最深跌幅 | SPY 为 -34.10% | 最大回撤体现持有过程中最痛苦的区间。 |
| Calmar ratio | Calmar Ratio | 年化收益相对最大回撤 | Stage 4 风险指标 | Calmar 用回撤而不是波动衡量风险。 |
| VaR | Value at Risk | 尾部损失分界 | SPY 95% VaR 为 -1.68% | VaR 是坏到一定程度的损失门槛。 |
| CVaR | Conditional Value at Risk | 进入尾部后的平均损失 | SPY 95% CVaR 为 -2.73% | CVaR 比 VaR 更关注极端损失。 |
| 偏度 | Skewness | 分布左右不对称程度 | SPY 为轻微负偏 | 负偏表示左侧尾部相对更值得关注。 |
| 峰度 | Kurtosis | 分布尾部厚度相关指标 | SPY 峰度较高 | 高峰度提示极端波动不能忽视。 |
| 权益曲线 | Equity Curve | 资金随时间变化曲线 | Stage 4 和 Stage 6 图片 | 权益曲线能展示收益过程和回撤。 |
| 强化学习 | Reinforcement Learning | 通过交互奖励学习行为 | Stage 6 | RL 不只预测，还学习动作价值。 |
| 智能体 | Agent | 做决策的主体 | Q-learning agent | 智能体根据状态选择仓位动作。 |
| 环境 | Environment | 接收动作并返回奖励的系统 | `StockTradingEnv` | 环境定义了交易过程的规则。 |
| 状态 | State | 决策时观察到的信息 | 三个离散变量组合 | 状态是智能体决策依据。 |
| 动作 | Action | 智能体可以选择的行为 | 空仓或持有 | 项目动作空间只有 2 类。 |
| 奖励 | Reward | 动作之后得到的反馈 | 下一期收益减变仓成本 | 奖励驱动智能体更新 Q 值。 |
| 策略 | Policy | 从状态到动作的规则 | 训练时 epsilon-greedy，评估时贪心 | 策略决定在不同状态下怎么行动。 |
| Q 表 | Q-table | 状态动作价值表格 | 27 状态 x 2 动作 | Q 表保存每个状态动作组合的价值估计。 |
| Q-learning | Q-learning | 用时序差分更新 Q 表的算法 | Stage 6 | Q-learning 用奖励和未来价值修正当前 Q 值。 |
| epsilon-greedy | Epsilon-Greedy | 随机探索与选择最优动作结合 | 训练时使用 | epsilon-greedy 平衡探索和利用。 |
| 学习率 | Learning Rate / Alpha | 每次更新改多少 | Q-learning `alpha=0.1` | 学习率控制新经验影响程度。 |
| 折扣因子 | Discount Factor / Gamma | 未来奖励重要程度 | Q-learning `gamma=0.95` | 折扣因子控制长期奖励权重。 |
| episode | Episode | 从环境开始到结束的一轮交互 | Q-learning 训练 50 轮 | 每个 episode 都会走完整段历史。 |
| 基线 | Baseline | 用来比较的简单方法 | LR、Random、Buy & Hold | 基线用于判断复杂方法是否真正增值。 |
| 样本外评价 | Out-of-Sample Evaluation | 在未参与训练的数据上测试 | RL 当前未严格实现 | 样本外评价更接近泛化能力判断。 |

