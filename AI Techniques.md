[TOC]



# 1 Introduction to AI

- 定义：A branch of computer science that deals with creating computer systems or software that can do tasks that usually require human intelligence.

- 图灵测试：图灵测试是一组指导原则，用于测试机器是否表现出类似于人类的智能行为。

- **AI分支**：

  - Machine Learning (ML)：Supervised，Unsupervised，Deep Learning

  - Natural Language Processing (NLP)：Content Extraction，Classification，Machine Translation，Question Answering，Text Generation

  - Expert Systems

  - Vision：Image Recognition，Machine Vision

  - Speech：Speech to Text，Text to Speech

  - Planning

  - Robotics

- 人和机器学习对比

| Category     | Details                                                      |
| ------------ | ------------------------------------------------------------ |
| Similarities | Learn, Experience, Feedback                                  |
| Differences  | General vs. specific learning, Small vs. big data, Learning speed |

- **Designing a Learning System**

  - Choosing Training Experience

  - Choosing Target Function

  - Choosing Representation of Target Function

  - Choosing Function Approximation

  - Final Design

- **ML Models**

  **Supervised Learning**

  - 从带标签的数据中学习。
  - 常用算法：决策树（Decision Tree）、支持向量机（Support Vector Machine）。
  - 示例应用：分类、回归等任务。

  **Unsupervised Learning**

  - 从无标签的数据中学习。
  - 常用算法：K均值聚类（K-Means Clustering）。
  - 示例应用：聚类、降维等任务。

  **Reinforcement Learning**

  - 通过持续的试错和奖励学习做出决策。
  - 常用方法：马尔可夫决策过程（Markov Decision Process）。
  - 示例应用：机器人控制、游戏策略等。

- **Types of Supervised Learning**

  - **Classification**
    - 用于预测类别型输出的问题，如正负分类、是否活体等。
    - 示例任务：
      - 图像分类（如医疗图像分类）
      - 文本分类（如情感分析、垃圾邮件检测）


  - **Regression**
    - 用于预测数值型输出的问题，如重量、预算等。
    - 示例任务：
      - 股票价格预测
      - 房价预测（基于面积）
      - 收入预测

- **Machine Learning vs Deep Learning**

  - **Machine Learning**：通过手动提取特征进行分类。例如，在图像识别中，需要人类定义特征（如边缘、形状等），然后机器学习算法进行分类。

  - **Deep Learning**：利用卷积神经网络（CNN）自动提取特征，减少人工干预。在图像识别中，深度学习模型能够从数据中直接学习特征，随着数据量的增加性能会显著提升。


- **Artificial Intelligence vs Machine Learning vs Deep Learning**

  - **Artificial Intelligence (AI)**：发展智能系统和机器，执行通常需要人类智能的任务。

  - **Machine Learning (ML)**：创建算法，从数据中学习并根据观察到的模式做出决策。当决策错误时可能需要人类干预。

  - **Deep Learning (DL)**：利用人工神经网络自动做出准确决策，通常不需要人类干预。
  - 关系AI包含ML，ML包含DL

- **Generative AI**
  - Generative AI models learn patterns from input data and then generate new content based on that learned information.
  - 模型使用神经网络和机器学习



# 2 INTELLIGENT AGENT

智能代理（Intelligent Agent）是一个自主实体，它通过传感器（sensors）感知环境（environment），并通过执行器（actuators）对环境进行行动（actions），以实现特定目标。

- 智能代理通过“感知-行动”循环来进行交互。它首先通过传感器接收来自环境的信息（称为感知 perceives），然后根据感知到的状态进行决策，最后通过执行器对环境产生影响。

<font color=blue>**Characteristics**</font>

| 特点                                 | 定义                                                         | 重要性                                                       | 示例                                                         |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 自主性 (Autonomy)                    | 智能代理无需人类或其他系统的直接干预即可独立运行             | 使系统能够独立决策，适用于需要实时响应且人工干预困难的应用   | 自主无人机避开障碍物，自动驾驶汽车换道决策                   |
| 社交能力 (Social Ability)            | 智能代理能够与其他代理、系统或人类有效互动                   | 使代理能够收集、共享和利用集体信息，对提升协作效率至关重要   | 聊天机器人参与人类式对话，多代理系统协同解决复杂问题         |
| 反应性 (Reactivity)                  | 智能代理能够实时感知并响应其环境                             | 使代理能够适应变化的情况，对安全关键应用至关重要             | 工业机器人应对意外障碍，语音助手响应语音指令                 |
| 前瞻性 (Proactiveness)               | 智能代理能够基于预测建模和对未来事件的预判主动采取行动       | 使代理具备前瞻性，提前规划行动，并在动态环境中具备竞争优势   | 智能温控器预测用户行为调整温度，交易机器人预测市场变化并预先交易 |
| 学习与适应 (Learning and Adaptation) | 智能代理能够通过从经验中学习来提升其表现，并根据情况调整行动 | 确保持续改进和优化，使代理在不断变化的情境中保持相关性和效率 | 根据用户推荐个性化内容，游戏中自适应AI根据玩家技能调整       |

<font color=blue>**Criterion to Design an Agent**</font>

**PEAS**: Performance, Environment, Actuators, Sensors

例子：

| Agent Type  | Performance Measure                                   | Environment                                  | Actuators                                           | Sensors                                                      |
| ----------- | ----------------------------------------------------- | -------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| Taxi Driver | Safe, fast, legal, comfortable trip, maximize profits | Roads, other traffic, pedestrians, customers | Steering, accelerator, brake, signal, horn, display | Cameras, sonar, speedometer, GPS, odometer, accelerometer, engine sensor, keyboard |

**Environment Type**

| 环境类型                              | 定义                                                         | 示例                                                 |
| ------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| 完全可观察环境 (Fully Observable)     | 智能代理可以在任意时间点访问环境的所有状态和细节             | 国际象棋游戏，所有棋子的位置都对双方玩家可见         |
| 部分可观察环境 (Partially Observable) | 智能代理只能有限度地访问环境的状态或细节，部分信息可能是隐藏或未知的。 | 扑克牌游戏，玩家看不到其他玩家的手牌                 |
|                                       |                                                              |                                                      |
| 确定性环境 (Deterministic)            | 行动的结果是预定的和确定的                                   | 国际象棋，在相同状态和动作下，结果状态总是相同的     |
| 随机性环境 (Stochastic)               | 行动的结果具有概率性，可能会变化                             | 股票市场预测，因不可预见的因素，决策可能导致不同结果 |
|                                       |                                                              |                                                      |
| 回合式环境 (Episodic)                 | 智能代理在不同的回合中体验环境，每个回合是独立的             | 机器人吸尘器清扫房间                                 |
| 顺序式环境 (Sequential)               | 智能代理有连续的体验，决策具有长期影响                       | 自动驾驶车辆导航                                     |
|                                       |                                                              |                                                      |
| 静态环境 (Static)                     | 环境在智能代理采取行动之前保持不变                           | 数独游戏，棋盘在玩家行动前保持不变                   |
| 动态环境 (Dynamic)                    | 环境可以在智能代理思考时改变                                 | 股票市场交易，价格随外部因素波动                     |
|                                       |                                                              |                                                      |
| 离散环境 (Discrete)                   | 环境有有限的、独立的状态或结果                               | 国际象棋中，智能代理选择有限的合法移动               |
| 连续环境 (Continuous)                 | 环境有无限数量的状态                                         | 自动驾驶车辆导航交通                                 |
|                                       |                                                              |                                                      |
| 单代理环境 (Single-Agent)             | 只有一个智能代理进行操作和决策，其成功取决于自身行为         | 机器人吸尘器在房间中独立清洁                         |
| 多代理环境 (Multi-Agent)              | 多个智能代理同时运行，彼此可能协作或竞争                     | 多辆自动驾驶汽车在繁忙的十字路口导航                 |

<font color=blue>**Agent Types**</font>

| Agent Type               | 特点                                                   | 描述                                         | 示例应用                                     |
| ------------------------ | ------------------------------------------------------ | -------------------------------------------- | -------------------------------------------- |
| Simple Reflex Agent      | 基于条件-动作规则直接做出反应，不考虑环境的状态        | 只根据当前感知做出反应，没有记忆或状态       | 简单机器人吸尘器，在脏的时候清扫，干净时移动 |
| Model-Based Reflex Agent | 通过状态维护环境的模型，可以记住部分过去的信息         | 基于当前感知和内置的环境模型做出决策         | 可以记住房间状态的清扫机器人                 |
| Goal-Based Agent         | 具备目标导向，根据设定的目标选择行动                   | 为达到目标而行动，考虑长远的效果             | 导航系统，寻找从A点到B点的路径               |
| Utility-Based Agent      | 选择能带来最高效用的行动，评估行动带来的幸福感或满意度 | 考虑目标及其效用，基于效用值来决定行动       | 自动驾驶车辆选择最安全或最省油的路径         |
| Learning Agent           | 具备学习能力，通过反馈和经验不断提高                   | 包含学习和性能模块，可以根据环境反馈调整策略 | 智能助手，如Alexa，根据用户的偏好调整响应    |

<font color=blue>**Future of Agent**</font>

- 与物联网设备的集成
- 提升类人交互能力
- 人工智能代理的伦理考量
- 潜在挑战：安全性、保障性、可信度

<font color=blue>**总结**</font>

| Topic                        | Summary                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| Nature of Intelligent Agents | 智能代理的本质是自主实体，能够感知环境、进行推理并做出决策，以实现特定目标。其应用范围从简单的自动化系统到复杂的AI驱动工具 |
| Architectural Diversity      | 介绍了不同类型智能代理的架构，从简单反射型代理到学习型代理，并分析了架构中增加的不同功能，以提高代理的性能 |
| Applications in Real-world   | 智能代理在日常生活中的广泛应用，从虚拟助手到推荐系统，突显了其在现代技术中的重要性 |
| Future Landscape             | 随着与物联网设备的深入集成及类人交互能力的追求，智能代理的未来充满机遇，但也面临挑战，如伦理对齐、安全性及可信度问题 |
| The Imperative to Understand | 随着数字与现实世界的界限逐渐模糊，理解智能代理的机制、潜力和风险变得至关重要。它们不仅能重塑行业、提升用户体验，还带来有关机器在我们生活中角色的哲学和伦理问题 |

# 3 Problem Solving by Searching

- **定义**：在AI中，问题求解是从起始状态移动到目标状态的过程。
- 多条路径可能通向目标，但并非所有路径都高效或可行。  一些问题可能无解。

- 目标导向型智能代理需要达成特定目标

**问题表述**  
- 状态空间：AI运行的环境，包含所有可能的状态
- 目标测试：确定某状态是否为目标状态
- 后继函数：提供从当前状态可能采取的所有行动
- 路径成本：从一个状态移动到另一个状态或达成目标的努力衡量

<font color=blue>**搜索算法类型**</font>

<font color = red>todo：复习每个搜索算法，给定一个图知道遍历顺序</font>

## 3.1 无信息搜索Uninformed/Blind Search

仅使用问题定义中的信息来进行搜索的策略。

- Breadth-first Search (BFS)
- Depth-first Search (DFS)
- Uniform-cost Search (UCS)
- Depth-limited Search
- Iterative Deepening Search
- Bidirectional Search

<font color=blue>**(1)广度优先搜索 (BFS)**</font>

**定义**：广度优先搜索是一种先探索当前深度所有节点，然后再移动到下一深度层的搜索策略

**特点**：

1. **完备性**：如果存在解，BFS一定会找到它
2. **最优性**：在均匀代价问题中，BFS能保证找到代价最小的解
3. **时间复杂度**：O(b^d)，其中b是分支因子，d是深度
4. **空间复杂度**：O(b^d)，因为需要存储当前层的所有节点以生成后继节点

**过程**：

- 使用队列（FIFO-先进先出）数据结构来存储节点。
- 选择图中的任意节点作为根节点，从该节点开始遍历。
- 遍历图中所有节点并逐个标记为完成。
- 访问相邻的未访问节点，标记为已访问，并将其插入队列。
- 如果没有相邻未访问节点，从队列中移除前一个节点。
- BFS算法会一直迭代，直到图中的所有节点都遍历并标记完成。
- 在遍历过程中不会出现环路。

<font color=blue>**(2)深度优先搜索 (DFS)**</font>

**概念**：递归算法，使用回溯来探索路径

**类比**：类似于解决迷宫问题，沿一条路径行进，直到遇到死胡同

**步骤**：

- 遇到死胡同时，返回到上一个未尝试的路径
- 如果没有前进的路径，则回退
- 在切换路径前，会先彻底探索当前路径上的所有节点
- 目标：每个节点都访问一次
- 关键思想：尽可能深入，再回溯

<font color=blue>**(3)深度限制搜索 (Depth-Limited Search, DLS)**</font>

**定义**：深度限制搜索（DLS）是深度优先搜索（DFS）的一个变种。

**特点**：不同于DFS，DLS设置了一个预定义的深度限制，防止陷入无限循环或过深的搜索。

**终止条件**：DLS有两个失败终止条件：

1. **标准失败值** (Standard failure value) ：当搜索无法在给定的深度限制内找到目标节点，且没有其他路径可供探索时，返回标准失败值。也就是说，在当前深度限制下搜索已经完成，但未能找到解决方案。
2. **截止失败值** (Cutoff failure value)：当搜索到达了深度限制，但仍可能存在更深层的路径时，返回截止失败值。这意味着搜索并未真正失败，而是被深度限制终止。可以通过增加深度限制来继续探索更深的路径。

<font color=blue>**(4)迭代加深搜索 (Iterative Deepening Search, IDS)**</font>

**定义**：迭代加深算法是深度优先搜索（DFS）和广度优先搜索（BFS）的一种组合

**过程**：

- 逐步增加搜索深度限制，直到找到目标，从而找到最佳的深度限制
- 每次迭代执行到特定的“深度限制”，在目标节点未找到时继续增加深度限制

**优点**：当搜索空间较大且目标节点深度未知时，IDS是非常有用的无信息搜索策略

<font color=blue>**(5)一致代价搜索 (Uniform Cost Search, UCS)**</font>

- 用于遍历加权树或图
- 探索累积代价最低的路径
- 使用优先队列实现，以最低代价为优先
- 当边有不同代价时启用
- 目标是找到到达目标节点的最小代价路径
- 如果所有边的代价相同，则等同于广度优先搜索



## 3.2 有信息搜索Informed/Heuristic Search

**智能搜索和场景背景**（**Intelligent Search and Scenario Context**）：智能搜索旨在通过避免穷举路径来有效地寻找解决方案，但在复杂场景中，有时可能会陷入死胡同。

**启发式搜索**（**Heuristic Search**）：此方法利用**启发式（经验法则）**来指导搜索过程，帮助跳过不相关区域，专注于可能的路径。然而，启发式搜索并不保证找到解决方案，可能会卡住。

**改进盲目搜索策略**（**Improving Blind Search Strategy**）：启发式函数用于避免盲目搜索的低效率，使搜索过程更加高效。目标是减少不必要的探索，从而节省时间和计算资源。

**选择一个好的启发式**（**Choosing a Good Heuristic**）：并非所有启发式方法都同样有效。一个好的启发式可以最小化搜索中所需的节点数量，从而提高搜索效率。提供更好指引的启发式被称为“信息更充分”的启发式。

<font color=blue>**(1)Best First Search**</font>

最佳优先搜索也称为贪心最佳优先搜索，通过启发式函数估算每个节点的代价，并选择看起来最接近目标的节点。

- 最佳优先搜索的核心思路是每次选择看起来离目标节点最近的节点，从而避免全面搜索。

**代价函数 f(n)**：评估每个节点的代价，通过代价函数f(n)来选择节点。

- 实际上，代价函数f(n)是一个启发式函数h(n)。
- h(n) 表示从节点 n 到目标节点的估计代价。
- f(n) 表示通过节点 n 到目标节点的总估计代价。

**节点选择**：选择估计代价最低的节点进行探索。

**操作步骤**：

1. 从初始节点开始。
2. 持续选择估计代价最低的节点。
3. 当到达目标节点时停止搜索。

<font color=blue>**A* Search**</font>

A* 是一种领先的路径搜索算法，用于找到从起点到目标的最优路径。其核心思想是选择看起来最有希望的节点进行探索，依据一个代价函数 $$f(n)$$ 来评估每个节点。

- A* 搜索算法通过结合实际代价 $$g(n)$$ 和启发式估计 $$h(n)$$，能有效地找到从起点到目标的最优路径。它利用开放和封闭列表来管理节点的状态，确保最小化搜索路径。

**代价函数：**$$f(n) = g(n) + h(n)$$

- $$g(n)$$：从起点到节点 $$n$$ 的实际代价。
- $$h(n)$$：从节点 $$n$$ 到目标的估计代价（启发式）。
- $$f(n)$$：从起点通过节点 $$n$$ 到目标的总估计代价。

**节点选择**：选择当前代价 $$f(n)$$ 最低的节点来继续探索。

**操作步骤**

1. 维护两个列表：
   - 开放节点（Open Nodes）：存放已发现但尚未评估的节点。
   - 封闭节点（Closed Nodes）：存放所有已评估的节点。(直线距离肯定比实际距离短,所以后面出现了实际加起来比前面直线算的要短的，可以封闭节点)
2. 初始节点：从初始节点开始。
3. 发现和评估节点：每次从开放节点中选择代价最低的节点进行探索。
4. 移动节点：
   - 探索后，节点进入开放列表。
   - 评估完成后，节点从开放列表移到封闭列表。
5. 终止条件：当目标节点成为当前代价最低的节点时，搜索结束。

**A* 搜索算法的性质**

- 完备性？是的（除非存在无限多个满足 $$f \leq f(G)$$ 的节点）
- 时间复杂度？指数级
- 空间复杂度？需要将所有节点保存在内存中
- 最优性？是的

<font color=blue>**极小极大算法（Minimax Algorithm）**</font>

- 极小极大算法是一种**回溯算法**，用于在决策和博弈论中帮助玩家找到最佳行动。
- 通常应用于**双人回合制游戏**，如井字棋（Tic-Tac-Toe）、西洋双陆棋（Backgammon）、曼卡拉（Mancala）、国际象棋（Chess）等。
- 在算法中，有两种角色的玩家：
  - 极大化者（Maximizer）：目标是获得最高得分
  - 极小化者（Minimizer）：目标是获得最低得分

在这个算法中，轮到最小化玩家选择的时候，最小化玩家会选择那个能够导致最大化玩家得分最低的节点。一旦最小化玩家做出选择，那个选择的分值就成为当前路径的最终得分，供最大化玩家在根节点做决定时参考。

<font color=blue>**Alpha-Beta 剪枝算法**</font>

在 Alpha-Beta 剪枝算法中：

- **最大化玩家的目标**是获得尽可能高的分数。
- **最小化玩家的目标**是让最大化玩家获得尽可能低的分数。

所以在决策树中，最大化和最小化节点的角色会交替进行，每个节点都在试图为自己争取最佳结果。

**何时剪枝？**

1. **定义**：
   - **α** 是当前已知的 **最大化玩家**可以确保的最低得分（在最大化节点上逐步更新）。
   - **β** 是当前已知的 **最小化玩家**可以确保的最高得分（在最小化节点上逐步更新）。
2. **场景描述**：
   - 假设我们在一个 **最小化节点**上，已经有一个分支得到了值 V，并且V≤α。
   - 这个值 V 表示的是这个分支的最小得分。而 **最大化玩家的目标是获得比 α 更高的得分**（因为 α 是在之前的最大化节点上选择的最佳分数下界）。
3. **推理过程**：
   - 如果当前分支的得分 V≤α，这意味着最小化玩家可以通过选择这个分支，让最大化玩家的得分小于或等于 α。
   - **对于最大化玩家来说，这是一个不可接受的选择**，因为他可以选择其他分支获得更高的得分（至少是 α）。
4. **结论**：在这种情况下，**我们可以安全地停止探索该分支的其他可能性**。即便后续的子节点可能存在更低的分值，最大化玩家都不会选择这一条路径。

---------

<font color = blue>**做题步骤：**</font>

1.初始化所有节点为 $$\alpha = -\infty$$ , $$\beta = \infty$$ ，然后深度优先搜索到叶子节点，再开始对节点进行修改

2.修改 $$\alpha$$ 和 $$\beta$$ 

| 操作              | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| 修改 $$\alpha$$  | 当前节点的所有子节点的  $$\alpha$$ ， $$\beta$$  和 value 的最大值 |
| 修改 $$\alpha$$   | 当前节点的所有子节点的  $$\alpha$$ ， $$\beta$$  和 value 的最小值 |
| 传递              | value根据minimax进行传递;  $$\alpha$$ ， $$\beta$$  向下传递 |
| 限制条件          | MAX 节点只能修改  $$\alpha$$ ，MIN 节点只能修改  $$\beta$$   |
| 趋势              |  $$\alpha$$ 不断变大， $$\beta$$ 不断变小                    |
| 剪枝              |  $$\alpha > \beta$$ 或  $$\alpha \geq \beta$$ (根据具体题目要求) |

- 单枝的直接传上去
- 是从左到右一个一个来的，不要一下把所有叶子算完


# 4 KNOWLEDGE REPRESENTATION & REASONING

**Introduction（介绍）**

- **Knowledge Representation (KR)（知识表示）** - 将复杂的现实世界信息转化为AI系统可利用的格式，从而模拟“智能”行为的方法。
- **Significance of KR（KR的意义）**
  - 提供必要的信息基础，促进智能决策。
  - 作为推理的前提——使AI具备推理和决策的能力。
- **Knowledge Reasoning（知识推理）** - AI系统从存储的知识中得出新结论的过程。涉及逻辑推理、推断规则和决策制定。

**What to Represent（表示什么）**

- **Object（对象）**: 关于我们世界领域中对象的所有事实。例如，吉他有琴弦，小号是铜管乐器。
- **Events（事件）**: 事件是发生在我们世界中的行为。
- **Performance（表现）**: 描述涉及如何做事情的知识的行为。
- **Meta-knowledge（元知识）**: 是关于我们所知道的知识。
- **Facts（事实）**: 是关于现实世界的真实情况和我们所表示的内容。
- **Knowledge-Base（知识库）**: 知识型代理的核心组件是知识库，表示为**KB**。知识库是**句子**的集合（这里，句子作为一个技术术语使用，与英语语言中的句子并不相同）。

## 4.1 Types of knowledge

**Type 1: Declarative Knowledge（类型 1：陈述性知识）**

- 陈述性知识是指对某事物的了解。（Declarative knowledge is to know about something.）
- 它包括概念、事实和对象。（It includes concepts, facts, and objects.）
- 它也称为描述性知识，并以陈述句表达。（It is also called descriptive knowledge and expressed in declarative sentences.）
- 它比程序性语言更简单。（It is simpler than procedural language.）

**Type 2: Procedural Knowledge（类型 2：程序性知识）**

- 也称为指令性知识。（It is also known as imperative knowledge.）
- 程序性知识是指知道如何做某事的知识。（Procedural knowledge is a type of knowledge that is responsible for knowing how to do something.）
- 它可以直接应用于任何任务。（It can be directly applied to any task.）
- 它包括规则、策略、程序、议程等。（It includes rules, strategies, procedures, agendas, etc.）
- 程序性知识取决于它可以应用的任务。（Procedural knowledge depends on the task on which it can be applied.）

**Type 3: Meta-knowledge（类型 3：元知识）**

- 关于其他知识类型的知识称为元知识。（Knowledge about the other types of knowledge is called Meta-knowledge.）

**Type 4: Heuristic Knowledge（类型 4：启发式知识）**

- 启发式知识是指某个领域或学科中一些专家的知识。（Heuristic knowledge is representing knowledge of some experts in a field or subject.）
- 启发式知识是基于过去经验的经验法则、对方法的认知，具有一定效果但不能保证。（Heuristic knowledge is rules of thumb based on previous experiences, awareness of approaches, and which are good to work but not guaranteed.）

**Type 5: Structural Knowledge（类型 5：结构性知识）**

- 结构性知识是解决问题的基本知识。（Structural knowledge is basic knowledge to problem-solving.）
- 它描述了各种概念之间的关系，如类别、部分和分组。（It describes relationships between various concepts such as kind of, part of, and grouping of something.）
- 它描述了概念或对象之间存在的关系。（It describes the relationship that exists between concepts or objects.）

---

**AI Knowledge Cycle（人工智能知识循环）**

- 一个人工智能系统具有以下组件来展示智能行为：（An Artificial intelligence system has the following components for displaying intelligent behavior:）
  - **感知（Perception）**
  - **学习（Learning）**
  - **知识表示和推理（Knowledge Representation and Reasoning）**
  - **规划（Planning）**
  - **执行（Execution）**

在这一循环中，感知数据进入系统，通过学习过程被处理，然后通过知识表示来存储并用于推理。推理的结果用于规划，最终执行相应的行动。（In this cycle, perception data enters the system, is processed through learning, then stored in knowledge representation for reasoning. The reasoning results are used for planning, leading to the execution of actions.）



## 4.2 Approaches to Knowledge Representation

**1. Simple Relational Knowledge（简单关系知识）**

- 这是使用关系方法存储事实的最简单方式，每个关于对象的事实都系统地以列的形式展示。（It is the simplest way of storing facts which uses the relational method, and each fact about a set of objects is set out systematically in columns.）
- 这种知识表示方法在数据库系统中很有名，表示不同实体之间的关系。（This approach of knowledge representation is famous in database systems where the relationship between different entities is represented.）
- 这种方法几乎没有推理的机会。（This approach has little opportunity for inference.）

**2. Inheritable Knowledge（可继承知识）**

- 在可继承知识方法中，所有数据都必须存储在类的层级结构中。（In the inheritable knowledge approach, all data must be stored in a hierarchy of classes.）
- 所有类应按一般形式或层次结构排列。（All classes should be arranged in a generalized form or a hierarchical manner.）
- 在这种方法中，我们应用继承属性。（In this approach, we apply inheritance property.）
- 元素从类的其他成员继承值。（Elements inherit values from other members of a class.）
- 这种方法包含可继承知识，显示实例与类之间的关系，称为实例关系。（This approach contains inheritable knowledge which shows a relation between instance and class, and it is called instance relation.）
- 每个独立的框架都可以表示属性及其值的集合。（Every individual frame can represent the collection of attributes and its value.）
- 在这种方法中，对象和值用方框表示。（In this approach, objects and values are represented in boxed nodes.）
- 使用箭头从对象指向其值。（We use arrows which point from objects to their values.）

**3. Inferential Knowledge（推理知识）**

- 推理知识方法以形式逻辑的形式表示知识。（Inferential knowledge approach represents knowledge in the form of formal logic.）
- 这种方法可以用于推导更多的事实。（This approach can be used to derive more facts.）
- 它保证正确性。（It guarantees correctness.）
- 示例：假设有两个陈述：
  - Marcus 是一个男人（Marcus is a man）
  - 所有男人都是凡人（All men are mortal）
- 它可以表示为：
  - man(Marcus)
  - $$\forall x = man(x) \rightarrow mortal(x)$$

**4. Procedural Knowledge（程序性知识）**

- 程序性知识方法使用小程序和代码来描述如何做特定的事情以及如何进行。（Procedural knowledge approach uses small programs and codes which describe how to do specific things, and how to proceed.）
- 在这种方法中，使用了一个重要规则，即If-Then规则。（In this approach, one important rule is used which is the If-Then rule.）
- 在这种知识中，可以使用LISP语言和Prolog语言等各种编程语言。（In this knowledge, we can use various coding languages such as LISP language and Prolog language.）
- 我们可以通过这种方法轻松表示启发式或领域特定知识。（We can easily represent heuristic or domain-specific knowledge using this approach.）
- 但并不一定能表示所有情况。（But it is not necessary that we can represent all cases in this approach.）



## 4.3 Techniques of Knowledge Representation

**1. Logical Representation（逻辑表示）**

- **语法**：语法规则决定了如何在逻辑中构建合法的句子，确定在知识表示中使用的符号以及如何书写这些符号。（Syntaxes are the rules that decide how we can construct legal sentences in the logic. It determines which symbol we can use in knowledge representation and how to write those symbols.）
- **语义**：语义规则帮助我们在逻辑中解释句子，包含给每个句子分配含义的过程。（Semantics are the rules by which we can interpret the sentence in the logic. Semantic also involves assigning a meaning to each sentence.）
- 逻辑表示主要分为两种逻辑：命题逻辑和谓词逻辑。（Logical representation can be categorized into mainly two logics: Propositional Logics and Predicate Logics.）

**2. Semantic Networks（语义网络）**

- 这种表示主要包含两种关系类型：IS-A关系（继承关系）和种类关系。（This representation consists of mainly two types of relations: IS-A relation (Inheritance) and Kind-of-relation.）
- 示例：以下是一些需要以节点和弧的形式表示的陈述。（Example: The following are some statements that we need to represent in the form of nodes and arcs.）
  - Jerry 是一只猫（Jerry is a cat）
  - Jerry 是哺乳动物（Jerry is a mammal）
  - Jerry 被 Priya 拥有（Jerry is owned by Priya）
  - Jerry 是棕色的（Jerry is brown colored）
  - 所有哺乳动物都是动物（All mammals are animals）

**3. Frame Representation（框架表示）**

- 包含属性集合及其值，用于描述世界中的一个实体。（Consists of a collection of attributes and its values to describe an entity in the world.）
- **面**：槽的各个方面称为面，面是框架的特征，允许对框架施加约束。（Facets: The various aspects of a slot are known as facets. Facets are features of frames which enable us to put constraints on the frames.）
- 示例：当需要某个特定槽的数据时调用IF-NEEDED事实。（Example: IF-NEEDED facts are called when data of any particular slot is needed.）
- 框架也称为槽-过滤知识表示。（A frame is also known as slot-filter knowledge representation.）

**4. Production Rules（产生式规则）**

- 产生式规则系统由（条件，动作）对组成，表示“如果条件满足，则执行动作”。（Production rules system consists of (condition, action) pairs which mean "If condition then action".）
- 主要包括三个部分：
  - **产生式规则集** - 检查条件，如果条件存在，则触发产生式规则并执行相应的动作。（The set of production rules - checks for the condition and if the condition exists then production rule fires and corresponding action is carried out.）
  - **工作记忆** - 包含问题解决当前状态的描述，规则可以将知识写入工作记忆中。（Working Memory - contains the description of the current state of problem-solving and rule can write knowledge to the working memory.）
  - **识别-行动循环** - 规则的条件部分确定可应用于问题的规则，动作部分执行相关的步骤。（The recognize-act-cycle - The condition part of the rule determines which rule may be applied to a problem. And the action part carries out the associated problem-solving steps.）
- 示例规则：
  - 如果在车站且公交车到达，则执行动作（上车）。（IF (at bus stop AND bus arrives) THEN action (get into the bus)）
  - 如果在公交车且已付费且有空座，则执行动作（坐下）。（IF (on the bus AND paid AND empty seat) THEN action (sit down)）
  - 如果在公交车且未付费，则执行动作（支付车费）。（IF (on bus AND unpaid) THEN action (pay charges)）
  - 如果公交车到达目的地，则执行动作（下车）。（IF (bus arrives at destination) THEN action (get down from the bus)）



## 4.4 First Order Logic (FOL)

**First-order Logic (FOL)（一阶逻辑）**

- 一阶逻辑也被称为谓词逻辑或一阶谓词逻辑。（First-order logic is also known as Predicate logic or First-order predicate logic.）
- 一阶逻辑不仅假设世界包含事实，还假设以下内容存在于世界中：
  - **对象**：如A、B、人、数字、颜色、战争、理论、正方形、坑等。（Objects: A, B, people, numbers, colors, wars, theories, squares, pits, etc.）
  - **关系**：可以是单一关系（如红色、圆形、邻接），或是n元关系（如姐妹、兄弟、拥有关联、介于之间）。（Relations: It can be unary relation such as red, round, is adjacent, or n-any relation such as the sister of, brother of, has color, comes between.）
  - **函数**：如父亲、最好的朋友、第三局、结束等。（Function: Father of, best friend, third inning of, end of, etc.）
- 作为一种自然语言，一阶逻辑也有两个主要部分：语法和语义。（As a natural language, first-order logic also has two main parts: Syntax and Semantics.）

**Syntax of FOL（FOL的语法）**

- FOL的语法确定了哪些符号的集合是逻辑表达式。（The syntax of FOL determines which collection of symbols is a logical expression in first-order logic.）
- 以下是FOL语法的基本元素：
  - 常量、变量、谓词、函数、连接词、量词、等价性等。（Constants, Variables, Predicates, Functions, Connectives, Quantifiers, Equality, etc.）

**Atomic Sentence（原子句）**

- 原子句是一阶逻辑中最基本的句子，由谓词符号和一系列项组成的括号构成。（Atomic sentences are the most basic sentences of first-order logic. These sentences are formed from a predicate symbol followed by a parenthesis with a sequence of terms.）
- 可以表示为谓词（项1，项2，…，项n）。（We can represent atomic sentences as Predicate (term1, term2, ..., term n).）
- 示例：Ravi和Ajay是兄弟 => Brothers(Ravi, Ajay)；Chinky是一只猫 => cat(Chinky)。（Example: Ravi and Ajay are brothers: => Brothers(Ravi, Ajay). Chinky is a cat: => cat(Chinky).）

**Complex Sentence（复合句）**

- 复合句是通过使用连接词将原子句结合在一起构成的。（Complex sentences are made by combining atomic sentences using connectives.）

**First-order Logic Statements（FOL语句）**

- FOL语句可以分为两部分：
  - **主语**：语句的主要部分。（Subject: Subject is the main part of the statement.）
  - **谓词**：谓词可以定义为一种关系，将两个原子结合在一个语句中。（Predicate: A predicate can be defined as a relation, which binds two atoms together in a statement.）
- 示例：“x是一个整数”中，x是主语，“是一个整数”是谓词。（Consider the statement: "x is an integer", where x is the subject and "is an integer" is known as a predicate.）

**Quantifiers in FOL（FOL中的量词）**

- 量词是生成量化的语言元素，指定宇宙中的样本数量。（A quantifier is a language element which generates quantification, specifying the quantity of specimen in the universe of discourse.）
- 有两种类型的量词：
  - **全称量词**，用于所有对象（Universal Quantifier: for all, everyone, everything）
  - **存在量词**，用于某些对象（Existential Quantifier: for some, at least one）

**Inference in FOL（FOL中的推理）**

- FOL的量词推理规则包括：
  - 全称泛化（Universal Generalization）
  - 全称实例化（Universal Instantiation）
  - 存在实例化（Existential Instantiation）
  - 存在引入（Existential Introduction）



## 4.5 Reasoning in AI

**Deductive Reasoning（演绎推理）**

- 从逻辑相关的已知信息中推导出新信息。（Deducing new information from logically related known information.）
- 当前提为真时，论点的结论也必须为真。（The argument's conclusion must be true when the premises are true.）
- 也称为自上而下推理，与归纳推理相对。（Referred to as top-down reasoning, and contradictory to inductive reasoning.）
- 示例：
  - 前提1：所有人类都吃蔬菜（Premise-1: All humans eat veggies）
  - 前提2：Suresh 是人类（Premise-2: Suresh is human）
  - 结论：Suresh 吃蔬菜（Conclusion: Suresh eats veggies）

**Inductive Reasoning（归纳推理）**

- 通过推广有限的事实集合来得出结论的推理形式。（A form of reasoning to arrive at a conclusion using limited sets of facts by the process of generalization.）
- 也称为因果推理或自下而上推理。（Also known as cause-effect reasoning or bottom-up reasoning.）
- 示例：
  - 前提：在动物园中见到的所有鸽子都是白色的（Premise: All of the pigeons we have seen in the zoo are white.）
  - 结论：因此，我们可以期望所有的鸽子都是白色的（Conclusion: Therefore, we can expect all the pigeons to be white.）

**Abductive Reasoning（溯因推理）**

- 一种从单一或多个观察开始并寻找最可能的解释或结论的逻辑推理形式。（A form of logical reasoning which starts with single or multiple observations and then seeks to find the most likely explanation or conclusion for the observation.）
- 示例：
  - 蕴含：如果下雨，板球场地会湿（Implication: Cricket ground is wet if it is raining）
  - 公理：板球场地是湿的（Axiom: Cricket ground is wet）
  - 结论：在下雨（Conclusion: It is raining）

**Common Sense Reasoning（常识推理）**

- 一种非正式的推理形式，通过经验获得。（An informal form of reasoning, which can be gained through experiences.）
- 依赖于良好的判断而不是精确的逻辑，基于启发式知识和规则。（It relies on good judgment rather than exact logic and operates on heuristic knowledge and heuristic rules.）
- 示例：
  - 一个人一次只能在一个地方。（One person can be at one place at a time.）
  - 如果我把手放在火里，它会被烧伤。（If I put my hand in a fire, then it will burn.）

**Monotonic Reasoning（单调推理）**

- 一旦得出结论，即使添加信息也不会改变结论。（Once the conclusion is taken, then it will remain the same even if we add more information to the existing information in our knowledge base.）
- 不适用于实时系统。（Not useful for real-time systems.）
- 单调推理用于传统推理系统。（Monotonic reasoning is used in conventional reasoning systems.）
- 示例：地球围绕太阳旋转。（Example: Earth revolves around the Sun.）

**Non-Monotonic Reasoning（非单调推理）**

- 如果我们向知识库添加更多信息，某些结论可能会失效。（Some conclusions may be invalidated if we add more information to our knowledge base.）

- 示例：

  - 假设知识库包含以下知识：鸟会飞，企鹅不会飞，Pitty 是鸟。因此我们可以得出结论：Pitty 会飞。（Suppose the knowledge base contains: Birds can fly, Penguins cannot fly, Pitty is a bird. So, we conclude: Pitty can fly.）
  - 但是，如果我们添加另一个信息“Pitty 是企鹅”，则推导出“Pitty 不能飞”，从而推翻之前的结论。（However, if we add "Pitty is a penguin", it concludes "Pitty cannot fly", thus invalidating the previous conclusion.）

  

<font color=blue>**Deductive vs. Inductive Reasoning**</font>

- **演绎推理（Deductive Reasoning）**
  - 通过已知相关事实和信息推导新信息或结论的推理形式，采用自上而下的方法。（Deductive reasoning is the form of valid reasoning, to deduce new information or conclusion from known related facts and information, following a top-down approach.）
  - 演绎推理从前提出发，要求前提为真时结论必须为真。（Deductive reasoning starts from premises and requires that the conclusion must be true if the premises are true.）
  - 演绎推理难以使用，因为需要事实必须为真。（Use of deductive reasoning is difficult, as we need facts which must be true.）
  - 过程：理论→假设→模式→确认。（Process: Theory → hypothesis → patterns → confirmation.）
  - 演绎推理中的论点可以是有效或无效的。（In deductive reasoning, arguments may be valid or invalid.）
  - 从一般事实推导具体结论。（Reaches from general facts to specific facts.）

- **归纳推理（Inductive Reasoning）**
  - 通过推广特定事实或数据来得出结论的推理形式，采用自下而上的方法。（Inductive reasoning arrives at a conclusion by the process of generalization using specific facts or data, following a bottom-up approach.）
  - 归纳推理从结论出发，但前提的真实性并不能保证结论的真实性。（Inductive reasoning starts from the conclusion, but the truth of premises does not guarantee the truth of conclusions.）
  - 归纳推理使用快速且易于使用的证据，而不是严格的事实，经常在日常生活中使用。（Use of inductive reasoning is fast and easy, as we need evidence instead of true facts. Often used in our daily life.）
  - 过程：观察→模式→假设→理论。（Process: Observations → patterns → hypothesis → Theory.）
  - 归纳推理中的论点可以是强或弱的。（In inductive reasoning, arguments may be weak or strong.）
  - 从具体事实推导出一般结论。（Reaches from specific facts to general facts.）

| Basis for comparison | Deductive Reasoning                                          | Inductive Reasoning                                          |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Definition           | 演绎推理是一种通过已知相关事实和信息推导新信息或结论的有效推理形式。 | 归纳推理通过推广特定事实或数据来得出结论的过程。             |
| Approach             | 演绎推理采用自上而下的方法。                                 | 归纳推理采用自下而上的方法。                                 |
| Starts from          | 演绎推理从前提出发。                                         | 归纳推理从结论出发。                                         |
| Validity             | 在演绎推理中，前提为真时结论必须为真。                       | 在归纳推理中，前提的真实性并不能保证结论的真实性。           |
| Usage                | 演绎推理难以使用，因为需要事实必须为真。                     | 归纳推理使用快速且易于使用的证据，而不是严格的事实，通常在日常生活中使用。 |
| Process              | 理论→假设→模式→确认。                                        | 观察→模式→假设→理论。                                        |
| Argument             | 演绎推理中的论点可以是有效或无效的。                         | 归纳推理中的论点可以是强或弱的。                             |
| Structure            | 从一般事实推导具体结论。                                     | 从具体事实推导出一般结论。                                   |


# 5 Machine Learning

## 5.1 Introduction

**机器学习：推动AI的引擎** 
机器学习是人工智能的一个分支，使机器能够从数据中学习，而无需明确编程。它使机器能够在没有人工干预的情况下进行预测或决策。（Machine learning is a subfield of AI that enables machines to learn from data without being explicitly programmed. It allows machines to make predictions or decisions without human intervention.）

**历史背景**  

- “机器学习”一词由Arthur Samuel在1959年提出。（The term "Machine Learning" was coined in 1959 by Arthur Samuel.）  
- 机器学习从简单算法演变为复杂的深度学习模型。（Evolution from simple algorithms to sophisticated deep learning models.）  
- 过去二十年中，由于计算能力和数据可用性的提高，机器学习取得了快速发展。（Rapid growth in the last two decades due to increased data availability and computational power.）

**传统建模与机器学习的对比**  

- 传统建模：基于手工创建的模型进行预测，依赖数据和人工规则。（Traditional modeling: Relies on handcrafted models and rules for predictions.）  
- 机器学习：基于样本数据进行训练，模型可以通过新数据不断改进。（Machine learning: Trains models using sample data and improves with new data.）

**人类学习与机器学习的异同**  

- **相似点**：  
  - 学习：人类通过认知过程学习，而机器从它们能访问的数据中学习。（Learning for humans involves a cognitive process, while for machines, it’s a function of their programming.）  
  - 反馈：人类和机器都通过反馈进行改进。（Both learn from feedback.）  
- **差异点**：  
  - 泛化学习与具体学习：人类更擅长在不同环境中泛化，而机器更专注于特定任务。（Humans excel at general learning, while machines focus on specific tasks.）  
  - 小数据与大数据：人类可以从少量数据中学习，而机器需要大量数据。（Machines need large datasets, while humans learn effectively from small data.）  
  - 学习速度：机器可以快速处理大量信息，而人类的学习较慢。（Machines process data quickly, but human learning is slower.）

**设计学习系统的步骤**  

1. 选择训练经验（Choosing Training Experience.）  
2. 选择目标函数（Choosing Target Function.）  
3. 选择目标函数的表示方式（Choosing Representation of Target Function.）  
4. 选择函数逼近方法（Choosing Function Approximation.）  
5. 完成设计（Final Design.）

**机器学习模型**  

- **监督学习**：从标注数据中学习，如决策树、支持向量机。（Supervised Learning: Learn from labeled data (e.g., Decision Tree, Support Vector Machine).）  
- **无监督学习**：从未标注数据中学习，如K均值聚类。（Unsupervised Learning: Learn from unlabeled data (e.g., K-Means Clustering).）  
- **强化学习**：通过试错法学习决策，如马尔可夫决策过程。（Reinforcement Learning: Learn by trial and error (e.g., Markov Decision Process).）

**监督学习的类型**  

- **分类**：输出变量是类别，例如图像分类、情感分析、垃圾邮件检测。（Classification: Output is a category (e.g., image classification, sentiment analysis).）  
- **回归**：输出变量是连续值，例如房价预测、收入预测。（Regression: Output is a continuous value (e.g., predicting house prices, revenue forecasting).）



## 5.2 Support Vector Machine

**今天的主题：支持向量机 (SVM)**  

- 支持向量机是一种基于统计学习理论的分类器，由Vapnik等人在1992年提出。（A classifier derived from statistical learning theory by Vapnik et al. in 1992.）  
- SVM在使用图像作为输入时，由于其分类准确性接近于神经网络，因此在手写识别任务中成为著名算法。（SVM became famous when, using images as input, it gave accuracy comparable to neural networks with hand-designed features in a handwriting recognition task.）  
- 目前，SVM广泛用于对象检测和识别、内容检索、文本识别、生物识别、语音识别等。（Currently, SVM is widely used in object detection & recognition, content-based image retrieval, text recognition, biometrics, speech recognition, etc.）  
- 此外，它也可用于回归，但今天不涉及。（Also used for regression but will not cover today.）

**大纲**  

- 线性判别函数（Linear Discriminant Function）  
- 大间隔线性分类器（Large Margin Linear Classifier）  
- 非线性SVM：核技巧（Nonlinear SVM: The Kernel Trick）  
- SVM演示（Demo of SVM）

**感知机回顾：线性分隔**  

- 二元分类可视为在特征空间中分隔两类的任务。（Binary classification can be viewed as the task of separating classes in feature space.）  
- 分类函数：$$f(y) = sign(w^T x + b)$$  

**线性分隔器**  

- 哪条线性分隔器最优？（Which of the linear separators is optimal?）

**分类间隔**  

- 从样本 $$x$$ 到超平面的距离为 $$\frac{w^T x + b}{||w||}$$。  
- 离超平面最近的样本称为支持向量。（Examples closest to the hyperplane are support vectors.）  
- 间隔 $$\rho$$ 为支持向量之间的距离。（Margin $$\rho$$ of the separator is the distance between support vectors.）

**最大间隔分类**  

- 根据直觉和PAC理论，最大化间隔是有利的。（Maximizing the margin is good according to intuition and PAC theory.）  
- 这意味着只有支持向量重要，其他训练样本可以忽略。（Implies that only support vectors matter; other training examples are ignorable.）

**线性SVM的数学定义**  

- 假设训练集为 $$\{(x_i, y_i)\}_{i=1}^n$$，每个样本 $$x_i \in R^d, y \in \{-1, 1\}$$，被超平面 $$w \cdot x + b$$ 以间隔 $$\rho$$ 分隔。（Let training set $$\{(x_i, y_i)\}_{i=1}^n, x \in R^d, y \in \{-1, 1\}$$ be separated by a hyperplane with margin $$\rho$$.）  
- 对于每个支持向量 $$x$$，不等式成立且距离可以重新定义为 $$\frac{w^T x + b}{||w||}$$。（For every support vector $$x$$, the above inequality is an equality. After rescaling $$w$$ and $$b$$, the hyperplane distance is $$\frac{w^T x + b}{||w||}$$.）  
- 间隔可以表示为 $$\rho = 2 \cdot \frac{1}{||w||}$$。（Then the margin can be expressed as $$\rho = 2 \cdot \frac{1}{||w||}$$.）

**优化问题的定义**  

- 要找到 $$w$$ 和 $$b$$，以使 $$||w||^2$$ 最小化，同时满足约束 $$y_i(w^T x_i + b) \geq 1$$。（Find $$w$$ and $$b$$ such that $$||w||^2$$ is minimized and $$y_i(w^T x_i + b) \geq 1$$.）  
- 转化为构建一个二次优化问题，其中引入了拉格朗日乘子以解决每个约束。（The solution involves constructing a dual problem where a Lagrange multiplier is associated with every inequality constraint in the primal problem.）

**优化问题的解**  

- 给定对偶问题的解 $$\alpha_1, \alpha_2, ..., \alpha_n$$，可以用原始变量表达为 $$w = \sum_i \alpha_i y_i x_i$$ 和 $$b = y_k - \sum_i \alpha_i y_i k(x_i, x_k)$$。（Given a solution $$\alpha_1, \alpha_2, ..., \alpha_n$$ to the dual problem, solution to the primal is $$w = \sum_i \alpha_i y_i x_i$$ and $$b = y_k - \sum_i \alpha_i y_i k(x_i, x_k)$$.）  
- 分类函数依赖于支持向量，但不需要显式求解 $$w$$。（Then classifying function is dependent only on support vectors; no need to explicitly solve $$w$$.）  
- 内积用于测试点和支持向量之间的关系，且通过对偶方法优化时，隐式包含训练点的内积。（Notice that it relies on an inner product between the test point and the support vectors $$x$$. Solving the dual optimization problem involved computing inner products of $$x$$ between all training points.）

**软间隔分类**  

- 如果训练集不是线性可分的，可以引入松弛变量 $$\xi_i$$，以允许对困难或噪声样本的错误分类，从而形成“软”间隔。（What if the training set is not linearly separable? Slack variables $$\xi_i$$ can be added to allow misclassification of difficult or noisy examples, resulting in a margin called soft.）

**软间隔分类的数学定义**  

- 旧的公式定义为： 
  $$\Phi(w) = w^T w$$ 最小化，同时对所有 $$(x_i, y_i)$$ 满足： 
  $$y_i(w^T x_i + b) \geq 1$$（Find $$w$$ and $$b$$ such that $$\Phi(w) = w^T w$$ is minimized and $$y_i(w^T x_i + b) \geq 1$$ for all $$(x_i, y_i).$$）  
- 修改后的公式引入了松弛变量： 
  $$\Phi(w) = w^T w + C \sum_{i=1}^n \xi_i$$ 最小化，同时满足： 
  $$y_i(w^T x_i + b) \geq 1 - \xi_i, \xi_i \geq 0$$（Find $$w$$ and $$b$$ such that $$\Phi(w) = w^T w + C \sum_{i=1}^n \xi_i$$ is minimized and $$y_i(w^T x_i + b) \geq 1 - \xi_i, \xi_i \geq 0.$$）  
- 参数 $$C$$ 可被视为控制过拟合的一种方式。（Parameter $$C$$ can be viewed as a way to control overfitting.）

**软间隔分类的解决方案**  

- 如果训练数据不是线性可分的，松弛变量 $$\xi_i$$ 可被引入以允许对困难或噪声样本的错误分类。（If the training data is not linearly separable, slack variables $$\xi_i$$ can be added to allow misclassification of difficult or noisy examples.）  
- 容许一定的错误：一些点可以移动到正确类别，但需付出代价。（Allow some errors: Let some points be moved to where they belong, at a cost.）  
- 尽量最小化训练集错误，并尽可能将超平面“远离”每个类别（即大间隔）。（Still, try to minimize training set errors, and to place hyperplane "far" from each class (large margin).）

**软间隔分类的对偶问题**  

- 对偶问题与线性可分的情况相同，但需要额外的拉格朗日乘子以处理松弛变量。（Dual problem is identical to separable case, but additional Lagrange multipliers are needed for slack variables.）  
- 对偶问题的解： 
  $$w = \sum_i \alpha_i y_i x_i$$ 
  $$b = y_k(1 - \xi_i) - \sum_i \alpha_i y_i x_i$$（Solution to the dual problem is $$w = \sum_i \alpha_i y_i x_i$$ and $$b = y_k(1 - \xi_i) - \sum_i \alpha_i y_i x_i.$$）  
- 分类函数可直接表示为： 
  $$f(x) = \sum_i \alpha_i y_i k(x_i, x) + b$$（Classification function: $$f(x) = \sum_i \alpha_i y_i k(x_i, x) + b.$$）

**最大间隔的理论依据**  

- Vapnik证明了以下结论： 
  $$h \leq \min\left(\left(\frac{\rho}{D}\right)^2 m_0 + 1\right)$$ 
  其中 $$\rho$$ 为间隔，$$D$$ 为可以包围所有训练样本的最小球体的直径，$$m_0$$ 为维度。（Vapnik has proved the following: $$h \leq \min\left(\left(\frac{\rho}{D}\right)^2 m_0 + 1\right),$$ where $$\rho$$ is the margin, $$D$$ is the diameter of the smallest sphere that can enclose all training examples, and $$m_0$$ is the dimensionality.）  
- 直观上，这意味着无论维度 $$m_0$$ 有多大，通过最大化间隔 $$\rho$$ 可以最小化VC维度，从而保持分类器的复杂度。（Intuitively, this implies that regardless of dimensionality $$m_0,$$ we can minimize the VC dimension by maximizing the margin $$\rho,$$ keeping the classifier's complexity small regardless of dimensionality.）

**线性SVM概述**  

- 分类器是分隔超平面。（The classifier is a separating hyperplane.）  
- 最“重要”的训练点是支持向量；它们定义了超平面。（Most "important" training points are support vectors; they define the hyperplane.）  
- 二次优化算法可识别具有非零拉格朗日乘子的支持向量训练点 $$\alpha_i$$。（Quadratic optimization algorithms can identify which training points $$x_i$$ are support vectors with non-zero Lagrangian multipliers $$\alpha_i.$$
- 在对偶公式中，支持向量仅通过内积出现。（In the dual formulation of the problem, training points appear only inside inner products.）

**非线性SVM**  

- 对于带有一些噪声的线性可分数据集效果很好。（Datasets that are linearly separable with some noise work out great.）  
- 如果数据集太难处理怎么办？可以考虑将数据映射到更高维的空间。（What are we going to do if the dataset is just too hard? How about mapping data to a higher-dimensional space?）

**非线性SVM：特征空间**  

- 基本思想：原始特征空间可以始终映射到某个更高维的特征空间，在该空间中训练集是可分的。（General idea: The original feature space can always be mapped to some higher-dimensional feature space where the training set is separable.）

**核技巧（Kernel Trick）**  

- 线性分类器依赖于向量间的内积 $$k(x_i, x_j) = x_i \cdot x_j$$。（The linear classifier relies on the inner product between vectors $$k(x_i, x_j) = x_i \cdot x_j$$.）  
- 如果通过某种变换 $$\phi$$ 将数据点映射到高维空间，则内积变为：$$k(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$$。（If every data point is mapped into a high-dimensional space via some transformation $$\phi$$, the inner product becomes $$k(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$$.）  
- 核函数是与高维空间内积等价的函数。例如，二维向量 $$x$$ 的核函数 $$K(x_i, x_j) = (1 + x_i^T x_j)^2$$。（A kernel function is a function that is equivalent to an inner product in some feature space. Example: $$K(x_i, x_j) = (1 + x_i^T x_j)^2$$ for 2D vectors.）

- 核函数隐式地将数据映射到高维空间，而无需显式计算 $$\phi(x)$$。（Thus, a kernel function implicitly maps data to a high-dimensional space without the need to compute each $$\phi(x)$$ explicitly.）

**什么样的函数可以是核函数？**  

- 对于某些函数 $$K(x_i, x_j)$$，检查 $$K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$$ 可能很麻烦。（For some functions $$K(x_i, x_j)$$, checking that $$K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$$ can be cumbersome.）  
- Mercer定理：每个半正定对称函数都是核函数。（Mercer's theorem: Every semi-positive definite symmetric function is a kernel.）  
- 半正定对称函数对应一个半正定对称Gram矩阵。（Semi-positive definite symmetric functions correspond to a semi-positive definite symmetric Gram matrix.）

**核函数的示例**  

- 线性核：$$K(x_i, x_j) = x_i^T x_j$$，映射为 $$\phi(x) = x$$。（Linear: $$K(x_i, x_j) = x_i^T x_j$$, mapping $$\phi(x) = x$$ itself.）  
- 多项式核：$$K(x_i, x_j) = (1 + x_i^T x_j)^p$$，映射为具有 $$p$$ 个维度的函数 $$\phi(x)$$。（Polynomial of power $$p$$: $$K(x_i, x_j) = (1 + x_i^T x_j)^p$$, mapping to $$\phi(x)$$ with $$p$$ dimensions.）  
- 高斯核（径向基核）：$$K(x_i, x_j) = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)$$，映射到无限维的函数空间。（Gaussian (radial-basis function): $$K(x_i, x_j) = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)$$, mapping $$\phi(x)$$ to an infinite-dimensional space.）

**非线性SVM的数学定义**  

- 对偶问题公式：  
  $$\max \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$  
  满足约束：  
  $$\sum_i \alpha_i y_i = 0, 0 \leq \alpha_i \leq C$$。（Dual problem formulation: $$\max \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$ subject to constraints $$\sum_i \alpha_i y_i = 0, 0 \leq \alpha_i \leq C.$$）  
- 解为：  
  $$f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$$。（The solution is: $$f(x) = \sum_i \alpha_i y_i K(x_i, x) + b.$$）

**SVM的应用**  

- SVM最早由Boser、Guyon和Vapnik在1992年提出，并在20世纪90年代末开始流行。（SVMs were originally proposed by Boser, Guyon, and Vapnik in 1992 and gained increasing popularity in the late 1990s.）  
- SVM目前是从文本到基因组数据等许多任务中表现最好的算法之一。（SVMs are currently among the best performers for a number of classification tasks ranging from text to genomic data.）  
- SVM可以通过设计适合该任务的核函数扩展到图、序列和其他任务。（SVMs can be applied to tasks beyond feature vectors (e.g., graphs, sequences, relational data) by designing kernel functions for such data.）  
- 常用的优化算法包括SMO和分解算法。（Most popular optimization algorithms for SVMs use decomposition methods, e.g., SMO.）  
- 调整SVM的核函数和参数通常通过尝试和测试完成。（Tuning SVM kernels and parameters is usually done in a try-and-see manner.）



## 5.3 Unsupervised Learning

### 5.3.1 Clustering

**无监督学习的类型**  

- **聚类（Clustering）**：一种通过将具有相似特征或特性的数据点分组到一起的技术。（Technique that involves grouping similar objects or data points together into clusters, based on certain features or characteristics.）  
- **降维（Dimensionality Reduction）**：一种将输出变量为实际值或定量值（如权重、预算等）的问题类型。（It involves a type of problem where the output variable is a real value or quantitative, such as weight, budget, etc.）

**聚类算法**  
无监督学习算法（Unsupervised Learning Algorithm）

**简介**  

- 聚类是一种机器学习技术，通过根据特定特征或属性将相似的对象或数据点分组到簇中。（Clustering is a machine learning technique that involves grouping similar objects or data points together into clusters, based on certain features or characteristics.）  
- 广泛应用于各行业，例如市场营销、医疗、金融等。（Important in various industries, such as marketing, healthcare, finance, and more.）  
- 聚类的主要目标是识别自然分组，最小化同一簇中数据点之间的距离或相似性，最大化不同簇中数据点之间的距离或不相似性。（The main goals of clustering are to identify natural groupings in data, minimize the distance or dissimilarity between data points within the same cluster, and maximize the distance or dissimilarity between data points in different clusters.）

**聚类的应用**  

- 市场营销（Marketing）  
- 医疗（Healthcare）  
- 金融（Finance）  
- 图像与语音识别（Image and speech recognition）  
- 异常检测（Anomaly detection）

**聚类的相似性和不相似性度量**  

- **欧几里得距离（Euclidean Distance）**：这是多维空间中两点之间的常见距离度量，计算公式为两点对应坐标差平方和的平方根。（This is a common measure of distance between two points in a multidimensional space. It is calculated as the square root of the sum of the squared differences between corresponding coordinates of the two points.）  
- **曼哈顿距离（Manhattan Distance）**：通过两点的对应坐标差的绝对值之和来测量距离，常用于图像识别或模式匹配。（This is another measure of distance between two points, based on the absolute differences between corresponding coordinates of the two points. It is often used in image recognition or pattern matching.）  
- **余弦相似度（Cosine Similarity）**：用于测量两个向量之间的相似性，基于它们夹角的余弦值，常用于自然语言处理或推荐系统。（This is a measure of similarity between two vectors, based on the cosine of the angle between them. It is often used in natural language processing or recommendation systems.）

**K均值聚类算法（K-Means Algorithm）**  

- K均值是一种广泛使用的聚类算法，通过将数据分成$$K$$个预定义的簇，每个簇由一个中心定义。（K-Means is a widely used algorithm for clustering that aims to partition a dataset into $$K$$ distinct clusters, where $$K$$ is a pre-specified number.）  
- 适用于分离良好的簇，并能处理多种类型的数据，例如数值或分类变量。（Effective for well-separated clusters and can handle different types of data, such as numerical or categorical variables.）  
- 需要预先指定簇的数量，这在某些情况下可能并不容易。（Requires the pre-specification of the number of clusters, which may not be known in advance or may be difficult to determine.）

**K均值算法步骤**  

1. **初始化**：从数据集中随机选择$$K$$个点作为初始中心点。（Initialization: Select $$K$$ random data points from the dataset as the initial centroids for the $$K$$ clusters.）  
2. **分配**：根据欧几里得距离或曼哈顿距离，将每个数据点分配到最近的中心点所属的簇中。（Assignment: Assign each data point to the cluster whose centroid is closest to it, based on a distance measure such as Euclidean distance or Manhattan distance.）  
3. **更新**：计算每个簇中新点的平均值，并将其作为新中心。（Update: Calculate the mean of each cluster as the mean of all the data points assigned to that cluster.）  
4. **收敛**：重复步骤2和3，直到中心点不再变化，或者达到最大迭代次数。（Convergence: Repeat steps 2 and 3 until the centroids converge, i.e., until the change in the centroids is below a certain threshold or the maximum number of iterations is reached.）

**K均值算法的初始化**  

- 初始化是K均值聚类算法的第一步，涉及为$$K$$个簇选择初始中心点。（The initialization step is the first step in the K-Means clustering algorithm, and it involves selecting the initial centroids for the $$K$$ clusters.）  
- 一种常见方法是随机选择$$K$$个点作为初始中心点。（One common method for initialization is random selection, where $$K$$ data points are randomly selected from the dataset to serve as the initial centroids.）  
- 例如：可以手动指定簇的数量，并随机挑选点作为中心。（Example: You can manually specify the number of clusters and randomly pick $$K$$ data points as centroids.）

**K均值算法 - 分配步骤**  (Assignment Step)

- 在分配步骤中，算法会将数据点分配到距离最近的簇中心。（In the assignment step, data points are assigned to the nearest cluster based on the distance to each cluster's centroid.）  
- 这里常用的距离度量是欧几里得距离，其计算公式为： 
  $$\text{Distance}(x_i, \mu_j) = \sqrt{\sum_{d=1}^{D}(x_{i,d} - \mu_{j,d})^2}$$ 
  其中，$$x_i$$ 是数据点，$$\mu_j$$ 是簇中心，$$D$$ 是特征的维度。（The common distance measure used is the Euclidean distance, given by $$\text{Distance}(x_i, \mu_j) = \sqrt{\sum_{d=1}^{D}(x_{i,d} - \mu_{j,d})^2},$$ where $$x_i$$ is the data point, $$\mu_j$$ is the centroid, and $$D$$ is the number of features.）  
- 分配步骤的目标是最小化每个簇内数据点到其中心的距离。（The goal of this step is to minimize the distance of each data point to its assigned cluster center.）

**K均值算法 - 更新步骤**  

- 更新步骤中，每个簇的中心会更新为当前簇中所有点的平均值。（In the update step, the centroid of each cluster is updated to the mean of all the data points assigned to that cluster.）  
- 更新公式为： 
  $$\mu_j = \frac{1}{n_j} \sum_{x \in C_j} x$$ 
  其中，$$\mu_j$$ 是更新后的簇中心，$$C_j$$ 是簇，$$n_j$$ 是属于簇 $$C_j$$ 的数据点数量。（The update formula is $$\mu_j = \frac{1}{n_j} \sum_{x \in C_j} x,$$ where $$\mu_j$$ is the updated centroid, $$C_j$$ is the cluster, and $$n_j$$ is the number of data points in cluster $$C_j$$.）  
- 更新步骤的目标是重新定位簇中心，以更好地反映当前簇内的数据分布。（The goal of this step is to reposition the centroids to better reflect the distribution of data points within each cluster.）

**K均值算法 - 收敛准则**  

- 收敛准则是K均值算法的最后一步，用于决定算法何时停止。（The convergence criteria is the final step in the K-Means clustering algorithm, and it determines when the algorithm should stop.）  
- 常见的收敛条件包括：  
  1. 中心点的位置不再发生显著变化。（The centroids' positions no longer change significantly.）  
  2. 达到预定义的最大迭代次数。（The maximum number of iterations is reached.）  
- 一旦满足收敛条件，算法终止，输出最终的簇分配。（Once the convergence condition is met, the algorithm stops, and the final cluster assignments are produced.）  

**收敛过程的可视化**  

- 收敛过程的图示展示了数据点的动态分配和簇中心的调整，直到最终分组稳定。（The visualization of the convergence process illustrates the dynamic reassignment of data points and adjustment of cluster centroids until the final groupings stabilize.）

<font color=blue>**聚类算法 - 层次聚类**  </font>

- 层次聚类是一种通过递归地分裂或合并簇，基于数据点之间的相似性创建簇树结构的算法。（Hierarchical clustering is a type of clustering algorithm that creates a tree-like structure of clusters by recursively dividing or merging clusters based on the similarity between data points.）  
- 两种主要类型：  
  - **凝聚式聚类（Agglomerative）**：从每个数据点为一个单独簇开始，逐步合并最相似的簇。（Start with each data point as its own cluster, then iteratively merge the closest clusters.）  
  - **分裂式聚类（Divisive）**：从一个大簇开始，递归地将其分裂为更小的簇。（Start with a single cluster containing all data points, then split it into smaller clusters recursively.）

**层次聚类策略**  

- **自底向上（Bottom-up，凝聚式）**：递归合并两个簇间距离最小的簇。（Recursively merge two groups with the smallest between-cluster dissimilarity.）  
- **自顶向下（Top-down，分裂式）**：递归地将簇分裂成两个子簇，直至每个叶子簇只包含一个对象。（Recursively split a cluster into two subclusters until each leaf cluster contains only one object.）

**凝聚式方法**  

- 初始化：每个对象为一个簇。（Initialization: Each object is a cluster.）  
- 迭代：合并两个最相似的簇，直到所有对象合并为一个簇。（Iteration: Merge two clusters that are most similar to each other until all objects are merged into a single cluster.）

**分裂式方法**  

- 初始化：所有对象开始时在同一个簇中。（Initialization: All objects stay in one cluster.）  
- 迭代：选择一个簇并将其分裂成两个子簇，直至每个簇只包含一个对象。（Iteration: Select a cluster and split it into two subclusters until each leaf cluster contains only one object.）

**层次切割**  

- 用户可以选择在层次图中切割以表示最自然的簇划分。（Users can choose a cut through the hierarchy to represent the most natural division into clusters.）  
- 示例：选择簇间不相似性超过某个阈值的切割点。（E.g., choose the cut where intergroup dissimilarity exceeds some threshold.）

**相似性测量方法**  

- 两个不相交簇 $$G$$ 和 $$H$$ 的不相似性 $$D(G, H)$$ 是基于成对数据点间的不相似性 $$D(i, j)$$ 计算的：  
  - **单链法（Single Linkage）**：簇间的最近距离。（Single Linkage: Minimum distance between clusters.） 
    $$D_{SL}(G, H) = \min_{i \in G, j \in H} D(i, j)$$  
  - **全链法（Complete Linkage）**：簇间的最远距离。（Complete Linkage: Maximum distance between clusters.） 
    $$D_{CL}(G, H) = \max_{i \in G, j \in H} D(i, j)$$  
  - **组平均法（Group Average）**：簇间平均距离，权衡单链法和全链法。（Group Average: Average distance between clusters.） 
    $$D_{GA}(G, H) = \frac{1}{|G| \cdot |H|} \sum_{i \in G, j \in H} D(i, j)$$

<font color=blue>**谱聚类（Spectral Clustering）**  </font>

- 将数据点表示为图 $$G$$ 的顶点。（Represent datapoints as the vertices $$V$$ of a graph $$G$$.）  
- 图的所有顶点通过边连接，边具有权重 $$W$$。（All pairs of vertices are connected by an edge, and edges have weights $$W$$.）  
- 边的权重反映相似性：大权重表示相邻顶点非常相似，小权重表示不相似。（Edge weights mean that adjacent vertices are very similar; small weights imply dissimilarity.）

**图分割（Graph Partitioning）**  

- 在图上进行聚类等价于对图的顶点进行分割。（Clustering on a graph is equivalent to partitioning the vertices of the graph.）  
- 损失函数用于划分顶点集 $$V$$ 为 $$A$$ 和 $$B$$ 两部分： 
  $$\text{cut}(A, B) = \sum_{i \in A, j \in B} W_{ij}$$ 
  其中 $$W_{ij}$$ 是边权重，较小的 $$\text{cut}(A, B)$$ 表示较好的分割。（The loss function for a partition of $$V$$ into sets $$A$$ and $$B$$ is $$\text{cut}(A, B) = \sum_{i \in A, j \in B} W_{ij}$$. A good partition has small $$\text{cut}(A, B)$$.）  
- 最小划分标准（Min-cut criterion）：寻找划分 $$A$$ 和 $$B$$，使 $$\text{cut}(A, B)$$ 最小。（Min-cut criterion: Find partition $$A, B$$ that minimizes $$\text{cut}(A, B)$$.）  

**归一化划分标准**  

- 最小划分可能忽略了子图的大小平衡问题。（Min-cut criterion ignores the size of the subgraphs formed.）  
- 归一化划分标准（Normalized cut criterion）： 
  $$\text{Ncut}(A, B) = \frac{\text{cut}(A, B)}{\text{assoc}(A, V)} + \frac{\text{cut}(A, B)}{\text{assoc}(B, V)}$$ 
  其中，$$\text{assoc}(A, V) = \sum_{i \in A, j \in V} W_{ij}$$。（Normalized cut criterion favors balanced partitions.）  
- 最小化归一化划分标准是一个NP难问题。（Minimizing the normalized cut criterion exactly is NP-hard.）

**谱聚类（Spectral Clustering）**  

- 在谱聚类中，数据点被视为图的节点。（In spectral clustering, the data points are treated as nodes of a graph.）  
- 谱聚类步骤：  
  1. **构建相似图（Build a similarity graph）**  
  2. **将数据投影到低维空间（Project the data onto a low-dimensional space）**  
  3. **创建簇（Create clusters）**

**谱聚类 - 第一步：构建相似图**  

- 构建一个无向图 $$G = (V, E)$$，顶点集为 $$V = \{v_1, v_2, ..., v_n\}$$，表示数据中的 $$n$$ 个观测点。（We first create an undirected graph $$G = (V, E)$$ with vertex set $$V = \{v_1, v_2, ..., v_n\}$$, $$n$$ observations in the data.）  
- 相似性可以通过邻接矩阵表示：  
  - $$k$$-邻近图（KNN graph）  
  - 完全连接图（Fully connected graph）  
  - 高斯核函数：$$s(x_i, x_j) = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)$$（Gaussian kernel: $$s(x_i, x_j) = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right).$$）

**谱聚类 - 第二步：低维投影**  

- 计算图拉普拉斯矩阵：$$L = D - A$$ 
  其中，$$A$$ 是邻接矩阵，$$D$$ 是度矩阵： 
  $$D_{ii} = \sum_{j} W_{ij}$$。（Compute the graph Laplacian: $$L = D - A,$$ where $$A$$ is the adjacency matrix, and $$D$$ is the degree matrix.）  
- 通过计算 $$L$$ 的特征值和特征向量，将数据投影到低维空间。（Project the data onto a low-dimensional space by computing eigenvalues and eigenvectors of $$L$$.）  
- 拉普拉斯矩阵的谱范围为 $$0 = \lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n$$。（The spectrum of the Laplacian is $$0 = \lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n.$$）

**谱聚类 - 第三步：创建簇**  

- 使用第二小的特征值对应的特征向量进行分割。（Use the eigenvector corresponding to the 2nd smallest eigenvalue to split the nodes.）  
- 例如，根据特征向量值的正负将节点分为两类。（For example, split nodes into two clusters based on the sign of the values in the eigenvector.）  
- 多簇问题可以通过使用多个特征向量进行扩展。（For multi-cluster problems, use multiple eigenvectors.）



### 5.3.2 Dimensionality Reduction
