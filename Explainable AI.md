[TOC]



# 1 入门

**1.传统算法和机器学习算法比较：**

- **传统算法：** 基于手写规则的程序，如能获得优惠计划的客户分类
- **机器学习算法：** 无需简单规则，基于多个输入数据自动学习，如贷款风险评估

**2.机器学习的基本成分与模型训练**

- **函数类:** 定义输出 (y) 与输入 (x) 的关系

- **训练数据:** 包含特征和标签的数据样本

**3.现代机器学习的黑盒问题**

- **黑盒模型：** 尽管模型的准确性与商业收入、用户体验密切相关，但缺乏透明度

- **透明性的重要性：** 需要识别关键因素并生成科学假设

**4.临床预测的关键问题：Why**

- **挑战：** 精确预测临床结果很重要，但核心问题在于理解预测背后的原因

**5.解释性AI在生物学和健康中的应用**

- **解释性的重要性：** 解释性有时与准确性同等重要，例如在临床应用中，了解特征对预测的影响至关重要

- **关键问题：**

  - 哪些特征对某一预测起到了关键作用？

  - 如何选择或学习最具解释性或信息量的特征？

  - 如何使黑盒模型具有生物学或临床意义？

**6.黑盒模型的透明性目标**

- 目标包括：
  - 识别潜在过程中的关键因素
  - 生成科学假设
  - 模型故障诊断和审查
  - 审核不必要的依赖项
  - 改善数据集和提高信任

**7.解释性AI的关键成分**

- **模型：** 可以是黑盒模型 (如DNN, GBM)
- **数据：** 单个数据样本或整个数据集
- **问题：** 需要了解什么方面？

**8.XAI 问题类型**

- **特征重要性**：哪些特征对预测最重要？
- **概念重要性**：模型理解的关键概念是什么？
- **数据实例重要性**：每个样本对模型训练的贡献程度

**9.临床预测中的准确性与解释性**

- 尽管准确预测临床结果很重要，但关键问题在于“为什么”模型做出这样的预测。
- **精确性与可解释性：** 简单模型（如线性模型）通常解释性强，但性能差；复杂模型（如黑盒模型）尽管准确，但难以解释。
- **SHAP方法：** Scott Lundberg 和 Su-In Lee提出了SHAP方法，用于任何模型的特征重要性计算，能解释单一预测的依据。
- 临床预测中的解释性提高了决策支持：使用XAI提高手术期间低氧血症的预测准确性，提供实时解释，帮助麻醉师识别潜在的低氧风险。

**10.解释性AI促进模型审核和成本意识**

- **模型审核的必要性：** 许多基于AI的COVID-19检测系统依赖于“捷径”特征而非真正的病理学依据，表明模型审核的重要性。

- **成本意识AI (`Cost-Aware AI`,CoAI)：** CoAI方法显著减少特征获取成本（如时间），可应用于急诊和重症监护。

  

# 2 什么是XAI

**1.解释性AI (XAI) 的定义**

- **狭义定义：** 让模型决策对人类更易理解的方法和技术
- **广义定义：** 让所有AI相关内容（如数据、功能、性能等）更易理解
- **重点：** 主要关注监督式机器学习的解释
- **权衡问题(Trade-Off Issue)：** 高性能模型（如深度神经网络）往往难以解释，直接可解释模型性能相对较低。
- **XAI重要性：** 解释性是负责任AI的基础，有助于实现公平性、可用性、安全性等目标。
  - **解释性调试：** XAI 有助于识别模型中的问题并通过调试改进模型。
  - Kulesza 等人提出的解释性调试方法，可以通过用户反馈不断优化模型

**2.监督式机器学习**

- **学习模型：** 使用算法训练模型，以预测新实例的标签。
- **特征解释：** XAI的重点是解释模型的决策过程，例如苹果蛋糕的颜色、形状和气味等特征。
- **模型解释：** 包括性能、限制、输出等。

**3.XAI类型**

- **直接解释模型**：如线性模型、决策树、基于规则的模型
- **事后解释模型**：针对黑盒模型，如深度神经网络，可用近似方法提供解释。

**案例：贷款申请审批中的决策支持**

- **参与角色：** 数据科学家、贷款官员、客户
- **需求：** 客户希望了解贷款审批的理由，贷款官员需要对预测结果有信心。

**4.模型的全局与局部解释**

- **Knowledge Distillation**：

  - 知识蒸馏是一种方法，通常用来将复杂模型（如深度神经网络）的知识“蒸馏”到一个更简单的模型中。

  - 在这里，知识蒸馏被用来将黑盒模型的行为通过近似方法转移到一个更易解释的模型上，例如决策树或线性模型，以便更好地理解原始模型的整体行为。

- **全局解释：** 如决策树近似，帮助数据科学家理解整体模型的决策逻辑。
- **局部解释：** 分析单个预测，如局部特征贡献和原型示例，帮助贷款官员做出判断。

**5.事后解释算法示例**

**LIME方法：** LIME 是一种用于解释黑盒模型（如深度神经网络和随机森林）的局部解释方法。它通过构建一个 **易解释的、线性的近似模型** 来模拟黑盒模型在特定输入附近的行为，进而揭示该输入如何影响模型的预测。

<font color=blue>**步骤：**</font>

**生成扰动数据**：

- LIME 生成与输入样本相似的扰动数据集（例如，通过小幅改变特征值），并将这些数据输入到原始黑盒模型中，记录模型的输出。

**权重分配**：

- 根据扰动数据点与原始输入样本的相似性，LIME 为这些数据点分配权重，确保越接近原始样本的扰动点在解释中权重越高。

**训练局部线性模型**：

- 使用带权重的扰动数据，LIME 训练一个简单的线性模型来近似原始模型在该局部区域的行为。

**生成解释**：

- LIME 的最终输出是每个特征对该预测结果的贡献度（正或负），以揭示哪些特征在多大程度上影响了该预测。

**6.反事实检查**

- **反事实特征检查：** 如部分依赖图和对比特征，展示不同决策条件下的特征影响。
- **反事实示例：** 分析类似实例，如其他按时还款的客户，帮助理解贷款被拒的原因。

**7.XAI评价指标**

- **固有指标：** 如保真度、稳定性、简洁性
- **用户依赖指标：** 如可理解性、用户满意度
- **任务导向指标：** 任务表现和用户信任度

**8.不想要的偏差(unwanted bias)**

- **偏差问题：** 偏差可能会导致某些非特权群体处于系统性不利地位，在某些背景下甚至是非法的。
  - **背景：** COMPAS 系统用于预测犯罪率，但被发现对某些群体存在歧视性偏见。

**XAI审查**：

- **作用：** XAI 可以作为一种工具，用于分析和揭示模型中的歧视性偏见。

- **方法：** 通过对比性示例和特征重要性分析帮助用户理解模型决策

**9.XAI优点**

- **重要性：** XAI 提供的解释有助于用户理解决策背后的原因，从而采取合适的行动。
- **信任校准：** XAI 可以帮助校准人们对 AI 的信任，增加用户对系统的依赖度。
- **AI 文档和治理的趋势：** AI 的文件记录和问责机制变得越来越重要，像 IBM 的 FactSheets 和 Google 的 Model Cards 等工具已经被提出。
  - **目标：** 这些工具有助于在 AI 系统中实现透明性和可追溯性，保障 AI 的安全性和公正性

# 3 解释方法

## 3.1 特征重要性解释(Feature Importance Explanations)

**目的**：解释每个特征如何影响模型，并回答模型决策相关的问题。

**例子**：

- 包括自然图像、医疗图像、表格数据和时间序列等场景。
- 研究例子：Fong 和 Vedaldi 的研究“通过有意义的扰动解释黑盒模型”
  - 通过对图像进行有意义的扰动（例如遮盖或模糊特定区域），观察模型的预测变化。该方法旨在发现哪些图像区域对于模型预测最重要。
- 研究例子：Sundararajan 等人提出的“深度网络的公理化归因”
  - 提出了一种基于深度神经网络的归因方法，称为“积分梯度法”（Integrated Gradients），它利用数学公理来确定每个像素对最终预测的贡献。这种方法在不改变模型结构的情况下生成解释性信息。
- 研究例子：Lundberg 等人关于“树的解释性 AI：从局部解释到全局理解”
  - 提出了 SHAP（Shapley Additive Explanations）方法，基于博弈论中的 Shapley 值来计算每个特征在特定预测中的边际贡献。该方法可以为树模型提供局部和全局的解释。
- 研究例子：Covert 等人关于“用于自动癫痫检测的时间图卷积网络”（2019年）
  - 提出了时间图卷积网络（Temporal Graph Convolutional Networks），在时间序列中检测出癫痫发作等重要事件，并解释模型如何识别时间序列中的关键模式。

### 3.1.1 符号说明

常用的符号：

1. **𝑥**：输入数据或特征向量

   这是模型接受的输入，表示一个具体的数据实例。𝑥 包含了多个特征的值。

2. **𝑦**：模型的输出或预测结果

   𝑦 是模型对输入𝑥的预测输出，例如在分类问题中，𝑦 可能是一个类别标签。

3. **𝑓(𝑥)**：机器学习模型

   表示机器学习模型，它根据输入𝑥 生成输出𝑦。𝑓(𝑥) 的结构和算法因模型类型而异，如神经网络、决策树等。

4. **𝑥₁, 𝑥₂, ..., 𝑥𝑑**：特征（Features）

   每个输入𝑥包含𝑑个特征（feature），即𝑥是一个包含𝑑个维度的向量。例如，𝑥₁可以代表年龄，𝑥₂可以代表收入。

5. **𝑑**：特征的总数

   表示输入数据中所包含的特征数量。

6. **𝑛**：样本数

   数据集中数据点的总数，即样本数。𝑛个数据点表示模型接受了多少个独立的输入数据实例进行训练或测试。

### 3.1.2 定义

**模型解释（Model Explanation）**

模型解释的目标是突出模型为什么会做出某个预测。通过揭示模型在做出特定决策时的原因，帮助用户理解模型的行为。

**特征重要性解释（Feature Importance Explanation）**

- 特征重要性解释专注于每个特征的作用，展示各个特征对模型预测的影响大小。
- 这种解释可以是针对单个预测的（局部解释，Local）或针对模型整体行为的（全局解释，Global）。

**局部解释与全局解释**

- **局部解释（Local）**：局部解释关注单个数据实例的特征重要性，展示某个特定预测的原因。
- **全局解释（Global）**：全局解释旨在描述模型整体的行为，展示模型在所有数据点上的特征重要性。

**特征重要性解释的两种类型**

- **特征归因（Feature Attribution）**：对每个特征 $$ x_i $$ 赋予一个分数 $$ a_i $$，这个分数表示该特征对模型预测的贡献。
- **特征选择（Feature Selection）**：从所有特征中选择一个重要特征子集$$ x_S \subset \{x_1, x_2, \dots, x_d\} $$，这些特征在预测中起到关键作用。

**解释算法（Explanation Algorithm）**

解释算法是一种基于输入数据和机器学习模型生成解释的方法。它的目的是使模型的预测过程对用户更加透明。



## 3.2 基于移除的解释(Removal-based Explanations)

- **基本思想**：移除特征并观察预测变化，判断特征重要性。

- **医生类比与机器学习转化**
  - 通过遮挡医疗图像的部分区域，观察诊断变化，类似于 ML 中移除特征。

### 3.2.1 案例研究

#### 3.2.1.1 置换测试(permutation test)

置换测试是一种用于评估输入特征重要性的“老方法”，最早在随机森林模型中应用。通过该方法，可以确定每个输入特征的总体（全局）重要性。

- **步骤 1**：首先使用原始数据评估模型的准确性。
- **步骤 2**：逐个特征进行破坏，即随机打乱数据集中特定特征列的值（对应于该特征）。
- **步骤 3**：记录模型准确性的下降幅度，以此来衡量特征的重要性。

该方法通过对数据集中的每个特征进行随机扰动，观察其对预测精度的影响，从而评估各个特征的重要性。这就是置换测试的基本原理。

- 置换测试适用于任何模型
- 可用于连续特征或类别特征
- 快速且易于实现

**数学定义**

(1)直观的数学定义
$$
a_i = Acc(\text{original}) - Acc(x_i \text{ corrupted})
$$

- **$$ a_i $$**：特征 $$ x_i $$ 的重要性分数，表示当特征 $$ x_i $$ 被破坏或置换时，模型准确性的下降程度。
- **$$ Acc(\text{original}) $$**：模型在使用完整原始数据时的准确性。
- **$$ Acc(x_i \text{ corrupted}) $$**：当特征 $$ x_i $$ 被随机打乱或破坏后，模型的准确性。

(2)详细视角的数学定义
$$
a_i = \frac{1}{n} \sum_{j=1}^{n} \ell(f(x_1^j, ..., \tilde{x}_i^j, ..., x_d^j), y^j) - \frac{1}{n} \sum_{j=1}^{n} \ell(f(x_1^j, ..., x_i^j, ..., x_d^j), y^j)
$$

- **$$ a_i $$**：特征 $$ x_i $$ 的重要性分数，衡量当特征 $$ x_i $$ 被扰乱后，对模型损失的平均影响。
- **$$ n $$**：样本数量。
- **$$ f(x_1^j, ..., x_d^j) $$**：模型 $$ f $$ 的输出，基于输入特征 $$ x_1^j, ..., x_d^j $$。
- **$$ \ell $$**：任意损失函数（例如均方误差或交叉熵），用于衡量模型预测值和真实值 $$ y^j $$ 之间的误差。
- **$$ \tilde{x}_i^j $$**：特征 $$ x_i $$ 在第 $$ j $$ 个样本中的扰乱版本（例如被随机打乱的值）。
- **第一部分** $$ \frac{1}{n} \sum_{j=1}^{n} \ell(f(x_1^j, ..., \tilde{x}_i^j, ..., x_d^j), y^j) $$：表示破坏特征 $$ x_i $$ 后的平均损失。
- **第二部分** $$ \frac{1}{n} \sum_{j=1}^{n} \ell(f(x_1^j, ..., x_i^j, ..., x_d^j), y^j) $$：表示原始数据下的平均损失（即特征未被破坏的情况）。



#### 3.2.1.2 遮挡(Occlusion)

遮挡方法是一种早期用于深度学习模型的解释技术，主要用于图像分类任务中的个别预测解释。

- **遮挡方法适用于任何模型**：不仅限于图像数据，还可用于非图像数据的模型解释。
- **速度适中**：对每个预测进行解释时，需要进行 d+1d + 1d+1 次模型评估（其中 ddd 为特征数）。
- **简单易实现**：遮挡方法实现简便，适合各种应用场景。

**方法概述**

- **深度学习的早期方法**：遮挡方法在深度学习刚兴起时被提出，作为理解复杂图像分类模型的技术。
- **用于解释单个预测**：主要关注于解释模型如何对单个图像做出预测，帮助理解模型为何对特定输入图像做出特定预测。
- **计算像素（或超像素）的重要性**：通过逐步遮挡图像中的某些像素或超像素区域，观察模型预测结果的变化，从而计算这些区域对预测结果的重要性。

遮挡方法的具体过程如下：

1. **使用完整图像进行预测**：首先，模型在未遮挡的完整图像上进行预测，得到基线预测结果。
2. **遮挡图像的不同区域**：将图像的不同区域逐一遮挡，并记录每次遮挡后预测结果的变化。这个过程帮助识别出哪些区域对模型预测最为关键。
3. **用无信息（如零）像素替换遮挡区域**：遮挡通常通过将像素替换为无信息的值（如零）来实现，这样可以保证该区域对预测不再有任何贡献。
4. **遮挡的粒度**：遮挡操作可以在不同的粒度上进行，例如 2x2 或 4x4 的超像素（superpixel）块。较小的遮挡块可以获得更精细的解释，而较大的遮挡块则有助于快速识别重要区域。

**数学定义**

(1)直观视角的数学定义
$$
a_i = f_y(x) - f_y(x_{(-i)})
$$

- **$$ a_i $$**：特征 $$ a_i $$ 的重要性分数，通过计算完整输入 $$ x $$ 的模型输出 $$ f_y(x) $$ 与去掉第 $$ i $$ 个特征后的模型输出 $$ f_y(x_{(-i)}) $$ 之间的差值来得出。
- **目的**：通过遮挡特定特征，观察模型输出的变化，从而量化该特征对模型预测的重要性。

(2)详细视角的数学定义
$$
a_i = f_y(x_1, ..., x_d) - f_y(x_1, ..., 0, ..., x_d)
$$

- **$$ a_i $$**：特征 $$ a_i $$ 的重要性分数，计算为完整输入 $$ (x_1, ..., x_d) $$ 的模型输出与将第 $$ i $$ 个特征替换为零后的模型输出之间的差值。
- **替换为零**：在详细视角中，遮挡操作通过将特征 $$ x_i $$ 替换为零（或其他无信息的值）实现，使其对模型预测不再产生影响。

#### 3.2.1.3 置换和遮挡方式对比

- **为不同模型设计**：置换适用于随机森林，遮挡适用于卷积神经网络（CNN）

- **全局与局部解释**：置换适用于全局解释，遮挡适用于局部解释。

- **相似之处**：尽管方法不同，但这些方法之间存在一些显著的相似之处。

|              | 置换测试         | 遮挡           |
| ------------ | ---------------- | -------------- |
| 破坏输入     | 随机化特征       | 设为零         |
| 观察模型变化 | 观察准确性的变化 | 观察预测的变化 |
| 计算影响     | 移除单个特征     | 移除单个特征   |

### 3.2.2 基于移除解释方法的统一框架

想法：通过更改实现选项来创建新的解释方法。

**三种设计选择**：**特征移除方法**、**模型行为**和**汇总技术**

<font color = blue>**特征移除**</font>

- **核心思想**：模型通常需要所有特征才能进行预测，但我们希望从某些特征中移除信息。
- **特征移除的模拟**：大多数模型不支持直接移除特征，因此需要模拟特征移除。
- **实现方式**：
  - **使用默认值（如零）替换**：将要移除的特征替换为零值。
  - **使用随机值替换**：用随机值替换特征内容。
  - **为每个特征集训练单独的模型**：构建包含不同特征的独立模型。
  - **使用支持缺失特征的模型**：选择可以接受缺失特征的模型。
  - **模糊处理（针对图像）**：对图像中的区域进行模糊处理，模拟特征移除。

<font color = blue>**模型行为**</font>

- **观察模型行为**：可以移除特征并观察其对模型的影响。

- 选择观察的量

  需要选择一个具体的量来评估模型行为，例如预测值、预测损失或数据集损失。

  - **Prediction（预测）**：直接观察预测结果。
  - **Prediction loss（预测损失）**：观察预测结果的损失值。
  - **Dataset loss（数据集损失）**：整体数据集上的损失值 $$ E[\ell (y, \hat{y})] $$。

<font color = blue>**汇总技术**</font>

- **选择汇总方法**：可以使用任意特征子集来观察模型行为。
- **组合过多问题**：给定 d 个特征，有 $$ 2^d $$ 个子集需要考虑。如何有效地传达信息是一个挑战。
- **常见的汇总类型**：
  - **Feature selection（特征选择）**：选取一组重要特征的子集。
  - **Feature attribution（特征归因）**：分配特征分数，例如使用置换测试和遮挡法。

----------

<font color=blue>**常用特征移除方法**</font>

- **PredDiff**：通过条件期望生成解释和交互，使用条件填充模型来删除信息
  - 条件删除模型会删除某些特征，但通过填充与邻近值或上下文相关的内容，尽可能保持数据的“合理性”或“自然性”。
- **Meaningful Perturbations**：考虑多种从输入图像中删除信息的方法，推荐的操作是模糊化

<font color=blue>**常用汇总方法**</font>

**(1)RISE**：对缺失特征的多个子集进行采样，计算包含 $$ x_i $$ 时的平均预测

**(2)LIME**：对特征子集应用加权核 $$ \pi(S) $$，拟合线性/加性代理模型
$$
\min_{a_0, ..., a_d} \sum_{S \subseteq D} \pi(S) \left( a_0 + \sum_{i \in S} a_i - f_y(x_S) \right)^2 + \Omega(a_1, ..., a_d)
$$

- **目标**：LIME 通过拟合一个 **加性近似模型**，来解释复杂模型的预测。LIME 的目标是找到一组系数 $$ a_0, a_1, ..., a_d $$，使得这些系数尽可能地描述目标模型在局部区域的行为。

- **加性近似（Additive Approximation）**：公式的核心部分 $$ a_0 + \sum_{i \in S} a_i $$ 表示一个线性模型的近似，其中 $$ a_0 $$ 是偏置项，$$ a_i $$ 是每个特征的贡献分数。

- **权重函数 $$ \pi(S) $$**：用于衡量不同特征子集的重要性，通常根据特征子集与被解释数据点的相似性进行加权，确保越接近原始实例的特征组合在解释中权重越高。

- **正则化项 $$ \Omega(a_1, ..., a_d) $$**：这是一个可选的正则化项（如 Lasso），用于控制系数的稀疏性，帮助简化模型，使解释更易于理解。

**工作流程部分**

1. **模型输入和预测**：

   复杂模型（例如神经网络）接受数据输入并生成预测结果。这里的数据包括了多个特征，例如打喷嚏、体重、头痛等，模型预测疾病为“流感”（Flu）。

2. **解释器（LIME）**：

   LIME 解释器接收原始数据和模型预测，选择一组与预测相关的重要特征，通过线性近似来解释模型的局部行为。LIME 提取了特征“打喷嚏”、“头痛”和“无疲劳”作为解释特征，用来简化复杂模型的输出。

3. **人类决策**：

   最终，解释器生成的解释（如“打喷嚏”、“头痛”、“无疲劳”）被展示给用户，人类基于这些解释做出进一步的判断或决策。

### 3.2.3 Meaningful perturbations(扩展学习)

- **目的**：从模型正确分类的图像开始，通过对图像进行模糊处理来改变预测结果。

- **过程**：通过模糊操作逐渐改变图像，使得模型的预测结果发生变化。
- **示例**：对比模糊图像和原始图像，展示了模糊如何影响模型的分类准确性。

<font color = blue>**步骤**</font>

- 设 $$ x \in \mathbb{R}^{w \times h} $$ 为图像
- 设 $$ m \in [0, 1]^{w \times h} $$ 为遮罩
- 设 $$ \Phi(x, m) $$ 为遮罩图像

**(1)问题**：如何进行遮罩操作？

- **使用常量值替换**：

$$
\Phi(x, m)_{ij} = m_{ij} \cdot x_{ij} + (1 - m_{ij}) \cdot \mu
$$

- **使用噪声替换**：

$$
\Phi(x, m)_{ij} = m_{ij} \cdot x_{ij} + (1 - m_{ij}) \cdot \epsilon_{ij}
$$

其中 $$ \epsilon \sim \mathcal{N}(0, \sigma^2) $$

- **使用高斯核模糊**：

$$
\Phi(x, m)_{ij} = \text{blur with kernel } g_{\sigma}
$$

**(2)学习最优模糊**

- **初始状态**：目标类别 $$ y $$ 对 $$ \Phi(x, 1) $$ 的预测概率接近 1

- **目标**：学习一个遮罩 $$ m $$，使得 $$ f_y(\Phi(x, m)) \approx 0 $$

- **最小化以下损失**：

  $$
  \min_m f_y(\Phi(x, m))
  $$

**(3)其他考虑因素**

1. 模糊应尽量最小
2. 遮罩应平滑。
3. 优化应对抗对抗性扰动。

**实际损失函数**
$$
\min_m \mathbb{E}_{x} [f_y(\Phi(x, m))] + \lambda_1 \|1 - m\|_1 + \lambda_2 \| \nabla m \|_2^2
$$

- 损失项
  - $$ f_y(\Phi(x, m)) $$：预测误差项
  - $$ \|1 - m\|_1 $$：稀疏遮罩正则化
  - $$ \| \nabla m \|_2^2 $$：平滑正则化

**(4)优化**

- 实际损失函数：
  $$
  L(m) = \mathbb{E}_{x}[f_y(\Phi(x, m))] + \lambda_1 \|1 - m\|_1 + \lambda_2 \| \nabla m \|_2^2
  $$

- 通过随机梯度下降（SGD）确定最优遮罩：

  $$
  m^{(t+1)} = m^{(t)} - \alpha \frac{\partial L}{\partial m^{(t)}}
  $$

**(5)结果**

- **不同模糊处理效果的比较**：展示了在模糊、常量替换和噪声扰动下的遮罩效果。
- **图像示例**：从不同的遮罩和扰动类型中可以看出，模糊处理能够有效地影响模型对目标的注意区域





## 3.3 Shapley值

### 3.3.1 背景

**SHAP值的定义**

Shapley 值是一种用于解释机器学习模型的方法，通过量化每个特征对预测结果的贡献，确保每个特征的影响被公平评估。这种方法基于合作博弈理论中的 Shapley 值，解决了复杂模型中特征贡献分配的问题。

**SHAP值的计算过程**

- 特征团队：模型的输入由多个特征构成，例如年龄、性别、BMI等。这些特征的组合影响最终的预测结果。
- 边际贡献：SHAP值通过计算每个特征在不同特征组合中的边际贡献，评估该特征对预测的平均影响。这个过程涉及计算特征在所有可能组合下的增量贡献。
- 公平分配：通过考虑所有可能的特征组合，SHAP值确保每个特征的贡献得到公平分配，避免单个特征被高估或低估。

**SHAP值的影响因素**

- **特征组合**：特征的边际贡献因组合的不同而变化。SHAP值计算时会综合所有可能的组合，以确保评估的全面性。
- **特征之间的关系**：相关性高的特征会相互影响彼此的SHAP值，例如两个高度相关的特征可能会分摊各自的贡献。
- **模型类型**：不同类型的模型（如决策树、神经网络等）对特征贡献的计算方式不同，从而影响SHAP值的结果。
- **特征重要性**：特定模型中更重要的特征通常会在SHAP值中获得更高的边际贡献分配。

**SHAP值的实际应用**

- **模型解释**：SHAP值帮助解释机器学习模型的决策过程，揭示每个特征对预测结果的影响程度，使得模型的预测结果更加透明。
- **奖金或资源分配**：在图示案例中，SHAP值用于公平地分配奖金或资源。不同特征的贡献量化后，可以用于制定奖励分配方案，确保分配的公平性。
- **领域专家的作用**：在一些应用场景中，领域专家的专业知识也可以纳入SHAP分析，以提供更加合理的解释和分配。

### 3.3.2 Shapley值计算

Shapley 值的计算公式如下：

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} \left[ v(S \cup \{i\}) - v(S) \right]
$$

其中：

- $$S$$ 为特征的子集；
- $$|S|$$ 为子集的大小；
- $$n$$ 为总特征数；
- $$v(S)$$ 表示特征子集 $$S$$ 对模型输出的贡献；
- $$\phi_i$$ 表示特征 $$i$$ 的 Shapley 值。

**边际贡献的计算**

Shapley 值通过计算边际贡献来衡量每个特征的影响：

- 边际贡献定义为 $$v(S \cup \{i\}) - v(S)$$，表示在子集 $$S$$ 的基础上加入特征 $$i$$ 后的贡献增量。
- 通过计算特征在不同组合中的边际贡献，评估每个特征对模型预测的具体影响。

**权重分配**

为了确保 Shapley 值的公平性，需要对每个子集的边际贡献进行加权，公式如下：

$$
\frac{|S|!(n - |S| - 1)!}{n!}
$$

这个权重确保了每种组合在计算过程中被公平地考虑。

**Shapley 值计算示例**

通过以下步骤计算每个特征的 Shapley 值：

- 计算特征在所有可能组合中的边际贡献。
- 加权平均每个特征的边际贡献，得出 Shapley 值。

例如，一个特征的边际贡献可能是 $$70\%$$、$$10\%$$、$$60\%$$，将这些值加权平均后，即为该特征的 Shapley 值。

**特征组合的排列和计算**

Shapley 值计算过程中涉及特征组合的排列，以确保计算的全面性：

- 当特征数量为 $$n = 4$$ 时，所有可能的子集组合数为 $$2^4 = 16$$。
- 每个特征的 Shapley 值通过在所有组合中的边际贡献加权求和得到。

**计算总结**

完整的 Shapley 值计算步骤为：

1. 遍历所有可能的特征组合。
2. 计算每个特征在不同组合中的边际贡献。
3. 对边际贡献加权，得到每个特征的 Shapley 值。
4. 将 Shapley 值应用于模型解释，以公平分配每个特征的贡献。

这种计算方法确保了每个特征贡献的公平分配，避免某一特征在模型中被高估或低估。

**总结**

- Shapley 值：提供了一种公平的方法来衡量每个特征对模型预测的贡献。
- 边际贡献：通过对特征子集的增量贡献进行计算，反映了特征在不同组合下的影响。
- 加权平均：确保所有组合的贡献被公平考虑，避免单一特征的影响被夸大或忽略。
- 排列组合：帮助遍历所有可能的特征组合，使得 Shapley 值的计算更加全面和准确。

### 3.3.3 Shapley values in XAI

**应用于机器学习**

- 将特征视为“玩家”（players）。
- 将模型行为视为“利润”（profit），例如预测结果、损失等。
- 使用 Shapley 值来量化每个特征的影响。

**SHAP**

- SHAP 是 Shapley Additive exPlanations 的缩写。
- SHAP 值在机器学习中的使用得到了推广，尤其在解释模型预测上非常有帮助。
- SHAP 使用 Shapley 值来解释个体预测结果。

**SHAP 作为基于移除的解释方法**

- 回顾基于移除解释的三个选择：
  1. **特征移除**：$$F(x_S) = E_{x_{\bar{S}}|x_S}[f(x_S, x_{\bar{S}})]$$
  2. **模型行为**：$$v(S) = F_y(x_S)$$
  3. **总结**：$$a_i = \phi_i(v)$$，其中 $$\phi_i(v)$$ 为 Shapley 值。

- 基于移除的方法通过逐步去除特征并计算其对模型输出的影响，来解释特征对预测结果的贡献。



## 3.4 Propagation-based explanations

### 3.4.1 Layerwise Relevance Propagation (LRP)

**Layer-wise Relevance Propagation (LRP)** 展示了逐层相关性传播方法的工作原理，用于解释深度学习模型的预测。LRP 是一种解释机器学习模型预测的方法，特别适用于深度神经网络，通过逐层传播相关性来识别哪些输入特征对最终的分类结果影响最大。以下是对图中内容的总结：

**输入图像**：图的左侧展示了一个输入图像，在示例中可能是一张包含动物或其他对象的图片。

**模型预测**：输入图像通过深度神经网络进行处理，模型输出对不同类别的预测概率。

**相关性传播过程**：LRP 从神经网络的输出层开始，将相关性分数逐层反向传播回去，一直到输入层。每一层中的每个神经元（或特征）都会根据其对输出的贡献获得一个相关性分数。

**逐层传播计算**：每一层的相关性分数会根据不同的规则进行分配，通常使用以下公式：
$$
R_j = \sum_i \frac{a_j w_{ji}}{\sum_j a_j w_{ji}} R_i
$$
其中，$$ R_j $$ 表示第 $$ j $$ 个神经元的相关性分数，$$ w_{ji} $$ 表示连接权重，$$ a_j $$ 表示第 $$ j $$ 个神经元的激活值。LRP 的目的是在各层之间传播相关性分数，最终在输入层生成一个相关性热图，展示对预测最有贡献的像素或特征。

**生成解释（相关性热图）**：通过逐层相关性传播，最终在输入图像上生成一个 **热图**，显示哪些区域对模型的预测结果贡献最大。热图中的红色区域表示对模型预测有正贡献的区域，蓝色区域表示负贡献区域。这样用户可以看到模型关注的图像部分，从而更好地理解预测背后的逻辑。

**主要优势**：直观可解释性：LRP 生成的热图使得用户可以直观地理解模型关注的区域。层级贡献分析：通过逐层传播，可以看到每层网络对最终结果的贡献，使得深度网络的决策过程更加透明。适用于深度神经网络：特别适合解释复杂的深度学习模型。

**应用场景**：LRP 常用于图像分类、对象识别等深度学习应用中，以帮助用户理解模型的预测依据。

---

- **直观性欠佳**：相比其他方法，LRP 的直观性较弱，并且需要一些启发式选择，例如选择哪种“规则”来分配相关性，这可能会让解释结果更加复杂。
- **适应性较差**：LRP 在不同的神经网络架构上可能难以适用，特别是在有不同结构的模型中。
- **示例**：例如，LRP 并不自动支持残差连接（如 ResNet 架构），在应用到变压器（transformer）模型时也需要进行扩展和修改。

<font color=blue>**例子：Layerwise Relevance Propagation (LRP) on MRI Data**</font>

图像展示了 LRP 应用于 MRI （脑肿瘤图像）脑部扫描图像，以解释深度学习模型如何识别不同类型的脑肿瘤。示例中的 MRI 图像包括 **胶质瘤**（Glioma）和 **脑膜瘤**（Meningioma）两种肿瘤类型。LRP 生成的热图展示了模型对不同肿瘤类型关注的区域，帮助解释模型的决策依据。

**数据集来源** 
数据集存储在 GitHub 仓库中，链接展示了脑肿瘤分类数据集的位置。数据集包含 MRI 图像，分为训练集和测试集，用于训练和评估分类模型。

**项目文件结构** 
展示了数据集的文件夹结构，包括 **glioma_tumor**、**meningioma_tumor** 等类别的文件夹，每个文件夹中包含对应类别的 MRI 图像。该结构有助于理解数据的组织方式，以便更方便地加载数据并用于模型训练。

**代码实现** 
代码主要展示了 LRP 方法的实现，代码文件位于 GitHub 中的 $$xai-series/05\_lrp.py$$。代码包含导入库、数据预处理、模型加载及 LRP 解释方法的实现。这些代码帮助将 LRP 应用到 MRI 图像分类任务中，以生成可解释性结果。

**代码运行界面** 
最后展示了在 IDE 中运行代码的界面，可以看到 LRP 解释生成的热图以及模型输出的结果。热图展示了模型在图像上关注的区域，使得用户可以直观地理解模型如何进行分类。

### 3.4.2 Gradient-based explanations

**Application to XAI** 

- **Idea**：找到在扰动时导致输出变化较大的特征。 
- **Remark**：该方法量化特征敏感性，但不一定与特征移除相关。

**Vanilla Gradients** 
对于输入 $$ x $$ 和标签 $$ y $$，计算预测 $$ f_y(x) $$ 的梯度：
$$
a_i = \frac{\partial f_y}{\partial x_i}(x)
$$
可以选择使用绝对值：
$$
a_i = \left|\frac{\partial f_y}{\partial x_i}(x)\right|
$$

**Variant 1: SmoothGrad** 
计算输入附近的梯度平均值。例如，添加高斯噪声：
$$
a_i = \mathbb{E}_{\epsilon}\left[\frac{\partial f_y}{\partial x_i}(x + \epsilon)\right] \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$
实际中，使用少量采样（50次），并需调节 $$ \sigma $$ 到适当水平。

**Variant 2: Integrated Gradient** 
梯度可能饱和，即使对重要输入也产生小梯度。模型对大输入变化敏感，但对小变化不敏感。 

- **Idea**：通过计算重缩放图像的梯度来解决饱和问题：

$$
x'(\alpha) = \bar{x} + \alpha(x - \bar{x}) \quad \text{for } 0 \leq \alpha \leq 1
$$

在重缩放图像范围内积分（平均）梯度：
$$
a_i = (x_i - \bar{x}_i) \int_0^1 \frac{\partial f_y(x'(\alpha))}{\partial x_i} d\alpha
$$

**GradCAM** 
在卷积神经网络（CNN）中，隐藏层表示高层视觉概念，隐藏层保留了因卷积结构而得的空间信息。 

- **Idea**：通过最后的卷积层而非输入层来解释模型。

**GradCAM Results** 
图示展示了 GradCAM 的结果。不同颜色表示模型关注的区域，从而直观显示模型决策时关注的图像部分。


### 3.4.3 Propagation vs. removal-based explanations

**Many explanation methods**  

- **Removal-based explanations**: 包括 SHAP、LIME、RISE、Occlusion、Permutation Tests。  
- **Propagation-based explanations**: 包括 SmoothGrad、IntGrad、GradCAM。  
- **实践选择**: 该使用哪种方法取决于特定需求和模型类型。

---

**Model flexibility**  

- **你在解释哪种模型？**  
  - **Removal-based explanations** 是模型无关的，可以应用于各种模型（如 CNNs、Trees 等）。  
  - **Propagation-based explanations** 主要用于神经网络，通常需要计算导数，有时还对模型架构有特定要求。

---

**Data flexibility**  

- **你有哪种类型的数据？**  
  - **Removal-based explanations** 可以处理离散和连续特征数据，例如，适用于数据集中具有不同值的缺失特征。  
  - **Propagation-based explanations** 更适合连续特征，能很好地捕捉输入特征的微小变化。

---

**Local or global**  

- **你需要哪种类型的解释？**  
  - 两种方法都可以生成局部解释。  
  - **Removal-based methods** 更适合全局解释，关注整体模型行为（如特征重要性）。  
  - 对于 **Propagation-based methods**，如果需要全局解释，则需将局部解释汇总。

---

**Speed**  

- **速度是否重要？**  
  - **Propagation-based methods** 更快，因为只需反向传播一次，与特征数量依赖性弱。  
  - **Removal-based methods** 通常较慢，因为需多次进行预测，尤其是 SHAP 等方法。

---

**Quality**  

- **哪个解释最具信息性或正确性？**  
  - 理论上可以作为指导，但也可以采取经验方法来衡量解释质量。  
  - **Perspective**: 无解释是完美的，但一些方法可能与用户问题不完全匹配。

---

**Popular methods**  

- **哪些方法最受欢迎？**  
  - 只有少数几种方法占主导地位，取决于数据领域（表格、图像、自然语言处理等）。

---

**Tabular data**  

- **Permutation tests** 广泛用于全局特征重要性。  
- **SHAP** 在局部解释中很流行，例如 **TreeSHAP** 内置于 XGBoost 和 LGBM 中，**KernelSHAP** 则用于其他模型。

---

**Computer vision**  

- **GradCAM** 和 **IntGrad** 在视觉领域最流行。  
- **Removal-based methods** 通常较慢。  
- 一些论文在尝试改进，但尚未流行。

---

**NLP**  

- 自然语言处理模型（如 LSTMs、Transformers）可以使用大多数解释方法。  
- **Gradient-based methods** 较流行，**Removal-based explanations** 较慢，但偶尔会用留一法（Occlusion）。  
- 对于 Transformers，可使用注意力作为解释。

---

**Popular packages** 
列出了一些流行的解释包，例如 **shap**、**lime**、**captum**、**innvestigate** 等，并展示了其 GitHub 星标数，说明其受欢迎程度。


# 4 Evaluating explanations

**现在，放眼全局**  

- 各种算法尽管多样，但都旨在实现一个目的：**识别有影响力的特征**  
- 我们如何测试哪些方法最有效地做到这一点？  

**需要考虑的问题**  

- 我们是否需要事先知道什么是重要的？  
- 解释应当反映模型认为重要的内容，还是人类认为重要的内容？  
- 我们的性能指标是否与特定的解释方法对齐？

**设置**  

- 假设一个模型 $$f(x)$$  
  - 分类器，以类别 $$y$$ 的概率 $$f_y(x)$$  
- 假设一个解释算法  
  - 局部解释（例如：RISE）  
  - 全局解释（例如：置换测试）  
  - 返回每个特征 $$x_i$$ 的分数 $$a_i \in \mathbb{R}$$  

## 4.1.1 Sanity checks

**合理性检查**  

- 合理性检查 = 用于识别明显问题的基本测试（Sanity check = basic test to identify obvious issues）  
  - 例如，用一个小列表测试排序算法，或用一些添加/删除操作测试数据结构（e.g., test a sorting algorithm with a small list, or a data structure with a few addition/deletion operations）  
- 对于解释算法，哪些合理性检查是好的？（What are good sanity checks for an explanation algorithm?）

**XAI的合理性检查**  

- 解释是否具有定性意义？（Does the explanation make qualitative sense?）  
- 它是否依赖于数据？（Does it depend on the data?）  
- 它是否依赖于模型？（Does it depend on the model?）

**定性评估**  

- 通过不同梯度方法比较解释的合理性，方法包括：Vanilla、Integrated、Guided BackProp和SmoothGrad，其中SmoothGrad表现得更合理（Comparison of explanations through various gradient methods, including Vanilla, Integrated, Guided BackProp, and SmoothGrad, with SmoothGrad appearing more reasonable）  

**数据依赖性**  

- 解释对数据有明确的依赖性。不同的输入图像在预测和解释图像之间存在显著差异，表明模型在输出解释时高度依赖输入数据（The explanation clearly depends on the data. Different input images show significant variation between the predicted and explained images, indicating a strong data dependency in the model's explanation）  

**模型依赖性**  

- 解释对模型依赖性明显。不同的模型结构（如VGG16和ResNet-18）在Grad-CAM解释结果上显示显著差异，表明模型架构对解释影响较大（The explanation clearly depends on the model. Different model architectures, such as VGG16 and ResNet-18, show significant differences in Grad-CAM explanation results, highlighting the influence of model architecture on explanations）  

**随机化测试**  

- 对先前检查的扩展，通过对模型或数据进行随机化后比较解释。解释应发生显著变化，但有些方法变化不明显（Scaled-up version of previous checks by comparing explanations after model or data randomization. Explanations should change significantly, but some methods show minimal change）  

**模型随机化**  

- 使用深度神经网络（Inception-v3架构），通过在特定层中随机化参数来评估解释的可靠性（Using a deep neural network (Inception-v3 architecture), parameters are randomized in specific layers to assess the robustness of explanations）  
  - 从最终层开始逐步随机化，依次应用到更早的层，这一过程称为“级联随机化”（Starting from the final layer and progressively applying randomization to earlier layers, termed "cascading randomization"）  

- 结果：随着层的逐步随机化，不同解释方法（如Grad-CAM、Integrated Gradients等）对模型解释的影响发生显著变化。这些方法在不同的随机化程度下展示了不同的表现，揭示了模型结构在解释质量中的作用（Results: As layers are randomized step-by-step, various explanation methods (e.g., Grad-CAM, Integrated Gradients) show significant changes in their effectiveness. Performance varies across levels of randomization, highlighting the role of model structure in explanation quality）

**数据随机化**  

- 思路：通过随机标签重新训练模型，将标签随机分配，从而迫使新模型使用不同的信号（Idea: retrain the model with randomized labels to ensure the new model relies on different signals by assigning labels randomly）  
  - 在随机标签训练的模型中，解释图显示与原始模型不相关的特征，这验证了数据随机化对解释的显著影响（In models trained on randomized labels, explanation maps show uncorrelated features compared to the original model, confirming a significant impact of data randomization on explanations）  

**备注** 

- 优点：  
  - 合理性检查简单，能够排除不合理的方法（Sanity checks are simple, effective in ruling out flawed methods）  
  - 可以作为深入研究前的初步步骤（Useful as a preliminary step before more in-depth analysis）  
- 缺点：  
  - 检查通常不是定量的（Checks are often not quantitative）  
  - 很少涉及解释的准确性（Provides limited insight into the correctness of explanations）  


## 4.1.2 Ground truth comparisons

**真实重要性的对比**  

- 假设对“真正重要”的特征有先验知识（Assume prior knowledge of "truly important" features）  
- 先验知识的来源包括：医生对医学图像的标注、非专业人士对自然图像的标注、生物学文献中已知与疾病相关的基因（Sources of prior knowledge include doctor annotations of medical images, non-expert annotations of natural images, and genes known to play a role in diseases from biology literature）  
- 然后，将解释结果与真实值进行对比，以评估其准确性（Then, compare explanations to ground truth to assess accuracy）

**对象定位**  

- 通过显著性图生成边界框，并与真实边界框对比，重叠区域超过阈值则计为正确定位（Generate a bounding box from the saliency map and compare it to the ground truth bounding box, counting as correct if the overlap exceeds a set threshold）  
- 生成边界框过程复杂，可能显著影响结果：  
  - 一种简单方法是使用阈值显著性（如50%分位数）找到包含显著特征的最小边界框（A simple approach is to use threshold saliency (e.g., at 50% quantile) to find the smallest bounding box containing salient features）  
  - Simonyan等人（2013）提出的改进方法通过显著特征比例来区分对象和背景，从而实现更好的分割（Simonyan et al. (2013) proposed an improved method by using salient feature ratios to distinguish object and background, achieving clearer segmentation）  
- 尽管模型未针对定位进行专门训练（弱监督），许多方法仍能实现较低的定位错误率（Many methods achieve low localization error rates despite the models not being explicitly trained for localization ("weakly supervised")）

**指点游戏**  

- 一种更简单的定位任务，无需生成边界框，直接检查解释中最重要的像素是否落在真实边界框内（A simpler localization task that avoids generating bounding boxes by directly checking if the most important pixel in the explanation falls within the ground truth bounding box）  

**放射学中的定位**  

- 使用多个显著性方法对胸部X光片进行定位，通过专家评估模型在识别病变区域的准确性，以辅助诊断（Using multiple saliency methods to localize chest X-rays, with expert evaluation of the model's accuracy in identifying lesion areas to aid diagnosis）

**用户研究**  

- 通过多种方法生成解释，并让用户（通常在Mechanical Turk上）选择最优方法（Generate explanations using multiple methods and let users (often on Mechanical Turk) decide which is best）  
  - 不同研究关注不同问题，如哪个解释更好、解释是否能够正确指示类别等（Different studies focus on questions like which explanation is better or whether the explanation accurately indicates the class）

**合成数据集**  

- 使用合成生成的数据可以控制真实值，确保模型和解释评估的可靠性（Synthetically generated data allows control over the ground truth, ensuring reliability in model and explanation evaluation）

**真实值的挑战**  

- 先验知识来源于人类，获取额外标注困难，且反映的是当前对世界的理解，可能会惩罚使用新信号的模型（Prior knowledge comes from humans, making it hard to obtain extra annotations, and reflects current world understanding, potentially penalizing models that use new signals）  
  - 这些知识不一定来自专家，Mechanical Turk用户的标注可信度较低，医生的标注通常更可信（Not always derived from experts; Mechanical Turk users are less reliable, while doctor annotations are more trustworthy）

**联合测试模型和解释**  

- 为获得最佳结果，需满足两点：（For best results, two conditions are required:）  
  1. 解释需正确识别模型的依赖关系（Explanations must correctly identify the model's dependencies）  
  2. 模型应依赖“正确”信号，不应使用捷径或混杂因素（如图像背景）（The model must depend on the "correct" signals and avoid using shortcuts or confounders (e.g., image background)）  
- 问题：差的结果可能是模型导致的，而真实值指标并未直接测试解释（Problem: poor results may be due to the model itself, as ground truth metrics don’t directly test the explanation）

**数学视角**  

- 在分类问题中，设 $$p(y | x)$$ 为真实的条件概率，假设输入为 $$x$$、标签为 $$y$$，我们可以检查特征 $$x_S$$ 是否满足以下条件（Consider a classification problem where $$p(y | x)$$ is the true conditional probability. Given input $$x$$ and label $$y$$, we can check if feature $$x_S$$ meets the conditions below）  
  - 理想情况下，“真正重要”的特征 $$x_S$$ 应满足：  
    - $$p(y | x_S) \approx 1$$（必要条件）（Necessary condition: $$p(y | x_S) \approx 1$$）  
    - $$p(y | x_S) \approx 0$$（充分条件）（Sufficient condition: $$p(y | x_S) \approx 0$$）  

- 假设 $$f_y(x) = p(y | x)$$，这是模型训练的隐含目标（Assume $$f_y(x) = p(y | x)$$, which is the implicit goal of model training）  
- 然后，假设我们可以通过条件分布对特征进行边缘化处理：  
  $$\mathbb{E}_{x_{\bar{S}}|x_S}[f_y(x)] = p(y | x_S)$$  
  （Then, assume we can marginalize out features with their conditional distribution:  
  $$\mathbb{E}_{x_{\bar{S}}|x_S}[f_y(x)] = p(y | x_S)$$）  
- 这表明我们可以使用基于去除的方法来识别正确的特征 $$x_S$$（This suggests that we can use removal-based methods to identify the correct features $$x_S$$）

**备注**  

- 优点：  
  - 真实值指标在某些使用案例中反映了XAI的目标，即识别数据中的真实关系（Ground truth metrics reflect the goal of XAI in some use cases: identifying true relationships in the data）  
- 缺点：  
  - 获取真实值困难且不完美（Obtaining ground truth is difficult and imperfect）  
  - 为了获得好的结果，需要正确的解释和正确的模型（For good results, a correct explanation and a correct model are necessary）  

## 4.2.1 Ablation metrics

**消融度量**  

- 假设我们可以使用保留特征来评估模型，重要性值暗示预测应如何变化（Assume we can evaluate models with held-out features, and importance values suggest how the prediction should change）  
  - 删除重要特征应显著改变预测结果（Removing important features should significantly change the prediction）  
- 思路：测试解释是否可以通过保留特征预测行为（Idea: test if explanations predict behavior with held-out features）

**插入/删除**  

- 按重要性 $$a_i$$ 对特征 $$x_i$$ 进行排序（Rank features $$x_i$$ by importance $$a_i$$）  
  - 插入：按顺序添加特征，预测值应迅速上升（Insertion: add features in order of importance; prediction should go up quickly）  
  - 删除：按顺序移除特征，预测值应迅速下降（Deletion: remove features in order of importance; prediction should drop quickly）  

**多种可能的变化**  

- 衡量不同的模型行为，例如预测概率、低概率、对数几率、准确性等（Measure different model behaviors, such as prediction probability, low-probability, log-odds, accuracy）  
- 以不同方式移除特征（例如用0、随机噪声、数据集中的样本值），不同的移除方式会显著影响结果（Remove features in different ways (e.g., zeros, random noise, values sampled from the dataset), which can make a big difference in results）

**特征选择的变化**  

- 可以应用相同的思想来评估全局解释，通过移除重要或不重要特征进行模型再训练，观察准确性变化（Apply the same idea to evaluate global explanations by retraining models with the most or least important features and observing accuracy changes）  

**移除并重训练（ROAR）**  

- 模型通常无法处理缺失特征，思路是用重要特征掩码并重新训练模型，测试准确性是否下降（Models aren't made to handle missing features, so retrain with masked important features and test if accuracy drops）  
- ROAR问题：  
  - 重新训练多个模型成本高，不测试原模型解释的正确性（Retraining many models is costly and does not test the correctness of explanations for the original model）  
  - 掩码训练易引入混杂因素，可能导致准确性虚高（Training with masks encourages confounders, yielding inflated accuracy）  
  - 存在信息泄漏问题，掩码非随机，移除特征可能暗示类别标签（Information leakage issue; masking isn't random, and removed features may indicate the class label）

**局限性**  

- 主要关注重要性排序，但缺乏针对分数 $$a_i \in \mathbb{R}$$ 的精确测试方法（Focuses on importance rankings but lacks a precise way to test the importance scores $$a_i \in \mathbb{R}$$）

**加性代理度量**  

- 许多方法的得分是和为模型预测的加性代理（例如IntGrad, LRP），可以测试这些分数作为加性代理的准确性（Many methods have scores that sum as additive proxies for the model prediction (e.g., IntGrad, LRP); test the accuracy of importance scores as additive proxies）

**Sensitivity-n**  

- 测试代理与固定基数的随机子集的相关性，通过在集合 $$S \subset (1, ..., d)$$ 中计算相关性（Test the proxy’s correlation for random subsets with fixed cardinality, using correlation over $$S \subset (1, ..., d)$$ with $$|S| = n$$）  
  - 公式为：$$\text{Corr} \left( f_y(x_S), \sum_{i \in S} a_i \right)$$  

**可变基数版本**  

- 计算相同的相关性，但使用不同基数的子集（Calculate the same correlation but with subsets of different cardinalities）  
  - 需要对所有基数分布 $$p(S)$$，或者在所有基数上使用均匀分布（Require a distribution $$p(S)$$ over all cardinalities or use a uniform distribution over all $$S \subset \{1, ..., d\}$$）  
  - 公式为：$$\text{Corr} \left( f_y(x_S), \sum_{i \in S} a_i \right)$$

**相关度量**  

- 插入/删除和Sensitivity-n等方法可用于衡量解释质量（Metrics like insertion/deletion and Sensitivity-n are used to assess explanation quality）  
  - 示例包括Samek等人（2015）和Lundberg等人（2020）在插入/删除方法方面的工作，以及Alvarez-Melis等人（2018）在Sensitivity-n方面的研究（Examples include works by Samek et al. (2015) and Lundberg et al. (2020) on insertion/deletion and Alvarez-Melis et al. (2018) on Sensitivity-n）

**特征移除的选择**  

- 消融度量与基于移除的解释相似，提出相同的移除特征问题（Ablation metrics mirror removal-based explanations, posing the same question of how to remove features）  
  - 重新训练会偏离原始模型，而随机替换值选择何种分布也存在挑战（Retraining diverges from the original model, and using random values poses challenges regarding distribution choice）  
  - 边缘化处理具有条件依赖性，可实现部分输入的最优预测，但难以实现（Marginalizing with conditional dependency provides best-effort predictions with partial input but is hard to implement）

- 度量的特征移除方式倾向于支持类似的解释方法。例如，使用插入/删除与零掩码时，SHAP使用零掩码优于SHAP的边缘分布（Feature removal choices in a metric favor similar explanations. For instance, using insertion/deletion with zeros masking makes SHAP with zeros outperform SHAP with marginal distribution）  

**备注**  

- 优点：  
  - 消融度量测试解释是否符合模型的正确性，而不是人类的关注点（Ablation metrics test an explanation’s correctness for the model, rather than what’s important to humans）  
  - 不需要额外的数据标注（No extra data annotation required）  
- 缺点：  
  - 如何移除特征的选择较难（Difficult choice of how to remove features）  
  - 在某些情况下，不关注原始模型（例如ROAR）（In some cases, not focused on the original model, like ROAR）  

## 4.2.2 Other criteria

**鲁棒性**  

- 对抗样本：一些不可察觉的微小变化可能会影响预测结果，这种现象在XAI中也被探索过（Adversarial examples: imperceptible changes that affect the prediction, explored in XAI）  
- 解释是否对数据中的小变化具有鲁棒性？（Are explanations robust to small changes in the data?）  
- 解释是否对模型中的小变化具有鲁棒性？（Are explanations robust to small changes in the model?）

- 通过不同的解释方法（如Grad、IntGrad、LRP）测试对篡改模型的解释，展示了对抗性攻击对解释稳定性的影响（Testing explanations for a manipulated model with various methods (e.g., Grad, IntGrad, LRP), showing the impact of adversarial attacks on explanation stability）

**超参数敏感性**  

- 许多方法具有超参数选择，例如样本数量（LIME）、基线/移除方法（IntGrad）、超像素大小（遮挡）（Many methods have hyperparameter choices, such as number of samples (LIME), baseline/removal approach (IntGrad), superpixel size (occlusion)）  
- 当一个参数对结果影响大且没有明确的“正确”选择时，会产生问题（Problematic when a parameter has a large impact on results and lacks a clear "right" choice）

**人类效用**  

- 解释的用途如何？需要明确使用场景（How useful is an explanation? Must specify the use-case）  
  - 人类与AI团队合作场景，例如校准对模型决策的信心（Human-AI team setting, e.g., calibrating confidence in model decisions）  
  - 科学研究场景，例如识别生物学假设以便后续验证，但难以大规模测试（Scientific setting, e.g., identifying biological hypotheses that are later verified, difficult to test at scale）  



## 4.3 Conclusion

**总结**  

- 合理性检查：失败是不允许的，但许多方法会通过（Sanity checks: Failing these is not okay, but many methods will pass）  
- 真实值比较：额外标注工作繁重，测试模型和解释的正确性，可能反映或不反映预期的用途（Ground truth comparisons: Extra annotations can be laborious; tests both model and explanation correctness, which may or may not reflect intended usage）  
- 消融：测试解释对模型的正确性最好的选择，有多个优质度量如插入/删除、Sensitivity-n，但特征保留的选择较难（Ablations: Best option to test an explanation’s correctness for the model, several good metrics (insertion/deletion, sensitivity-n), tricky choice of how to hold out features）

**何时使用这些度量？**  

- 主要用于开发新方法时，证明其有效性并展示其优于先前方法的优点（Mainly when developing a new method to prove that it works and show benefits over prior methods）  
- 还可用于新模型/数据集的选择，以验证实现选择的正确性（Additionally, when deciding what to use with a new model/dataset, to verify implementation choices）

**观点**  

- 没有错误的方法，但一些方法可能与用户问题不一致（No method is wrong, but some are misaligned with user questions）  
  - 度量有效地将用户问题形式化，也可设计用于其他用户目标的度量（Metrics effectively formalize user questions; can design metrics for other user objectives as needed）  



# 5 Inherently interpretable models

## 5.1 Introduction

**Post-hoc explanations**

- **过程**: 从世界获取数据，训练模型，提取解释，并将其传递给人类。(Process: Capture data from the world, train models, extract explanations, and deliver them to humans.)
- **特点**: 通过后续分析对黑箱模型进行解释，以提高人类对模型输出的理解。(Post-hoc explanations analyze black-box models to improve human understanding of outputs.)

---

**Inherently interpretable models**

- **过程**: 使用可解释性模型直接输出透明的预测，使人类可以理解。(Process: Use inherently interpretable models to provide transparent outputs that humans can understand.)
- **特点**: 模型设计本身确保其可解释性。(The design ensures explainability.)

---

**Defining interpretability**

- **含义**: 模型的“可解释性”可能有三种含义：(Interpretability may refer to:)
  1. **可模拟性 (Simulatability)**：人类能否直观理解模型。(Can humans intuitively understand the model?)
  2. **可分解性 (Decomposability)**：模型的每个部分是否具有直观意义。(Does each component have an intuitive meaning?)
  3. **算法透明性 (Algorithmic Transparency)**：能否证明算法行为的某些性质。(Can we prove algorithmic properties?)

---

**Simulatability**

- **描述**: 人类是否可以合理模拟模型的行为。(Description: Can humans reasonably simulate the model's behavior?)
- **限制**:
  - 定义因人而异，可能与领域相关。(Definitions are subjective and domain-specific.)
  - 例如，无法心算50层ResNet，但可以理解简单线性模型。(For example, simulating a 50-layer ResNet is unreasonable, but a simple linear model is manageable.)

---

**Decomposability**

- **问题**: 模型的每个组成部分是否有直观作用？(Question: Does each model component have an intuitive role?)
- **示例**:
  - 决策树中的每个分裂基于单一特征及其阈值。(Each split in a decision tree relies on a single feature and threshold.)
  - 线性模型系数表示特征与结果间的关系强度。(Linear model coefficients represent the association strength between features and outcomes.)

---

**Algorithmic transparency**

- **问题**: 能否证明学习算法的性质？(Question: Can we prove properties of learning algorithms?)
- **应用**:
  - 对线性模型已有许多理论研究。(Extensive research exists for linear models.)
  - 深度模型的挑战：(Challenges for deep models include:)
    1. 训练后模型的收敛行为。(Post-training convergence behavior.)
    2. SGD对数据分布的影响及公平性问题。(The impact of SGD on data distribution fairness.)

---

**Why post-hoc explanations?**

- **解释**: 在复杂性与可解释性之间权衡。(Explanation: Balances complexity and interpretability.)
- **图示**: 模型的准确性越高，可解释性可能越低，反之亦然。(Diagram: Higher accuracy often reduces interpretability, and vice versa.)

---

**Is this tradeoff real?**

- **现状**:
  - 结构化数据（图像、文本、音频）：神经网络最优。(Neural networks excel with structured data (images, text, audio).)
  - 表格数据：简单模型（线性回归、逻辑回归）也表现良好。(For tabular data, simpler models like linear/logistic regression perform well.)
- **问题**: 复杂模型在简单场景中收益有限。(Problem: Complex models offer limited benefits in simple cases.)

---

**Why this tradeoff?**

- **原因**:
  - 可解释模型受约束，缺乏灵活性。(Interpretable models are constrained and inflexible.)
  - 受限模型难以表示复杂关系。(Constrained models struggle to represent complex relationships.)
  - 挑战领域（如计算机视觉、自然语言处理）中，可解释模型表现不佳。(In challenging fields like CV or NLP, interpretable models underperform.)

---

**Examples**

- **模型约束**: 模型可能需要满足线性性、可加性、单调性、因果性等。(Models may be constrained to satisfy linearity, additivity, monotonicity, causality, etc.)
- **常见示例**:
  - 线性模型 (Linear models): 满足线性性。(Linearity.)
  - 广义加性模型 (GAMs): 限制特征交互。(Limited feature interactions.)
  - 决策树 (Decision trees): 二元特征分裂。(Binary feature splits.)

---

**Caveats**

- **简单模型的局限性**:
  - 如果使用人工设计特征，可能不可分解。(If they use engineered features, decomposability may be affected.)
  - 如果特征过多，可能难以模拟。(If they use too many features, simulatability may decrease.)

- **模型的复杂问题**:
  - 模型使用的高级概念可能不直观。(What higher-level concepts does the model use?)
  - 哪些训练样本对模型影响最大也可能不易明确。(Which training samples influenced the model most?)



## 5.2 Linear regression

**Linear regression**

- **线性预测函数**: $$f(x) = \beta_0 + \beta_1x_1 + \cdots + \beta_dx_d$$  
  (Linear prediction function: $$f(x) = \beta_0 + \beta_1x_1 + \cdots + \beta_dx_d$$)
- **训练方法**: 通过最小化均方误差(MSE)进行训练：  
  $$\mathcal{L}(\beta) = \sum_{i=1}^n \left(y^i - f(x^i)\right)^2$$  
  (Trained by minimizing MSE: $$\mathcal{L}(\beta) = \sum_{i=1}^n \left(y^i - f(x^i)\right)^2$$)

---

**Interpreting a linear model**

- **解释方法**: 通过学习的权重$$\beta$$及其置信区间解释。(Can interpret via learned weights $$\beta$$ and their confidence intervals.)
- **作用**:
  - 定量化特征重要性。(Quantify feature importance.)
  - 模拟新输入的预测。(Mentally simulate prediction with new inputs.)
  - 理解小变化的影响。(Understand the impact of small changes.)

---

**Lasso regression**

- **修改方式**: 寻找最小特征集。(Modified approach: find minimal feature set.)
- **损失函数**: 正则化损失函数：  
  $$\mathcal{L}(\beta) = \frac{1}{n} \sum_{i=1}^n \left(y^i - f(x^i)\right)^2 + \lambda \sum_{j=1}^d |\beta_j|$$  
  (Minimize a regularized loss function: $$\mathcal{L}(\beta) = \frac{1}{n} \sum_{i=1}^n \left(y^i - f(x^i)\right)^2 + \lambda \sum_{j=1}^d |\beta_j|$$)
- **特点**: 鼓励模型将部分权重$$\beta_j$$置零，生成稀疏解。(Encourage model to set some weights $$\beta_j$$ to zero, producing a sparse solution.)

---

**Ridge regression**

- **修改方式**: 通过岭惩罚进行正则化：  
  $$\mathcal{L}(\beta) = \frac{1}{n} \sum_{i=1}^n \left(y^i - f(x^i)\right)^2 + \lambda \sum_{j=1}^d \beta_j^2$$  
  (Regularize with ridge penalty: $$\mathcal{L}(\beta) = \frac{1}{n} \sum_{i=1}^n \left(y^i - f(x^i)\right)^2 + \lambda \sum_{j=1}^d \beta_j^2$$)
- **特点**: 保留所有特征的相关性，但不会鼓励权重完全为零。(Useful properties but does not encourage weights to be exactly zero.)

---

**Remarks**

- **优点**:
  - 线性模型易于解释，可心算模拟。(Linear models are easy to interpret and mentally simulate.)
  - 广泛使用，训练快速。(Widely used and fast to train.)
- **缺点**:
  - 限制多，某些任务中的预测性能较差。(Highly constrained, worse predictive performance for some tasks.)
  - 当特征相关性较高时，解释变得困难。(Interpretation is challenging with correlated features.)



## 5.3 Generalized additive models (GAMs)

**GAMs**

- **定义**: 广义加性模型(GAM)结合非线性单特征模型(形状函数)：  
  $$f(x) = f_1(x_1) + \cdots + f_d(x_d)$$  
  (Generalized additive models (GAMs) combine non-linear single-feature models (shape functions): $$f(x) = f_1(x_1) + \cdots + f_d(x_d)$$)
- **常见形状函数**:
  - 样条函数 (Splines)
  - 树 (Trees)
  - 线性函数 (Linear function = linear regression)

---

**Example result**

- **描述**: 关联混凝土强度与年龄和成分。(Relating concrete strength to age and ingredients.)
- **分析**:
  - 样条函数揭示了水泥的线性关系。(Splines uncover linear relationships with cement.)
  - 非线性函数揭示了水和空气的非线性关系。(Non-linear relationships with water and age.)

---

**More shape functions**

- **新增选项**:
  - 分段线性曲线。(Piecewise linear curves.)
  - 深度模型。(Deep models.)
- **描述**:
  - 分段线性曲线使模型更加易读。(Piecewise linear curves improve human readability.)
  - 深度模型可解释的广义加性模型。(Deep models for interpretable GAMs.)

---

**GA²Ms**

- **定义**: GAMs扩展为包含交互项：(Definition: GAMs extended with interaction terms:)  
  $$f(x) = \sum f_i(x_i) + \sum f_{ij}(x_i, x_j)$$
- **作用**: 通过排名交互强度选择交互项并进行优化。(Rank interaction strength to decide which interactions to include.)

---

**Interactions boost accuracy**

- **结果**: 学习排序数据集预测网站相关性。(Learning-to-rank dataset predicts website relevance.)
- **分析**: 树状GA²M捕获了交互效果。(Interaction effects captured by tree-based GA²Ms.)

---

**Remarks**

- **优点**:
  - 比线性模型更灵活，涵盖更多模型。(More flexible than linear models, covers a wide range of models.)
  - 可直接编辑模型参数，如政策场景中的再犯预测。(E.g., edit model parameters for recidivism prediction in policy scenarios.)
- **缺点**:
  - 忽略了更高阶的交互。(Ignores higher-order interactions.)
  - 适合有限特征交互的场景。(Best for scenarios with limited feature interactions.)



## 5.4 Decision trees

**Decision trees**

- **特点**:
  - 简单的二元分裂。(Simple binary splits.)
  - 内部节点: 基于单一特征和阈值进行划分。(Internal nodes partition on single features and thresholds.)
  - 叶节点: 对样本进行预测。(Leaf nodes predict outcomes for samples.)
- **优点**: 相对容易模拟。(Relatively easy to simulate.)
- **缺点**: 分裂过多时可能变得复杂。(Can become difficult with more splits.)

---

**Decision/rule lists**

- **定义**: 决策树的简化分支结构。(Decision trees with simplified branching structures.)
  - 每个内部节点至少有一个叶节点。(For each internal node, at least one child must be a leaf.)
  - 类似扩展的"if-elseif-else"规则。(Like extended "if-elseif-else" rules.)
- **决策列表**: 是受限的决策树，更易解释。(Decision lists are constrained decision trees, easier to interpret.)

---

**Example**

- **目标**: 预测用户是否喜欢职业体育。(Goal: Predict if a user "likes professional sports.")
- **结构**:
  - 使用年龄和其他条件分支，逐步作出判断。(Split based on age and other conditions step by step.)
  - 决策列表的结构比复杂决策树更简单。(Simpler structure compared to complex decision trees.)

---

**CORELS**

- **定义**: 可验证的最优规则列表。(Certifiably optimal rule lists.)
  - 使用正则化经验风险优化特定类别模型。(Optimal for specific models based on regularized empirical risk.)
  - 通过分支与界算法生成最优决策列表。(Branch and bound algorithm produces optimal decision lists.)
- **优点**: 在灵活模型中桥接准确性差距。(Helps bridge accuracy gaps with more flexible models.)

---

**Recidivism prediction**

- **应用**: 基于条件结合进行再犯预测。(Predict recidivism using a conjunction of conditions.)
- **示例规则**: 条件组合如年龄和犯罪记录，推断再犯概率。(Rules like age and record combine to infer recidivism probability.)

---

**Remarks**

- **优点**:
  - CORELS通过复杂算法实现最优。(CORELS achieves optimality using complex algorithms.)
  - 决策树和列表适合可解释性要求高的应用。(Decision trees and lists suit applications needing high interpretability.)
- **缺点**:
  - CORELS在处理大型数据集时较慢。(CORELS can be slow for large datasets.)
  - 集成模型(随机森林、梯度提升树)在性能上更优，但可解释性较差。(Ensemble models (e.g., random forests, gradient boosting trees) outperform but are less interpretable.)

---

**Additional desiderata?**

- **其他潜在标准**:
  - 是否能调整特征以实现不同结果？(Can we adjust features to achieve different outcomes?)
  - 能否确定隐藏特征对模型的影响？(Can we determine the impact of withholding certain features?)
  - 是否易于修改模型以解决行为问题？(Is it easy to modify the model to fix undesired behaviors?)
  - 是否能确定哪些数据点影响了模型预测？(Can we identify which data points influenced model predictions?)
- **分析**: 某些模型支持上述操作，但并非所有模型都能实现。(Some models support these criteria, but not all can.)



# 6 Interpretable complex models

**Inherently interpretable models**

- **特点**:
  - 数据从世界中获取，通过模型学习后直接输出透明预测。(Data captured from the world, trained into a model, and directly outputs transparent predictions.)
  - 示例: 线性模型可解释每个特征对输出的贡献。(Example: Linear models explain each feature's contribution to the output.)

---

**Interpretable complex models**

- **目标**: 使复杂模型（如深度神经网络）变得更加可解释。(Goal: Make inherently complex models (e.g., DNNs) more interpretable.)
- **方法**: 通过对复杂模型的特定设计或后处理步骤提升其可解释性。(Improve interpretability through specific designs or post-processing steps for complex models.)

- **深度学习中的解释性增强方法**:
  1. **全局平均池化的卷积神经网络(CNNs)**:
     - 使用类别激活图(CAM)可视化输入对特定类别的贡献。(Use class activation maps (CAM) to visualize how inputs contribute to specific categories.)
  2. **基于自注意力的Transformer**:
     - 生成基于注意力机制的解释。(Generate attention-based explanations.)

## 6.1 Class activation maps (CAM)

**Class activation maps (CAM)**

- **定义**: 针对卷积神经网络（CNNs）特定输出层的特征归因。(Built-in feature attribution for CNNs with specific output layers.)
- **方法**: 结合全局平均池化(GAP)和线性层生成类别激活图。(Use global average pooling (GAP) followed by a linear layer to generate class activation maps.)
- **作用**: 可视化图像中与特定分类相关的区域。(Visualizes regions in the image related to specific classes.)

---

**CNN architecture refresher**

- **常见CNN架构**:
  - AlexNet, VGG, ResNet, DenseNet。(Examples include AlexNet, VGG, ResNet, DenseNet.)
- **组成部分**:
  - 卷积层: 提取局部特征。(Convolutional layers: Extract localized features.)
  - 最大池化层: 降采样特征。(Max-pooling layers: Downsample features.)
  - 全连接层: 进行分类或回归。(Fully-connected layers: Perform classification or regression.)

---

**Layer types**

- **卷积层**:
  - 对每个位置应用核函数。(Apply kernel to each position.)
  - 提取共享的局部特征。(Extract shared localized features.)
- **最大池化层**:
  - 计算局部窗口的最大值。(Calculate the max value within a sliding window.)
  - 降低分辨率。(Downsample to lower resolution.)

---

**VGG architecture**

- **特点**:
  - 使用多层卷积和最大池化操作。(Uses multiple convolution and max-pooling operations.)
  - 最终输出扁平化为向量。(Flattens the output into a vector.)
- **优点**: 易于理解的分层结构。(Easily understood layered structure.)

---

**CNN output layers**

- **挑战**: 卷积和池化层产生额外维度。(Conv and max-pool layers add extra dimensions.)
- **解决方法**:
  1. 扁平化为$$k \times w \times c$$的向量。(Flatten to vector of $$k \times w \times c$$.)
  2. 对空间维度进行池化，生成长度为$$c$$的向量。(Pool along spatial dimensions to produce vector of length $$c$$.)
- **后续步骤**:
  - 应用全连接层和Softmax生成预测概率。(Apply fully-connected layers and Softmax to generate probabilities.)

---

**Global average pooling (GAP)**

- **方法**:
  - 对最后一层特征的空间平均值计算。(Calculate spatial average of last layer features.)
  - $$A_k = \frac{1}{h \times w} \sum_{i,j} a_{ij}^k$$。(Formula: $$A_k = \frac{1}{h \times w} \sum_{i,j} a_{ij}^k$$.)
- **优点**: 减少可学习参数，降低过拟合风险。(Fewer learnable parameters, reduces overfitting.)
- **应用**: 广泛用于流行架构，如ResNet, DenseNet。(Widely used in popular architectures like ResNet, DenseNet.)

---

**Putting it together**

- **步骤**:
  - 卷积和池化生成张量$$A \in \mathbb{R}^{k \times w \times c}$$。(Conv + max-pooling generates tensor $$A \in \mathbb{R}^{k \times w \times c}$$.)
  - GAP生成向量$$\hat{A} \in \mathbb{R}^c$$。(GAP generates vector $$\hat{A} \in \mathbb{R}^c$$.)
  - 全连接层计算分类分数$$z_y$$。(Fully-connected layer computes class scores $$z_y$$.)
  - 最后使用Softmax生成概率。(Finally, Softmax turns scores into probabilities.)

---

**Applied to final tensor A**

- **应用步骤**:
  - 将最终张量$$A$$压缩到GAP向量$$\hat{A}$$。(Final tensor $$A$$ is compressed into GAP vector $$\hat{A}$$.)
  - 使用全连接层生成类别预测。(Fully-connected layer generates class predictions.)
- **示例**: 在VGG架构中，最终张量长度为512。(In VGG architecture, final tensor length is 512.)

---

**CAM (Class Activation Maps)**

- **思路**: GAP+全连接层视为对每个空间位置的单独预测。(GAP + FC layers viewed as averaging predictions for each spatial position.)
- **公式**:
  $$z_y = \sum_k w_k^y \hat{A}_k$$ (计算类的特征重要性)。(Feature importance for class calculated as $$z_y = \sum_k w_k^y \hat{A}_k$$.)

---

**Qualitative evaluation**

- **可视化**: 使用CAM观察分类所依据的图像区域。(Use CAM to visualize image regions relevant to classification.)
- **示例**: CAM展示图像中激活的特定区域。(CAM highlights specific regions in the image activated for a class.)

---

**Relationship with GradCAM**

- **区别**:
  - CAM使用全局平均池化和全连接层。(CAM uses GAP and FC layers.)
  - GradCAM允许在无GAP的情况下操作。(GradCAM allows operations without GAP.)
- **公式比较**: GradCAM中的重要性公式与CAM类似。(GradCAM importance formula is similar to CAM.)

---

**Spatial locality assumption**

- **假设**: CAM/GradCAM假设内部特征图与原始输入空间相关。(Assume internal feature maps correspond to original input space.)
- **限制**:
  - 在深层网络中，后期层可能不满足空间局部性。(For very deep networks, later layers may not preserve spatial locality.)
  - GradCAM可在中间层操作以保留局部性。(GradCAM operates in intermediate layers to retain locality.)

---

**CAM remarks**

- **优点**:
  - 在对象定位中表现强。(Strong performance in object localization.)
  - 适用于特定架构，如使用GAP+全连接层。(Applicable for specific architectures like GAP + FC.)
- **缺点**:
  - 假设深层网络最后一层保持空间局部性，这在非常深的模型中可能失效。(Assumes spatial locality in final layer, which may fail in very deep models.)



## 6.2 Attention as explanation

**Attention**

- **定义**: 在深度学习中使用一小部分特征生成预测。(Using a small portion of features to generate predictions in deep learning.)
- **类比人类注意力**: 聚焦于视觉或听觉中的某些刺激。(Focus on certain stimuli, e.g., visual or auditory.)
- **应用**:
  - 通常在隐藏层中使用。(Typically used in hidden layers with internal features.)
  - 无关注的特征值设置为零。(Features without attention are set to zero.)

---

**Attention in DL**

- **核心作用**: 现代NLP和视觉模型的核心组件。(Core component of modern NLP and vision models.)
- **硬注意力 vs 软注意力**:
  - 硬注意力将值完全置零。(Hard: Multiply by exactly zero.)
  - 软注意力通过梯度下降学习更容易。(Soft: Approximate zero, easier to learn via gradient descent.)
- **关键问题**:
  1. 如何计算注意力值？(How are attention values computed?)
  2. 如何使用这些值？(How are these values used?)

---

**Self-attention example**

- **过程**:
  - 使用特征图生成注意力掩码，形状为$$[0,1]^{hw}$$。(Generate an attention mask from a feature map, shape $$[0,1]^{hw}$$.)
  - 每个位置的特征值与掩码逐元素相乘。(Element-wise multiply feature values by the attention mask.)

---

**Self-attention**

- **应用**:
  - 在CNN中偶有使用。(Occasionally used in CNNs.)
  - 广泛用于Transformer。(Widely used in transformers.)
- **Transformer中的用例**:
  - 语言建模(GPT-3, BERT)。(Language modeling, e.g., GPT-3, BERT.)
  - 蛋白质建模(AlphaFold)。(Protein modeling, e.g., AlphaFold.)
  - 视觉Transformer (ViT)。(Vision Transformers, ViT.)

---

**Case study: ViTs**

- **定义**: 基于自注意力的CNN替代方案。(An alternative to CNNs based on self-attention.)
- **特点**: 将输入图像分割为“tokens”，如同NLP模型的词。(Split input image into "tokens," similar to words in NLP.)

---

**Self-attention operations**

- **操作步骤**:
  1. 每个token生成查询(query)、键(key)和值(value)向量。(Generate query, key, and value vectors for each token.)
  2. 使用查询和键计算每对token的相关性。(Use query and key to compute relevance for each token pair.)
  3. 归一化相关性以得到注意力值。(Normalize relevance to get attention values.)
  4. 使用注意力值加权平均值向量以生成输出。(Use attention to average value vectors for output.)

---

**Attention matrix**

- **公式**:
  - 查询和键计算相关性矩阵：$$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$。(Calculate relevance as $$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$.)
  - 使用值和相关性生成输出：$$\text{SA}(Q,K,V) = AV$$。(Generate output as $$\text{SA}(Q,K,V) = AV$$.)

---

**Complete architecture**

- **特点**:
  - ViTs由多层自注意力组成，之间插入全连接层。(ViTs consist of many self-attention layers with fully-connected layers in between.)
  - 使用类token生成最终输出。(Use class token to produce final output.)

---

**Raw attention**

- **定义**: 将高注意力区域视为重要特征。(Define important features as those with high attention.)
- **方法**:
  - 检查每一层的注意力，特别是对类token的注意力。(Examine attention at each layer, especially for the class token.)

---

**Attention rollout**

- **问题**: 每层的token间信息会混合。(Information mixes between tokens at each layer.)
- **解决方案**:
  - 将注意力视为图，计算其传播路径。(Treat attention as a graph and calculate its flow.)
  - 提取传播路径中的类token行。(Extract class token row from the rollout matrix.)

---

**Examples**

- **用途**:
  - 使用注意力可视化深度学习模型的重要特征。(Use attention to visualize important features in deep learning models.)
  - 示例包括BERT、ViTs等的注意力分布。(Examples include attention distributions in BERT, ViTs, etc.)

---

**Attention skepticism**

- **疑问**:
  - 注意力是否真正反映了特征重要性？(Does attention truly reflect feature importance?)
  - 注意力可能与人类关注不同。(Attention may differ from human attention.)
- **研究**: 一些论文探讨注意力的解释性问题。(Several papers explore skepticism about attention's interpretability.)

---

**Remarks**

- **优点**:
  - 注意力自动计算，便于预测。(Attention is automatically calculated for prediction.)
  - 提供直观的特征加权机制。(Provides intuitive feature weighting.)
- **缺点**:
  - 不易聚合不同层间的注意力。(Not easy to aggregate attention across layers.)
  - 可能忽略重要特征。(May ignore other important operations.)

---

**Summary**

- **发展**:
  - 全局平均池化和自注意力最初为提高预测性能而引入。(Global average pooling and self-attention were introduced to improve predictive performance.)
  - 后来被用于提升模型解释性。(Later used to improve model interpretability.)
- **扩展**: 其他方法也被用于提升深度学习模型的可解释性。(Other approaches explicitly aim to make deep learning models more interpretable.)

---

**Perspective**

- **观点**:
  - 没有绝对“错误”的方法，但某些方法可能不符合用户需求。(No method is "wrong," but some may misalign with user needs.)
  - 度量标准应根据用户目标设计。(Metrics should be designed for specific user objectives.)







