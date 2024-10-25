[TOC]

# 1 线性回归(Linear Regression)

## 1.1 概括

线性回归的关键概念和方法，如**成本函数**、**梯度下降法**和**均方误差（MSE）**。总结如下：

**(1)学习模型**：

- 目标是根据特征（如出生体重）预测目标值（如智商）。

- 目标与特征之间的关系用一条直线表示，公式为 y=mx+b。
- 关键是找到 m（斜率）和 b（截距）的最佳值。

**(2)均方误差（MSE,`Mean Squared Error`）**：

- MSE 用于衡量预测值与实际值的接近程度。
- 通过预测值与实际值的平方差的平均值计算。

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$，MSE 越小，表示模型的预测越准确。其中：

- $$n$$是数据点的数量
- $$y_i$$是第 $$i$$ 个实际值， - $$\hat{y}_i$$ 是第 $$i$$ 个预测值， - $$y_i - \hat{y}_i$$ 是预测误差，表示第 $$i$$ 个预测值与实际值的差异。 

注意：PPT中前面是 $$\frac{1}{2n}$$ ，多除以一个2不会影响衡量，但可以简化后续梯度下降法中的计算(因为MSE对参数的偏导数有一个2)

**(3)梯度下降法**：

- 梯度下降法用于最小化成本函数（即MSE），从而找到最优的模型参数。
- 通过反复更新 m 和 b 的值，朝着减少误差的方向优化。

## 1.2 代价函数(Cost Function)

**MSE（均方误差）**本身是一种**衡量误差的度量**，而**代价函数（Cost Function）**是用于**优化模型参数**的函数。MSE 是代价函数的具体形式之一，代价函数的目标是通过衡量模型的误差，找到能够最小化这个误差的参数组合。

- **代价函数**是一个用于衡量模型预测与实际值之间差异的函数，通常用于指导模型优化。在线性回归中，代价函数是 MSE 的形式，即：

$$
J(\theta_0, \theta_1) = \frac{1}{2n} \sum_{i=1}^{n} (h_{\theta}(x^{(i)}) - y^{(i)})^2
$$

这里， $$h_{\theta}(x^{(i)})$$ 是线性回归模型的预测值， $$y^{(i)}$$ 是实际值， $$n$$ 是样本数量。

- **MSE 作为代价函数**：在这个公式中，MSE 是代价函数的一种形式。代价函数 $$J(\theta_0, \theta_1)$$ 表示模型在当前参数下的预测误差，目标是通过调整 $$\theta_0$$ 和 $$\theta_1$$ 来最小化它。



## 1.3 梯度下降(Gradient Descent)

**梯度下降法**是一个用于优化代价函数（如均方误差 MSE）的算法，通过不断调整模型参数，使代价函数的值逐步减少，直到找到全局最小值。

<font color = blue>**1. 梯度下降法的目标**</font>

梯度下降的主要目标是**最小化代价函数**，例如线性回归中的**均方误差（MSE）**。通过找到能够使代价函数最小化的模型参数（如 $$\theta_0$$ 和 $$\theta_1$$ ），来提升模型的预测精度。代价函数的公式为：

$$
J(\theta_0, \theta_1) = \frac{1}{2n} \sum_{i=1}^{n} (h_{\theta}(x^{(i)}) - y^{(i)})^2
$$

这里 $$h_{\theta}(x^{(i)})$$ 是模型的预测值， $$y^{(i)}$$ 是实际值， $$n$$ 是数据点的数量。

<font color = blue>**2.梯度的概念**</font>

梯度是指代价函数关于模型参数的**导数**，它指示了代价函数在当前点的变化方向。通过计算代价函数对参数的梯度，我们可以确定应该如何调整参数以减少误差。

- 当代价函数对某个参数的导数为正时，表示参数值需要减少；
- 当导数为负时，表示参数值需要增加。

<font color = blue>**3. 梯度下降法的步骤**</font>

梯度下降法的核心步骤是不断调整模型参数，使得代价函数的值逐步下降，直到收敛到最小值。具体步骤如下：

1. **初始化参数**：随机初始化 $$\theta_0$$ 和 $$\theta_1$$ 的初始值，通常初始值为 0。

2. **计算梯度**：对每个参数 $$\theta_j$$ （这里 $$j = 0, 1$$ ），计算代价函数对其的偏导数（梯度）。对于线性回归模型，梯度的计算公式如下：
 $$\frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) = \frac{1}{n} \sum_{i=1}^{n} (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}$$ 
 $$\theta_j$$ 是模型的参数， $$x_j^{(i)}$$ 是第 $$i$$ 个数据样本的第 $$j$$ 个特征值(平方项前面的常数?)。

3. **更新参数**：根据学习率 $$\alpha$$ 和梯度的值，更新每个参数。更新公式为：
 $$\theta_j = \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$ 
   
   这里的 $$\alpha$$ 是**学习率**，它控制参数更新的步长大小(可以理解为求导后的一个常数，会同时增大斜率和截距)。学习率过大可能导致跳过最优解，学习率过小则会导致收敛速度变慢。
   
4. **重复步骤**：重复计算梯度和更新参数，直到代价函数的值不再显著变化，也就是收敛到全局最小值。

<font color = blue>**4. 学习率的选择**</font>

学习率是梯度下降法中的一个重要参数。它决定了每次更新参数时的步长大小。如果学习率太大，更新步伐太快，可能导致跳过最优解；如果学习率太小，模型更新会非常缓慢，导致训练时间过长。通常会进行多次实验来选择合适的学习率。

<font color = blue>**5. 收敛条件**</font>

梯度下降的过程会一直进行，直到找到一个使代价函数几乎不再下降的参数值，也就是当代价函数接近某个最小值时停止更新。这时的参数就是最优参数。

<font color=blue>**6. 批量梯度下降和随机梯度下降**</font>

梯度下降的不同变种：

- **批量梯度下降（Batch Gradient Descent）**：使用整个训练数据集来计算梯度。每次参数更新使用所有数据点的均值。这种方法比较稳健，但当数据集非常大时，计算代价也较高。
  
- **随机梯度下降（Stochastic Gradient Descent, SGD）**：每次参数更新仅使用一个数据点的梯度。SGD 的更新频率高，计算速度快，但误差曲线可能较为波动。
  
- **小批量梯度下降（Mini-batch Gradient Descent）**：结合了以上两种方法，每次使用一小部分数据进行更新，既降低了波动性，又减少了计算开销。

<font color=blue>**7.梯度下降的优缺点**</font>

- **优点**：简单且易于实现；能够有效找到代价函数的最优解（特别是对于凸函数）。
  
- **缺点**：选择学习率较为困难，可能会导致收敛过慢或跳过最优解；对非凸函数，可能会陷入局部最小值，而非全局最小值。

## 1.4 代码实例

让我们考虑一个非常基本的线性方程，即y=2x+1。在这里，‘x’是自变量，‘y’是因变量。我们将使用这个方程创建一个用于训练线性回归模型的虚拟数据集。以下是创建数据集的代码。

```
import numpy as np
# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
```

首先要做的是定义模型架构：

```
import torch
from torch.autograd import Variable
class linearRegression(torch.nn.Module):
	# 输入和输出的维度
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
```

我们定义了一个线性回归的类，该类继承了 `torch.nn.Module`，这是包含所有必要函数的基本神经网络模块。我们的线性回归模型只包含一个简单的线性函数。接下来，我们使用以下代码实例化模型。

```
in_feature_size = 1       # takes variable 'x' - 1 features
out_feature_size= 1       # takes variable 'y'
// 学习率，模型在训练过程中更新权重时的步长。较小的学习率可以让模型缓慢收敛，更准确地找到最优解
learningRate = 0.01
// 定义训练的轮数，即模型要遍历整个训练集的次数
epochs = 100

model = linearRegression(in_feature_size, out_feature_size)
##### For GPU #######
// 检查当前系统是否有可用的 GPU,如果有则将模型移到 GPU 上
if torch.cuda.is_available():
    model.cuda()
```

之后，我们初始化用于模型训练的损失函数（均方误差）和优化函数（Adam）。

```
// MSE函数,计算模型预测值和真实值之间差距的平方平均值，用于衡量模型的误差
criterion = torch.nn.MSELoss()
// 使用Adam优化算法,lr为学习率，控制每次权重更新的步长
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
```

完成所有初始化后，我们现在可以开始训练模型了。以下是训练模型的代码。

```
// 用来记录每个 epoch 的损失值，以便之后查看或绘制训练损失的变化趋势
training_loss = []

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
    	//将x_train转换为 PyTorch 张量并放到 GPU 上，且包装成 Variable，使其支持自动求导
        inputs = Variable(torch.from_numpy(x_train).cuda())
        //将y_train转换为 PyTorch 张量并放到 GPU 上，包装成 Variable，用于计算损失函数。
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # 清除优化器中的梯度缓存。每次计算完梯度后，需要清空之前累积的梯度，以免影响本次训练的参数更新。
    optimizer.zero_grad()

    # 将输入数据 inputs 传入模型，得到模型的预测输出 outputs。
    outputs = model(inputs)

    # 计算模型预测输出outputs与真实标签labels之间的损失值，使用之前定义的MSE损失函数criterion
    loss = criterion(outputs, labels)
    # 计算损失函数对模型参数的梯度,PyTorch会自动根据损失的大小,计算模型中每个参数需要调整的方向和幅度
    loss.backward()

    # 根据梯度更新模型参数。Adam 优化器会根据学习率和梯度信息对参数进行调整，以减小损失。
    optimizer.step()
	# 打印当前迭代的 epoch 和对应的损失值 loss.item()，以便在训练过程中实时查看模型的收敛情况
    print('epoch {}, loss {}'.format(epoch, loss.item()))
	# 将当前的损失值保存到 training_loss 列表中，方便之后分析模型的损失变化。
    training_loss.append(loss.item())
```

现在我们的线性回归模型已经训练完成，让我们来测试一下。由于这是一个非常简单的模型，我们将在现有的数据集上进行测试，并绘制图表来查看原始输出与预测输出的对比。

```
使用 torch.no_grad() 环境上下文，表示在测试阶段不需要计算梯度(不会进行反向传播)
with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
    	// 将 x_train 转换为张量，传入模型得到预测值，并转换为 NumPy 数组。
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

推断当 `x = 7` 时，模型的预测输出 `y`。

```
# to infer y when x = 7

x_infer = np.array([7], dtype=np.float32)
x_infer = x_infer.reshape(-1, 1)

with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_infer).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_infer))).data.numpy()
    print('when x = 7, y = ', predicted)
```

