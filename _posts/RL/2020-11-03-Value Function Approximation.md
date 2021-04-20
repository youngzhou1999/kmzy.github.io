---
layout: post
title: 价值函数近似
categories: RL
description: Value Function Approximation
keywords: Reinforcement Learning
---

## Value Function Approximation

上一节介绍的求解强化学习问题的方法都归属于表格型方法，当问题规模很大时，求解会遇到几个问题：

1. 太多的状态或动作需要保存在内存中
2. 单独地求解某个状态的价值函数太慢

因此，当强化学习规模较大时，通常使用一些函数来近似价值函数，通过评估这些拟合函数来求解问题。其表达形式如下：

$$
\hat{V}(s,\omega)\approx{V(s)} \\
\hat{Q}(s,a,\omega)\approx{Q(s,a)} \\
\hat{\pi}(a,s,\omega)\approx{\pi{(a|s})}
$$

其中，$\omega$为近似函数的参数。

几种常见的近似函数形式如下：

1. 线性特征组合（Linear Combination of Features）
2. 神经网络（Neural Network）
3. 决策树（Decision Tree）
4. 最近邻（Nearest Neighbor）

其中，前两种近似函数属于可微函数，利用此性质可以很好地进行优化。例如梯度下降算法。

#### 梯度下降（Gradient Descend）

1. 目标（Objective）

考虑一个可微的目标函数$J(\omega)$，其参数为向量$\omega$，梯度下降的目标为最小化目标函数即：

$$
\min{J(\omega)}\\
\rm{where}\quad\omega=\omega^*
$$

2. 梯度（Gradient）

定义目标函数的梯度如下：

$$
\nabla_{\omega}{J(\omega)}=\Big(\frac{\partial{J(\omega)}}{\partial{\omega_1}},\frac{\partial{J(\omega)}}{\partial{\omega_2}},\cdots,\frac{\partial{J(\omega)}}{\partial{w_n}}\Big)^T
$$

3. 更新

调整参数$\omega$，目标函数朝着负梯度的方向更新：

$$
\Delta{\omega}=-\frac{1}{2}\alpha\nabla_{\omega}J(\omega) \\
\omega\gets{\omega+\Delta{\omega}}
$$

其中，$\alpha$为超参数学习率，又叫步长；系数$\frac{1}{2}$是为了二次函数求导时方便约去常数项。

### 1. 价值函数的评估（VFA Evaluation）

#### 1. 已知价值函数（假设）（Value Function Approximation with an Oracle）

若假设已知真实的价值函数，则价值函数估计类似于监督学习的方法，计算真值（label）和输出值（output）的误差即可。

具体地，使用平均方差误差（Mean Square Error（MSE））来定义损失函数（Loss Function）：

$$
J(\omega)=\mathbb{E}[\Big(V^\pi(s)-\hat{V}(s)\Big)^2]
$$

更新过程即为上述梯度下降算法过程。

##### 1. 使用特征向量描述状态（Describe States by Feature Vectors）

使用特征向量来表征状态是非常常见的一种方法，它可以在一个较复杂的系统中，提炼出状态，其表达式为：

$$
\mathbb{x}(s)=\Big(\mathbb{x_1(s)},\mathbb{x_2(s)},\cdots,\mathbb{x_n(s)}\Big)^T
$$

##### 2. 线性价值函数近似（已知真实值）（Linear Value Function Approximation with an Oracle）

考虑在已知真实的价值函数的条件下，使用线性函数来估计价值函数。

1. 状态价值函数为：使用参数向量$\omega$对特征向量进行线性组合，即：

$$
\hat{V}(s,\omega)=\sum_{j=1}^n{\mathbb{x_j(s)}\omega_j}
$$

2. 目标函数采用MSE定义，是一个关于参数$\omega$的二元式：

$$
\begin{aligned}
J(\omega)&=E_\pi[\Big(V^{\pi}(s)-\hat{V}(s,\omega)\Big)^2] \\
&=E_\pi[\Big(V^{\pi}(s)-\mathbb{x(s)}^T\omega\Big)^2]
\end{aligned}
$$

3. 更新，计算出参数$\omega$的更新值：

$$
\begin{aligned}
\Delta{\omega}&=-\frac{1}{2}\alpha\cdot2\Big(V^{\pi}(s)-\hat{V}(s,\omega)\Big)\cdot(-\mathbb{x}(s)) \\
&=\alpha\Big(V^{\pi}(s)-\hat{V}(s,\omega)\Big)\mathbb{x(s)}
\end{aligned}
$$

因此对于线性函数来说，参数$\omega$的更新值为：

$$
\mathbb{update=stepsize\times{prediction\_eoor\times{feature\_values}}}
$$

对于线性函数估计来说，随机梯度下降（stochastic gradient descend）可以收敛到全局最优，因为线性函数只有一个最优值。

- 线性价值函数近似——直接查表的特征

考虑一种特殊的情况，查表的特征指的是基于单热点码的特征向量，其形式如下：

$$
\mathbb{x}^{table}(s)=\Big(\rm{1}(s=s_1),\cdots,\rm{1}(s=s_n)\Big)^T
$$

通过这样的构造，我们可以发现在参数$\omega$向量中的每一个元素就代表了那一个状态的价值。即：

$$
\hat{V}(s)=(\rm{1}(s=s_1),\cdots,\rm{1}(s=s_n))(\omega_1,\cdots,\omega_n)
$$

因此，此时有$\hat{V}(s_k,\omega)=\omega_k$。

#### 2. 不知价值函数（Without Oracle）（Value Function Approximation without Oracle）

首先来回顾一下无模型的预测。在无模型的预测中：

1. 目标为：在当前策略$\pi$下，估计状态价值函数
2. 用表格来存储/估计$V$函数和$Q$函数
3. 有两种常见算法

MC：在整个幕完成后做更新；TD：每走一步更新一次（TD(0)）

这两种算法的原理都是在循环中更新状态价值函数或动作-状态价值函数，即：

$$
\pi\to{V}\to{\hat{V}}\quad\rm{or}\quad{\pi\to{V}\to{Q}\to{\hat{V}}}
$$

这很具有启示作用，当不知道价值函数时，可以把函数近似的思想放入循环中更新，这样就可以达到不断更新的作用，从而最终收敛。

由上述的梯度下降更新公式，我们可以得到：

$$
\Delta{\omega=\alpha(\textcolor{red}{V^{\pi}(s)}-\hat{V}(s,\omega))}\nabla_\omega{\hat{V}(s,\omega)}
$$

因此，考虑将上述红色部分的target替换成相应的累计收益，可以达到很好的近似效果。与第二节类似，根据选取的target不同，有两种不同版本的算法：

- 蒙特卡洛估计

- TD估计

1. 蒙特卡洛估计

使用累计的衰减的回报和$G_t$作为target，其更新公式为：

$$
\Delta{\omega}=\alpha(\textcolor{red}{G_t}-\hat{V}(s_t,\omega))\nabla_\omega{\hat{V}(s_t,\omega)}
$$

2. TD估计

考虑最简单的一种情况TD(0)：使用往前走一步的回报作为taregt，其更新公式为：

$$
\Delta\omega=\alpha(\textcolor{red}{R_{t+1}+\gamma\hat{V}(s_{t+1},\omega)}-\hat{V}(s_t,\omega))\nabla_\omega{\hat{V}(s_t),\omega}
$$

> 1. MC估计和TD估计的比较（Comparison Between MC and TD Prediction）

对于蒙特卡洛估计（MC）：

1. 使用的目标$G_t$是无偏的，即重复次数足够多时，$G_t$会收敛到期望值即$V^\pi$；但是$G_t$带有噪声，因为它采样了整个幕，因此方差会比较大
2. 对于VFA即价值函数近似（Value Function Approximation），因为设定了目标，因此可以把训练数据视作一个一个的<输出，标签>对，故其与监督学习思想类似，即：

$$
<s_1,G_1>,<s_2,G_2>,\cdots,<s_t,G_t>
$$

3. 线性函数拟合时，其更新公式为：

$$
\Delta{\omega}=\alpha(\textcolor{red}{G_t}-\hat{V}(s_t,\omega)){\mathbb{x(s_t)}}
$$

4. MC估计在线性与非线性函数拟合时，都能够收敛（因为它的梯度信息是全的，即等式右边的target中不含参数$\omega$）

对于时序差分估计（TD）：

1. 使用的目标$R_{t+1}+\gamma\hat{V}(s_{t+1},\omega)$是有偏的，因为它的参数$\omega$包含在优化式中；从另一种角度也可以分析：它的target是从过去的数据中估计出来的，即$V{(s_{t+n})}$是过去的值，还未更新，因此TD-trager的期望不等于价值函数
2. 同样，训练数据也可以当作监督学习：

$$
<s_1,R_{2}+\gamma\hat{V}(s_2,\omega)>,<s_2,R_3+\gamma\hat{V}(s_3,\omega)>,\cdots,<s_{t-1},R_t>\quad\rm{or}\quad<s_t,0>
$$

3. 线性函数拟合时，其更新公式为：

$$
\Delta{\textcolor{red}{\omega}}=\alpha(R_{t+1}+\gamma\hat{V}(s_{t+1},\textcolor{red}{\omega})-\hat{V}(s_t,\omega))\mathbb{x}(s_t)
$$

可以看到，其更新的目标中含有正在优化的参数$\omega$，因此，这种梯度叫半梯度（semi-gardient）

4. TD估计只有在线性函数拟合时，能够近似地收敛到最优值

### 2. 价值函数近似的控制（VAF Control）

##### 1. 广义策略迭代（Generalized Policy Iteration）

广义策略迭代的核心为：近似$Q$函数，其步骤仍分为两个部分，即：

1. 策略评估：使用近似的策略评估即$\hat{Q}(.,.,\omega)\approx{Q^\pi}$
2. 策略提升：使用$\epsilon$-greedy来做策略提升

##### 2. 已知动作-状态价值函数（假设）（Value Function Approximation Control with an Oracle）

在已知$Q^{\pi}$函数的情况下：

1. 目标：使用参数$\omega$近似动作-状态价值函数

$$
Q(s,a,\omega)\approx{Q^\pi}(s,a)
$$

2. 优化目标函数：通过最小化$Q_\omega$和$Q^\pi$两个函数的MSE来优化

$$
J(\omega)=\mathbb{E}_\pi{p[(Q^\pi(s,a)-\hat{Q}(s,a,\omega))^2]}
$$

3. 梯度下降：使用随机梯度下降（sGD）得到局部最小值

$$
\Delta{\omega}=\alpha(Q^\pi(s,a)-\hat{Q}(s,a,\omega))\nabla_\omega\hat{Q}(s,a,\omega)
$$

##### 3. 线性动作-状态价值函数近似（Linear Action-Value Function Approximation）

与线性状态价值函数估计类似，使用特征向量来构造$Q$函数，并使用sGD来更新参数$\omega$，即：

1. 特征向量

$$
\mathbb{x}(s,a)=\Big(\mathbb{x_1(s,a)},\mathbb{x_2(s,a)},\cdots,\mathbb{x_n(s，a)}\Big)^T
$$

2. 使用参数向量$\omega$线性组合特征以构建Q函数

$$
\hat{Q}(s,a,\omega)=\mathbb{x}(s,a)^T\omega=\sum_{j=1}^n\mathbb{x_j}(s,a)w_j
$$

3. 使用随机梯度下降（sGD）来更新参数

$$
\Delta\omega=\alpha(Q^\pi(s,a,\omega)-\hat{Q}(s,a))\mathbb{x}(s,a)
$$

##### 4. 不知动作-状态价值函数（Value Function Approximation Control without Oracle）

同预测时类似，当我们不知道在策略$\pi$下的动作-状态函数时，我们将目标值替换为收益和，并放入循环中迭代，以达到收敛。

同理，根据选择的收益和的不同，分为两大类算法：

1. 广义蒙特卡洛控制
2. 广义TD-learning控制，其中根据贝尔曼方程的选择不同，具体可以细分为两种子算法：
    1. Sarsa
    2. Q-learning

##### 5. 广义蒙特卡洛控制（Generalized Monte Carlo Control）

使用累计的衰减的幕收益和$G_t$来作为目标，故更新表达式如下：

$$
\Delta\omega=\alpha(\textcolor{red}{G_t}-\hat{Q}(s_t,a_t))\nabla_\omega\hat{Q}(s_t,a_t,\omega)
$$

##### 6. 广义TD-learning控制（Generalized TD-Learning Control）

基于贝尔曼期望方程，可以得到Sarsa算法的更新表达式为：

$$
\Delta\omega=\alpha(\textcolor{red}{R_{t+1}+\gamma{\hat{Q}(s_{t+1},a_{t+1},\omega)}}-\hat{Q}(s,a))\nabla_\omega\hat{Q}(s_t,a_t,\omega)
$$

基于贝尔曼最优方程，可以得到Q-learning算法的更新表达式为：

$$
\Delta\omega=\alpha(\textcolor{red}{R_{t+1}+\gamma\max_{a}{\hat{Q}(s_{t+1},a,\omega)}}-\hat{Q}(s,a))\nabla_\omega\hat{Q}(s_t,a_t,\omega)
$$

以Sarsa算法为例，其算法框架如下：

```python
# Sarsa for VAF control
input: a differential function q: s x a x R^d -> R

initialize: wights w in R^d arbitrarily(eg w = 0)

Reapeat(for each episode):
    s, a = initial state and action of episode(e.g. eps_greedy() or reset())
    Repeat(for each episode):
        take action a, observe reward r and next state s_
        if s_ is terminal:
            # update in terminal
            w <- w + alpha * (r - q(s,a,w)) * diff_w(q(s,a,w))
        a_ = eps_greedy(s_)
        # uodate in normal case
        w <- w + alpha * (r + gamma * q(s_,a_,w) - q(s,a,w)) * diff_w(q(s,a,w))
        s = s_
        a = a_
```

##### 7. 几种价值函数近似的控制方法的收敛情况（Convergence with TD and MC control）

- TD方法总结

首先，基于TD的方法，其更新式右边的TD-target部分含有正在优化的参数$\omega$，因此其仅为半梯度（semi-gradient），因此TD方法在使用离轨学习形式（off-policy）或非线性函数拟合时可能会发散。其次，TD的更新式中，包含由两部分的优化：贝尔曼方程的逼近和价值函数的逼近，因此优化过程不稳定。

因此，在使用离轨学习的控制算法时，由于行为策略和目标策略并不相同，在价值函数近似时容易发散。

- 几种算法的收敛性

| 算法       | 表格 | 线性函数近似 | 非线性函数近似 |
| ---------- | ---- | ------------ | -------------- |
| MC         | √    | （√）        | ×              |
| Sarsa      | √    | （√）        | ×              |
| Q-learning | √    | ×            | ×              |

其中，√表示全局最优，（√）表示大致的全局最优，×表示可能不收敛

##### 8. 强化学习训练时的死亡三角（The Deadly Triad in Reinforcement Learning）

在强化学习中，有三方面因素可能导致训练时的不稳定和不收敛。

1. 函数近似（function approximation）

函数近似指的是针对状态空间很大时，利用提取特征等方式重新生成状态的方式。因为只是近似，所以引入了很多误差。

2. 自举（Bootstrapping）

自举指的是在更新目标的时候包含了之前已经存在的观测，而不是使用真实的回报和。因此也引入了噪声，增大了不确定性。一个明显的问题就是对估计值会过度估计（over-estimate）

3. 离轨学习（off-policy learning）

离轨学习值的是使用一个行为策略（即一个分布）的数据来训练另外一个目标策略。因此也引入了不确定因素。

##### 9. 批量强化学习（Batch Reinforcement Learning）

上述的每执行一步就做一次梯度下降更新的方法很简单，但是它不高效。为此引入了基于批量数据的方法来做价值函数近似。具体以价值函数近似的预测过程为例：

设数据集表示如下：

$$
D=\{<s_1,V_1^\pi>,<s_2,V_2^\pi>,\cdots,<s_T,V_T^\pi>\}
$$

则目标为：找到最优的参数$\omega^*$来最好地拟合数据集合$D$。因此$\omega^*$的计算表达式为：

$$
\begin{aligned}
\omega^*&=\arg\min_{\omega}\mathbb{E}[(V^\pi-\hat{V}(s,\omega))^2] \\
&=\arg\min_{\omega}\sum_{t=1}^T{(V_t^\pi-\hat{V}(s_t,\omega))^2}
\end{aligned}
$$

但是一个很显然的问题在于，这样选择数据的方式选出来的数据的关联性很强，不适合于训练。因此，考虑从数据集合中随机选取一定批量大小（batch-size）的数据来做训练。其操作步骤为：

1. 从数据集合$D$中随机选取一个批量大小的数据

$$
<s,V^\pi>\quad\sim\quad{D}
$$

2. 利用采集到的数据在做随机梯度下降（sGD）

$$
\Delta\omega=\alpha(V^\pi-\hat{V}(s,\omega))\nabla_\omega\hat{V}(s,\omega)
$$

可以证明：这种批量处理数据的方式可以收敛到最小方差解法，即：

$$
\omega^{LS}=\arg\min_{w}\sum_{t=1}^T(V_t^\pi-\hat{V}(s,\omega))^2
$$

##### 10. 线性与非线性价值函数近似的比较（Comparison Between Linear and Non-Linear Function Approximation）

线性价值函数近似在给定正确的特征集的条件下表现很好，但是它有一个缺点就是需要人类手动来设计特征集。因此，一种可以替代的近似方式为直接学习状态而不需要特别地设置特征。深度神经网络可以很好地做到这一点，且后续基于卷积神经网络地价值函数近似可以实现端到端地特征提取与训练，比较适用。因此，下面将主要讨论利用深度神经网络来近似价值函数，也就是深度强化学习（Deep Reinforcement Learning）。

##### 11. 深度神经网络（Deep Neural Network）

深度神经网络的图示如下：

<center>
    <img src="/images/posts/blog/image-20210410144451160.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>

它由许多层线性函数组成，层与层之间使用非线性操作连接，其表达式为：

$$
f(x;\theta)=W_{L+1}^T\sigma(W_L^T\sigma(\dots\sigma{W_1^Tx+b_1})+\dots+b_L)+b_{L+1}
$$

其中，$x$表示输入，$W$表示权重向量，$b$表示偏移，$\sigma$表示激活函数，$\theta$表示参数

- 损失函数与反向传播算法

神经网络使用输出与实际值的差值的平方的均值来构造损失函数，即：

$$
L(\theta)=\frac{1}{n}\sum_{i=1}^{n}\Big(y^{(i)}-f(x;\theta)\Big)^2
$$

同时，使用链式法则来反向传播梯度以更新权重向量，即：

- 反向传播阶段

$$
\begin{cases}
\delta_j^{L+1}=(y^{(j)}-f(x;\theta))f'(v_j^{L+1}) \\
\delta_j^{k}=\Big(\sum_{i=1}^n\delta_{i}^{k+1}w_{ij}^{k+1}\Big)f'(v_j^{k}),\quad{k\le{L}}
\end{cases}
$$

在输出层，误差来源于预测误差和激活函数对输出值的梯度，而在输出层以内的层，误差来源于其后一层的加权误差和与激活函数的梯度。

- 权值更新阶段

$$
w_{ji}^k(s+1)\gets{w_{ji}^k}(s)-\alpha\frac{\partial{E}}{\partial{w_{ji}^k}}=w_{ji}^k+\alpha\delta_j^kx_i^{k-1}
$$

即：

$$
\rm{update=stepsize\times{(prediction\_error\times}gradient)}\times{input}
$$

##### 12. 卷积神经网络（Convolution Neural Network）

卷积神经网络的图示如下：

<center>
    <img src="/images/posts/blog/image-20210410161823829.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>

它由卷积层、池化层、全连接层和输出层组成。其中：

- 卷积层用于提取2D图像特征信息，其使用小规模的过滤器（e.g. 3x3）对图像的局部像素做内积和
- 池化层用于降维或强化特征信息（e.g. maxpool()）
- 全连接层和输出层即为普通的神经网络

卷积神经网络在计算机视觉领域使用非常广泛，更多关于卷积神经网络的资料或视频可以查看斯坦福大学的[CS231n课程](https://cs231n.github.io/convolutional-networks/)。

下面是关于卷积神经网络处理图像时数据变化的笔记内容。

###### convolution layer

input shape: $W\times{H}\times{D}$

hyper parameters: 

- number of filters: $K$ (usually power of 2)
- filter size: $F\times{F}$（depth is the same as input）
- the stride: $S$
- the amount of zero padding: $P$

hence, the output shape is: 

- $W'=(W+2P-F)/S+1$
- $H'=(H+2P-F)/S+1$
- $D'=K$

number of parameters: (weight + bias)

$$
K*(F*F*D+1)
$$

in the output, the $d$-th depth silce (of size $W'\times{H'}$) is the result of performing a valid convolution of the $d$-th filter over the input with a stride of $S$, and the offset by $d$-th bias.

###### pooling layer

input shape: $W\times{H}\times{D}$

hyper parameters: 

- pooling size: $F\times{F}$
- the stride: $S$

hence, the output shape is: 

- $W'=(W-F)/S+1$
- $H'=(H-F)/S+1$
- $D'=D$


$$
out(N_i, C_{out_j})=bias(C_{out_j})+\sum^{C_{in}-1}_{k=0}weight(C{out_j},k)\bigotimes input(N_i,k)
$$

##### 13. 深度强化学习简介（Deep Reinforcement Learning）

深度强化学习是机器学习和人工智呢领域的前沿方向。它使用深度神经网络来表示：

1. 价值函数
2. 策略函数（基于策略梯度的强化学习）
3. 环境模型

利用深度神经网络，损失函数使用随机梯度下降算法（sGD）来优化。

同时，深度强化学习领域也存在一些挑战：

1. 效率问题：深度神经网络有很多的模型参数需要优化
2. 死亡三角，即：
    1. 非线性函数近似
    2. 自举
    3. 离轨学习

##### 14. Deep Q-network（DQN）

DQN是Deepmind在2015年发表在nature上的一个算法，它使用神经网络来近似动作-状态价值函数即$Q$函数，在雅达利游戏上达到了专业玩家的水平。

首先，使用Q-learning来做价值函数近似有两个重要的问题需要考虑：

1. 样本之间的关联性

正如之前所说，强化学习的数据并不是独立同分布（i.i.d）的，图像的下一帧往往与当前帧有很大关系。如果样本之间本身就有较大的关系，那么是不利于训练的。

2. 不稳定的目标

正如之前所说，基于TD方法的更新，其target中包含了正在优化的参数，梯度是不完全正确的，因此这也会给训练带来很大困难。

针对这两个问题，DQN使用了两个技巧：

1. 经验重放（experience replay）

即使用一个容量较大的容器，把每一次的转移关系数据存入其中，在训练时，从容器中随机选取一个batch大小的数据进行训练。

2. 相对固定的目标

DQN使用两个神经网络来进行训练，一个用来表示目标网络，另一个是当前的评估网络，目标网络的参数的更新会慢于当前的评估网络，这样target会更稳定。

##### 1. 经验重放（experience replay）

为了减小样本数据之间的相关性，DQN构造了一个存储转移关系的缓冲容器$D$，即:

$$
D=\{<s_1,a_1,r_1,s_{2}>,<s_2,a_2,r_2,s_{3}>,\cdots,<s_n,r_n,a_n,s_{n+1}>\}
$$

每一个元组$<s_t,a_t,r_t,s_{t+1}>$表示一次智能体与环境的交互，从而形成转移关系。其示意图如下：

<center>
    <img src="/images/posts/blog/image-20210410171729586.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>

执行经验重放算法的框架为：

1. 采样

从容器中随机采取数据样本：

$$
(s,a,r,s')\quad\sim\quad{D}
$$

2. 利用采集到批量数据计算TD-target

$$
\rm{TD\_{target}}=r+\gamma\max_{a'}\hat{Q}(s',a',\omega)
$$

3. 使用随机梯度下降来更新网络权值

$$
\Delta{\omega}={\alpha(\rm{TD\_target}-\hat{Q}(s,a,\omega))}\nabla_\omega\hat{Q(s,a,\omega)}
$$

##### 2. 固定目标（Fixed Target）

由于TD-target中含有正在优化的参数$\omega$，因此带来了不确定性，会导致训练结果不稳定。为了训练的时候更稳定，需要固定在TD-target中的参数。

因此，考虑一个新的网络，其参数为$\omega^-$，表示这个网络参数的更新来自于当前的评估网络，但是更新速度较慢。那么基于此，可以得到带固定目标的算法框架为：

1. 采样

从容器中随机采取数据样本：

$$
(s,a,r,s')\quad\sim\quad{D}
$$

2. 利用采集到批量数据和目标网络来计算TD-target

$$
\rm{TD\_{target}}=r+\gamma\max_{a'}\hat{Q}(s',a',\textcolor{red}{\omega^-})
$$

3. 使用随机梯度下降来更新当前的评价网络权值

$$
\begin{aligned}
\Delta{\omega}&={\alpha(\rm{TD\_target}-\hat{Q}(s,a,\omega))}\nabla_\omega\hat{Q(s,a,\omega)} \\
&={\alpha(r+\gamma\max_{a'}\hat{Q}(s',a',\textcolor{red}{\omega^-})-\hat{Q}(s,a,\omega))}\nabla_\omega\hat{Q(s,a,\omega)}
\end{aligned}
$$

4. 训练一定步数后，更新目标网络的参数$\omega^-$

$$
\omega^-=\omega\quad\rm{if\ {n\_step\ \%\ update\_frequency}==0}
$$

简要说明算法有效性的示意图如下：

- 使用猫（estimation）追老鼠（target）来表征两个网络

<center>
    <img src="/images/posts/blog/image-20210410174646116.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>

- 在固定之前，猫和老是参数一样，在状态空间内，猫很难追上老鼠

<center>
    <img src="/images/posts/blog/image-20210410174849192.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>

而固定后，可以看作猫比老鼠跑得快，更容易追上老鼠。

#### 15. DQN算法的总结（Summary on DQN）

1. DQN使用了两个技巧——经验重放和固定目标
2. 存储数据：将转移关系元组$(s_t,a_t,r_t,s_{t+1})$存入缓冲器容器$D$中（D可以是deque或者list）
3. 采样：在$D$中随机采样本batch大小的数据用于训练更新
4. 固定目标：两个网络，其中目标网络的参数更新较慢
5. 优化：使用随机梯度下降算法优化两个网络之间的MSE值

