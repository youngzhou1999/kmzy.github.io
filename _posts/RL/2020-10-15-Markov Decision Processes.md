---
layout: post
title: 马尔科夫决策过程的预测和控制
categories: RL
description: MDP's prediction and control
keywords: Reinforcement Learning
---

## MDP（Markov Decision Process）

> 1. 贝尔曼方程（Bellman Equation）

贝尔曼方程由MRP（Markov Reward Process）中得出，描述状态之间价值函数的计算关系：

$$
V(s)=R+\gamma\sum_{s'\in{S}}{P(s'|s)}V(s')
$$

此为强化学习的一个重要基石。

> 2. 贝尔曼期望方程（Bellman Expectation Equation）

在MDP中，引入了动作$a$，对应的策略函数$\pi$和动作-价值函数（Q函数），$\pi(s,a)$表示在当前状态$s$执行动作$a$的概率大小，$Q(s,a)$表示在当前状态$s$执行动作$a$可以获得价值。

Q函数与$V$函数的计算关系为：

$$
Q^{\pi}(s,a)=R_s^a+\gamma{\sum_{s'\in{S}}P(s'|s)V^{\pi}(s')}\tag{1}
$$

$$
V^{\pi}(s)=\sum_{a\in{A}}\pi(s,a)Q^{\pi}(s,a)\tag{2}
$$

贝尔曼期望方程由MDP中得出，描述状态之间价值函数，以及动作-价值函数的计算关系：

$$
V^{\pi}(s)=\mathbb{E}[R_{t+1}+\gamma{V^{\pi}(s_{t+1})}|s_t=s] \\
Q^{\pi}(s,a)=\mathbb{E}[R_{t+1}+\gamma{Q^{\pi}(s_{t+1},a_{t+1})|s_t=s,a_t=a}]
$$

利用式(1)和(2)可以得出$V(s)$与$V(s')$和$Q(s,a)$与$Q(s',a')$的计算关系，具体地：

- 将(1)代入(2)：

$$
V^{\pi}(s)=\sum_{a\in{A}}\pi{(a|s)}\bigg(R(s,a)+\gamma{\sum_{s'\in{S}}P(s'|s,a)V^{\pi}(s')}\bigg)
$$

- 将(2)代入(1)：

$$
Q^{\pi}(s,a)=R(s,a)+\gamma\sum_{s'\in{S}}P(s'|s,a)\sum_{a'\in{A}}\pi(a'|s')Q^{\pi}(s',a')
$$

> 3. 策略评估（Policy Evaluation）

使用贝尔曼期望方程来评价一个策略的价值。

利用策略$\pi$，迭代计算出状态价值函数，直至其值收敛。其计算公式为：

$$
V^{\pi}(s)=\sum_{a\in{A}}\pi{(a|s)}\bigg(R(s,a)+\gamma{\sum_{s'\in{S}}P(s'|s,a)V^{\pi}(s')}\bigg)
$$

当MDP达到稳态即上述过程收敛时，其表示该MDP被解决。此时最优状态价值函数表达式为：

$$
V^*(s)=\max_{\pi}V^{\pi}(s)
$$

其表示所有的策略$\pi$的状态价值的最大值。此时，最优策略表达式为：

$$
\pi^*{(s)}=\arg\max_{\pi}V^{\pi}(s)
$$

显然，最有状态价值函数是唯一的，但最优策略可能不唯一。

最优策略与最优动作-状态价值函数有密切的关系，我们可以通过最优动作-状态价值函数直接获得最优策略，具体地：

$$
\pi^*{(a|s)}=\begin{cases}1,\quad\rm{if\ a=arg\ max_{a\in{A}}Q^*(s,a)}\\
0,\quad\rm{otherwise}
\end{cases}
$$

因此，一旦知道了最优动作-价值函数，就可以得到最优策略。

> 4. 策略迭代（Policy Iteration）

策略迭代用于MDP的控制过程，策略迭代分为两个步骤，策略评估和策略提升。策略评估即上述的过程——利用策略$\pi$计算出状态价值函数。策略提升为贪心地选择当前最优的策略，即

$$
\pi'=\rm{greedy}(V^{\pi})
$$

策略迭代的一次迭代流程为：

$$
\pi\to{V}\to{Q}\to{\arg\max_{\pi}Q}\to\pi'
$$

策略迭代的计算表达式为：

- 策略评估——利用贝尔曼期望方程计算动作-价值函数

$$
Q^{\pi{i}}(s,a) = R(s,a)+\gamma\sum_{s'\in{S}}P(s'|s,a)V^{\pi{i}}(s')
$$

- 策略提升——贪心地从动作-价值函数中选取价值最大的动作作为新的策略

$$
\pi_{i+1}=\arg\max_{a}Q^{\pi{i}}(s,a)
$$

Shutton书中有关于此贪心的提升为单调递增的证明，具体见76页。

> 5. 贝尔曼最优方程（Bellman Optimality Equation）

当上述策略迭代过程收敛即达到稳态后，此时状态间满足的关系即为贝尔曼最优方程。其表达式为：

$$
V^*(s)=\max_{a}{Q^*(s,a)}\tag{1}
$$

$$
Q^*(s,a)=R(s,a)+\gamma\sum_{s'\in{S}}P(s'|s,a)V^*(s')\tag{2}
$$

将(1)代入(2)可以得到$Q^*(s,a)$和$Q^*(s',a')$的计算关系：

$$
Q^*(s,a)=R(s,a)+\gamma\sum_{s'\in{S}}P(s'|s,a)\max_a{Q^*(s',a')}
$$

同理，将(2)代入(1)也可以得到$V^*(s)$和$V^*(s')$的计算关系：

$$
V^*(s)=\max_a\bigg({R(s,a)+\gamma\sum_{s'\in{S}}P(s'|s,a)V^*(s')}\bigg)
$$

> 6. 价值迭代（Value Iteration）

由上述贝尔曼最优方程，我们可以直接得到$V^*(s)$和$V*(s')$的关系。价值迭代的原理为：直接利用此关系迭代价值函数，即假设MDP处于稳态。其迭代关系为：

$$
V(s)\gets{\max_{a}\bigg(R(s,a)+\gamma\sum_{s'\in{S}}P(s'|s,a)V(s')\bigg)}
$$

对于价值迭代，其不需要显示地计算策略，只有状态价值函数的迭代过程。如果需要最优策略时，只需要将上述迭代过程展开，即添加动作-状态函数，最后迭代过程结束后利用动作-价值函数即可得到最优策略，具体过程为：

```python
# 迭代过程
for k in range(1, H):
	for s in S:
		Q(s,a) = R(s,a) + gamma * sum_{s_} P(s_|s,a) * V(s_)
		V(s) = max_{a} Q(s,a)
# 获得最优策略
for s in S:
	pi(s) = arg max_{a} Q(s,a)
```

> 7. 总结（Summary）

- 策略迭代与价值迭代

两种方法都用于在已知MDP模型参数的情况下，求解MDP。

策略迭代分为两个步骤：策略评估 + 策略提升

价值迭代直接迭代计算状态价值函数，其步骤为：迭代价值函数 + 最后反解得到最优策略

- 算法与原理方程的对应关系

上述的策略评估，策略迭代和价值迭代都属于动态规划（Dynamic Programming）算法。其中，策略评估用于MDP的预测，而策略迭代和价值迭代用于MDP的控制。其与贝尔曼方程的关系如下：
<center>

| 问题 | 贝尔曼方程     | 算法         |
| ---- | -------------- | ------------ |
| 预测 | 贝尔曼期望方程 | 迭代策略评估 |
| 控制 | 贝尔曼期望方程 | 策略迭代     |
| 控制 | 贝尔曼最优方程 | 价值迭代     |
</center>