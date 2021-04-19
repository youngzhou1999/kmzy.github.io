---

layout: post
title: 无模型环境的评估和控制
categories: RL
description: Model free's prediction and control
keywords: Reinforcement Learning

---

## Model Free Method

上述的求解MDP的方法都需要知道MDP的环境参数，即转移矩阵$P$和收益信号$R$。但实际上很难知道MDP的参数或计算时很复杂，为此，需要考虑Model-free的方法来求解。

#### 1. Model-free模型的预测

Model-free模型的预测：在不知道模型参数的情况下，估计给定当前策略$\pi$下的期望的回报。主要有以下两种方法：

- 蒙特卡洛策略估计
- 时序差分学习（Temporal-Difference-learning）

> 1. 蒙特卡洛策略评估（Monte-Carlo Policy Evaluation）

蒙特卡洛策略评估算法的主要思想为：通过采样来估计期望回报。具体地，通过智能体与环境不断交互来产生轨迹，计算出每个轨迹的实际回报，最后取平均，计算方式如下：

$$
V^{\pi}(s)=\mathbb{E}_{\tau\sim{\pi}}[G_t|s_t=s]
$$

其中，$\tau$为从策略$\pi$中采样产生的样本轨迹，$G_t$为时序累计衰减的收益和，其表达式为：

$$
G_t=R_t+\gamma{R_{t+1}}+\cdots=\sum_{k=t}^{\infin}\gamma^{k}R_k
$$

蒙特卡洛策略评估有几点不足：

- 首先，蒙特卡洛策略评估只适用带幕（episode）的MDP模型，因为其采样需要一个终态（terminal）
- 近似性：蒙特卡洛策略评估使用的是实验的平均回报，而非理论上的数学期望

当然，这个算法也有几点优势：

- 首先，它不需要知道MDP的动态特征，也不需要估计状态价值函数（因为是纯采样的算法）
- 它没有迭代的过程，不需要自举（no bootstrapping）

蒙特卡洛策略评估的算法框架如下：

```python
# 预测/评估 V(s)
1. To evaluate V(s)
	1. Every time-step t that state s is visited in an episode
	2. Increment counter N(S) <- N(s) + 1
	3. Increment total return S(s) <- S(s) + G_t
	4. Value is estimated by mean return v(s) = S(s) / N(s)
```

由大数定理，当$N(s)\to\infin$时，$V(s)\to{V^{\pi}(s)}$。

> 2. 增长均值（Incremental Mean）

增长均值是一个很重要的技巧，它把累计的均值转变为当前的均值与上次的均值之间的计算关系。其推导的关系如下：

$$
\begin{aligned}
\mu_t&=\frac{1}{t}\sum_{j=1}^{t}x_j \\
&=\frac{1}{t}(\sum_{j=1}^{t-1}x_j+x_t) \\
&=\frac{1}{t}\big((t-1)\mu_{t-1}+x_t\big) \\
&=\mu_{t-1}+\frac{1}{t}(x_t-\mu_{t-1})
\end{aligned}
$$

利用增长均值，可以得到增长版的蒙特卡洛策略估计算法，其流程如下：

```python
# incremental MC evaluation
1. collect spisode (s1,a1,r1,..., s_t)
2. use computed G_t to update s:
    N(s_t) = N(s_t) + 1
    V(s_t) = V(s_t) + 1 / N(s_t) (G_{i,t} - v(s_t))
```

进一步地，可以把$\frac{1}{n}$替换为一个超参数$\alpha$，即为学习率或步长，替换后的更新过程如下：

$$
V(s_t)=V(s_t)+\alpha(G_{i,t}-V(s_t))
$$

> 3. 动态规划算法与蒙特卡洛算法用于策略评估的比较（Comparison Between DP and MC）

- 不同之处

动态规划算法利用贝尔曼期望方程估计状态价值函数，需要自举（bootstrapping），且需要知道MDP的参数：

$$
V_i(s)\gets{\sum_{a\in{A}}\pi(a|s)\bigg(R(s,a)+\gamma\sum_{s'\in{S}}P(s'|s,a)V_{i-1}(s')\bigg)}
$$

蒙特卡洛算法通过采样，使用增长均值来估计状态价值函数：

$$
V(s_t)=V(s_t)+\alpha(G_{i,t}-V(s_t))
$$

- MC算法的优势

首先，MC不需要知道MDP的参数也可以工作，它是纯采样的一种算法；其次，MC的计算更简单，特别是当MDP很大的时候，状态数和转移矩阵$P$会变得很大，计算复杂；最后，MC算法在估计单个状态的价值函数时，不会影响到其他状态的价值函数，换句话说，计算一个状态的价值函数和其他状态是独立的，因为它不需要自举。

> 4. TD-learning

时序差分学习（TD-learning）是介于DP和MC之间的一种算法，其有几个特点：

- 直接从经历过的幕（episode）中学习
- 属于Model-free方法，不需要知道MDP参数
- 通过自举从不完整的幕中学习（在线学习）

TD-learning算法的框架如下：

1. 目标（Objective）

从经验（experience）中学习在策略$\pi$下的状态价值函数$V^{\pi}$

2. 最简单的TD-learning TD(0)：表示仅仅向前迭代一步，对比MC是走完整个幕

使用向前走了一步的收益来更新状态价值函数，计算方式如下：

$$
V(s_t)\gets{V(s_t)+\alpha\bigg(\textcolor{red}{R_{t+1}+\gamma{V(s_{t+1})}}-V(s_t)\bigg)}
$$

对比MC算法，其更新方式为：

$$
V(s_t)\gets{V(s_t)+\alpha\bigg(\textcolor{red}{G_{i,t}}-V(s_t)\bigg)}
$$

其中$i$表示为第$i$幕。

3. TD-target：为向前走一步的衰减的、自举的收益，即$R_{t+1}+\gamma{V(s_{t+1})}$

4. TD-error：为向前走收益于当前状态价值函数的差，即$R_{t+1}+\gamma{V(s_{t+1})}-V(s_t)$

> 5. TD-learning和MC的比较（Comparison Between TD-learning and MC）

- TD-learning可以在线学习，只要有采集到的数据即可以更新，而MC必须等到整个幕结束才能更新
- TD-learning可以从不完整的序列来学习，而MC不可以，其必须要到达幕的终态
- TD-learning可以在连续的（non-terminating）环境工作，而MC只能在分幕（episodic）的环境工作
- TD-learning显示出了马尔可夫性质（Markov Property），在马尔可夫性的环境中更有效；而MC没有显示马尔可夫性，故在非马尔可夫性的环境中更有效

> 6. n步时序差分学习（n-step TD-learning）

上述的TD-learning是最简单的一种情况，即往前走一步。$n$步TD-learning为往前走$n$步，其介于TD(0)与MC之间，我们可以通过控制$n$来实现到TD(0)与MC的转变。

1. $n$步回报$G_t^n$

首先考虑$n=1$，即$TD(0)$，其回报为$G_t^1=R_{t+1}+\gamma{V(s_{t+1}})$

考虑$n=2$，此时回报为$G_t^2=R_{t+1}+\gamma{R_{t+2}+\gamma^2V(s_{t+2})}$

类似于此，我们得到$n$步回报的定义如下：

$$
G_t^n=R_{t+1}+\gamma{R_{t+2}}+\gamma^2{R_{t+3}}+\cdots+\gamma^{n-1}{R_{t+n}}+\gamma^n{V(s_{t+n})}
$$

前$n-1$项表示样本的回报，第$n$项表示自举之前的状态价值函数。特别地，当$n\to\infin$时，$n$步回报即变为MC中的累计回报$G_t$。

2. 利用$G_t^n$更新状态价值函数

基于上述的$n$步回报$G_t^n$，可以得到$n$步TD的更新表达式如下：

$$
V(s_t)\gets{V(s_t)+\alpha\big(\textcolor{red}{G_t^n}-V(s_t)\big)}
$$

> 7. 三种算法的比较（Comparison Between DP, MC and TD）

- 是否需要自举和采样

|               | DP          | MC          | TD                 |
| ------------- | ----------- | ----------- | ------------------ |
| Bootstrapping | YES（pure） | NO          | YES（combination） |
| Sampling      | NO          | YES（pure） | YES（combination） |

- 更新计算表达式

    - DP

        $$
        V(s_t)\gets{\mathbb{E_{\pi}}[G_t|s_t=s]}={\mathbb{E_{\pi}}[R_{t+1}+\gamma{V(s_{t+1})}|s_t=s]}
        $$

    - MC

    $$
    V(s_t)\gets{V_{S_t}+\alpha(G_{i,t}-V(s_t))}
    $$

    - TD-learning（TD(0)）

    $$
    V(s_t)\gets{V(s_t)+\alpha\Big(R_{t+1}+\gamma{V(s_{t+1})}-V(s_t)\Big)}
    $$

#### 2. Model-free模型的控制

在model-free的模型下，没有收益信号$R$和转移矩阵$P$，而之前两种MDP的控制算法即策略迭代和价值迭代都依赖与这两个变量，因此这两种算法不能在model-free的模型下使用。

> 1. 广义策略迭代（Generalized Policy Iteration）

广义策略迭代包括两个同时进行的相互作用的流程，一个做策略评估，一个做策略提升，但两者可以以任意的流程交替进行。当然，在强化学习中，广义策略迭代值代所有让策略评估和策略提升相互作用的一般思路。其示意图如下：
<center>
    <img src="/images/posts/blog/image-20210409151640707.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>

以蒙特卡洛版本为例，广义策略迭代基于动作-状态价值函数，来做策略迭代，其包含两个过程：

1. 策略评估：将平均回报$G$作为动作-状态价值函数的估计值，即$Q=\bar{G}$
2. 策略提升：贪心选取时Q函数最大的动作，即$\pi(s)=\arg\max_{a}Q(s,a)$

这种算法收敛的一个条件是在环境中有探索起点（exploring starts），算法框架如下：

```python
# Monte Carlo ES
Loop forever(for each episode):
    choose s0 in S, a0 in A randomly such all pairs(s,a) have prob > 0
    generate trajectory(episode) from(s0, a0)
    G <- 0
    for each step in epsiode, t = T-1, T-2 , ... , 0:
        G <- gamma * G + R_{t+1}
        unless pair (s_t, a_t) appears in trajectory:
            append G to Returns(s_t, a_t)
            Q(s_t, a_t) = avg(Returns(s_t, a_t))
            pi(s_t) = argmax_a(Q(s_t, a))
```

> 2. $\varepsilon$-greedy

$\varepsilon$-greedy是强化学习一个重要的技巧，它让搜索未知过程在探索与利用当前最优中得到一定的平衡。其表达形式如下：

$$
\pi(a|S)=
\begin{cases}
\varepsilon/|A|+1-\varepsilon,\quad\rm{if\ a^*=\arg\max_{a\in{A}}Q(s,a)}\\
\varepsilon/|A|,\quad\rm{otherwise}
\end{cases}
$$

意义为：以概率$\varepsilon$选取随机的动作，以概率$1-\varepsilon$选取当前最优动作。

$\varepsilon$-greedy的提升是单调的，即可以满足$V^{\pi'}(s)\ge{V^\pi}(s)$对任意策略成立。具体证明见Shutton书第100页。

> 3. 广义蒙特卡洛迭代（Generalized Monte Carlo iteration）

借助$\varepsilon$-greedy，可以得到广义蒙特卡洛迭代算法，其框架如下：

```python
# Monte Carlo With epsilon-greedy exploration
Q(s,a) = 0, N(s,a) = 0 for all s in S, a in A
eps = 1 
k = 0
pi_k = eps_greedy(Q)

def eps_greedy(Q):
    if np.random.random <= eps:
        actions = np.random.choice(n_actions)
    else:
        actions = np.argmax(Q, dim=1)
    return actions

loop:
    sample k-th episode(s1,a1,r2,...,s_T) ~ pi_k
    for each s_t and a_t in the episode:
        N(s_t,a_t) += 1
        Q(s_t,a_t) = Q(s_t,a_t) + 1 / N(s_t,a_t) * (G_t - Q(s_t,a_t))
    k += 1
    eps = 1/k
    pi_k = eps_greedy(Q)
```

> 4. 蒙特卡洛与时序差分在预测和控制上的比较（Comparison Between MC and TD on Prediction and Control）

相对于MC，TD-learning有几点优势：

1. 更低的方差
2. 在线学习的特性
3. 对于不完整的序列（sequences）也能工作

因此，通常利用TD来求解MDP控制问题更优。

> 5. 广义TD-learning（Generalized TD-learning）

广义的TD-learning即为将传统的TD-learning应用到动作-状态价值函数即Q函数，同时使用$\varepsilon$-greedy做策略提升。根据TD-target/TD-error的选取不同，有两种常见的算法：

1. Sarsa算法
2. Q-learning算法

> 6. Sarsa

SARAS算法的原理为：使用$\varepsilon$-greedy执行一步，并自举Q函数。其过程为：

$$
\rm{state}\to\rm{action}\to\rm{reward}\to\rm{state}\to\rm{action}
$$

这也是其名字的由来。考虑其最简单的形式即$Sarsa(0)$，其更新表达式如下：

$$
Q(s_t,a_t)\gets{Q(s_t,a_t)}+\alpha\Big(\textcolor{red}{R_{t+1}+\gamma{Q(s_{t+1},a_{t+1})}}-Q(s_t,a_t)\Big)
$$

可以看到，SARSA算法的TD-target为$\delta_{t}=R_{t+1}+\gamma{Q(s_{t+1,a_{t+1}})}$。Sarsa算法的计算框架如下：

```python
# Sarsa
Q(s, a) = any for all s in S, a in A
Q(s_t, ) = 0	# Q terminal state = 0
def eps_greedy(s):
    if np.random.random <= eps:
        action = np.random.randint(n_actions)
    else:
        action = argmax(Q(s,))
    return action
Repeat(for each episode):
    initialize s
	a = eps_greedy(s)
    Reapeat(for each step in episode):
        take action a, get reward R and next state s_
        a_ = eps_greedy(s_)
        # update
        Q(s, a) <- Q(s, a) + alpha * (R + gamma * Q(s_, a_) - Q(s, a))
        s = s_
        a = a_
    until s is terminal
```

- n-step Sarsa

考虑执行$n$步的情况，分析过程与MC类似。

$n=1$时，即为上述的Sarsa(0)，有$q_t^1=R_{t+1}+\gamma{Q(s_{t+1},a_{t+1})}$

$n=2$时，同理有$q_t^2=R_{t+1}+\gamma{R_{t+2}}+\gamma^2{Q(s_{t+2},a_{t+2})}$

因此，可以得到执行$n$步时的Q函数回报为：

$$
q_t^n=R_{t+1}+\gamma{R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n{Q(s_{t+n},a_{t+n})}}
$$

前面$n-1$项表示累计的衰减的回报，第$n$项表示自举的Q函数。特别地，当$n\to\infin$时，上式转变为MC的期望回报$G$。因此，可以得到$n$步Sarsa的计算表达式为：

$$
Q(s_t,a_t)\gets{Q(s_t,a_t)}+\alpha\Big(\textcolor{red}{q_t^n}-Q(s_t,a_t)\Big)
$$

> 7. 在轨学习和离轨学习（On-policy and Off-policy Learning）

在轨学习（on-policy learning）指从策略$\pi$收集到的经验来学习策略$\pi$，只有一个策略。在轨学习通过表现不是最优，即它的表现会相对保守，因为它会去探索所有可能的动作，所以需要使用$\varepsilon$-greedy算法来逐渐减小探索的可能性。

离轨学习（off-policy learning）有两个不同策略，一个是目标策略$\pi$，其表示正在训练的策略，它最终会变为最优的策略；另一个是行为策略$\mu$，其表示相对更具有探索性的策略，用于产生轨迹。算法思想为：从策略$\mu$产生的样本中学习策略$\pi$。其执行过程为：

$$
\rm{follow\ \mu(a|s)\ to\ collect\ data:\ }(s_1,a_1,r_2,\cdots,s_t)\sim\mu\\
\rm{upudate\ \pi\ using\ (s_1,a_1,r_2,\cdots,s_t) }
$$

离轨学习有几点优势：

1. 在探索策略即$\mu$的引导下，目标策略$\pi$会达到最优策略
2. 可以从人类或其他智能体的观测中学习（即把他们当成行为策略$\mu$）
3. 可以重复使用旧策略产生的轨迹

> 8. Q-leaning

Q-learning是一种离轨学习算法，它的灵感/原理来自于贝尔曼最优方程，即区别于Sarsa，它在第二次选取动作时，不去真正地执行动作，而是假想一个动作来执行。它的表达式如下：

$$
Q(s_t,a_t)\gets{Q(s_t,a_t)+\alpha{\Big(R_{t+1}+\gamma{Q(s_{t+1},a')}-Q(s_t,a_t)\Big)}}
$$

其中，$a'\sim\pi(\.|s_t)$。

在实际使用TD-learning时，需要让行为策略和目标策略都有提升，因此可以对目标策略做贪心，即

$$
\pi{(s_{t+1})}=\arg\max_{a'}Q(s_{t+1},a')
$$

而对于行为策略，可以让它完全随机，但是一种更好的方式是使用$\varepsilon$-greedy算法来平衡探索与利用。因此，Q-learning的计算式可以表示如下：

$$
\begin{aligned}
Q(s_t,a_t)&\gets{Q(s_t,a_t)+\alpha{[R_{t+1}+\gamma{Q\Big(s_{t+1},\arg\max_{a'}Q(s_{t+1},a')\Big)}-Q(s_t,a_t)]}}\\
&\gets{Q(s_t,a_t)+\alpha{\Big(\textcolor{red}{R_{t+1}+\gamma\max_{a'}Q(s_{t+1},a')}-Q(s_t,a_t)}\Big)}
\end{aligned}
$$

上式中红色的部分即为Q-learning的TD-target。Q-learning的算法框架如下所示：

```python
# Q learning
Q(s, a) = any for all s in S, a in A
Q(s_t, ) = 0
def eps_greedy(s):
    if np.random.random <= eps:
        action = np.random.randint(n_actions)
    else:
        action = np,argmax(Q(s, ), dim=1)
    return action

Repeat(for each episode):
    initialize s
    a = eps_greedy(s)
    take action a, observe reward r and next satte s_
    # update
    Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s_, ), dim=1) - Q(s, a))
    s = s_
until s is terminal
```

> 9. Sarsa算法与Q-learning算法的比较（Comparision Between Sarsa and TD-learning）

- 在轨学习和离轨学习

Sarsa属于在轨学习算法，它正在执行的策略就是需要优化的策略。其算法流程为：

1. 利用$Q$函数使用$\varepsilon$-gereedy在状态$s_t$处采取动作$a_t$，并观察回报$R_{t+1}$和下一状态$s_t$
2. 利用$Q$函数使用$\varepsilon$\greedy在状态$s_{t+1}$采取动作$a_{t+1}$，并更新Q函数：

$$
Q(s_t,a_t)\gets{Q(s_t,a_t)}+\alpha\Big(\textcolor{red}{R_{t+1}+\gamma{Q(s_{t+1},a_{t+1})}}-Q(s_t,a_t)\Big)
$$

Q-learning属于离轨学习算法，它有两个策略：执行策略和目标策略。其算法流程为：

1. 利用$Q$函数使用$\varepsilon$-gereedy在状态$s_t$处采取动作$a_t$，并观察回报$R_{t+1}$和下一状态$s_t$（与Sarsa同）
2. 假想一个贪心的动作$a_{t+1}=\arg\max_{a}Q(s_{t+1},a)$，并利用整个假想动作更新Q函数：

$$
Q(s_t,a_t)\gets{Q(s_t,a_t)}+\alpha\Big(\textcolor{red}{R_{t+1}+\gamma\max_{a}{Q(s_{t+1},a)}}-Q(s_t,a_t)\Big)
$$

两种算法的迭代备份图如下：
<center>
    <img src="/images/posts/blog/image-20210409151111429.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>


> 10. DP算法和TD算法在无模型MDP的总结（Summary Betwwen DP and TD in Model-free MDP）

对于DP算法，在无模型MDP中，传统的策略迭代算法无法再适用，因此使用Q函数来做广义策略迭代。

- 预测

预测仍然是使用迭代策略评估算法，即使用贝尔曼期望方程，其表达式如下：

$$
V(s)\gets{\mathbb{E}[R+\gamma{V(s')}|s]}
$$

- 控制

使用Q函数可以做广义的策略迭代和价值迭代。对于策略迭代，其表达式如下：

$$
Q(s,a)\gets{\mathbb{E}[R+\gamma{Q(s',a')}|s,a]}
$$

对于价值迭代，其表达式如下：

$$
Q(s,a)\gets{\mathbb{E}[R+\gamma\max_{a'}{Q(s',a')}|s,a]}
$$

对于TD算法：

- 预测

预测时，TD算法介于MC和DP之间，只往前走$n$步，考虑$n=1$的情况，其表达式为：

$$
V(s)\gets^\alpha{R+\gamma{V(s')}}
$$

- 控制

同样，使用Q函数做广义的控制，基于贝尔曼期望方程和贝尔曼最优方程，衍生出两种算法。第一种为在轨学习的Sarsa算法，其更新表达式为：

$$
Q(s,a)\gets^{\alpha}{R+\gamma{Q(s',a')}}
$$

第二种为离轨学习的Q-learning算法，其更新表达式为：

$$
Q(s,a)\gets^{\alpha}{R+\gamma\max_{a'}{Q(s',a')}}
$$

其中，$x\gets^{\alpha}y$表示$x=x+\alpha(y-x)$。

时序差分学习随搜索深度与宽度变化情况如下图所示：
<center>
    <img src="/images/posts/blog/image-20210409153454873.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>

可以看到，当深度加深后，TD逐渐演化为MC算法；当宽度增加后，TD逐渐演化为DP算法。