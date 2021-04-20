---
ayout: post
title: DQN实现例子
categories: RL
description: An Example of DQN
keywords: Reinforcement Learning
---

## Example of DQN in Pytorch

> 1. 回顾DQN原理

DQN使用神经网络来拟合动作-状态价值函数即$Q$函数，同时为了使训练效果更稳定，加入了经验重放和固定目标机制。经验重放指的是每次将转移元组$(s_t,a_t,r_t,s_{t+1})$存入缓冲器容器$D$中，然后从容器中随机采样，用作训练数据；固定目标是指使用两个网络，一个用作目标网络，一个用作当前的评估网络，目标网络的参数更新滞后于评估网络，使得评估网络更好地接近目标网络。

> 2. DQN实现

首先，需要导入包，具体作用见注释。

```python
import torch as T	# 导入Pytorch
import torch.nn as nn	# 导入神经网络模块
import torch.nn.functional as F		# 导入激活函数模块
import torch.optim as optim		# 导入优化器			
import numpy as np	# 导入向量处理模块
```

DQN的实现大致可以分为3部分：

1. 神经网络部分
2. 缓冲容器部分
3. 智能体（agent）部分

#### 1. 神经网络部分

在Pytorch中，必须显示地实现神经网络地前向传播数据函数，而梯度回传可以直接使用`loss.backward()`自动完成。因此在此部分，我们做两件事：

1. 显式地定义网络结构
2. 实现前向传播`forward()`函数

其代码如下：

```python
class DeepQNetwork(nn.Module):
    def __init__(self, lr , input_dims, fc1_dims, fc2_dims, n_actions):
        '''
        initilize the deep q network which has 2 fully connected(fc)   layers and take an input as the number of states and ouput the   value of each action.  
        
        Args:
        	lr : learning rate
        	input_dim : the  number of states
        	fc1_dims : the number of neurons in the first fc layer
        	fc2_dims : the number of neurons in the second fc layer
            n_actions : the number of actions 
        '''
        super(DeepQNetwork, self).__init__()
        # store args
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # build network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
		# build optimizer, loss function and device
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        T.cuda.current_device()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        calculate the ouput of the deep q network
        
        Args:
        	state : current state as input
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
```

由`__init__()`函数可以看到，此网络由两层全连接层和一层输出层构成，使用`Adam`作为优化器，`MSE`作为损失函数，在有`GPU`的情况下，支持使用`cuda`加速运算。

【填坑Adam】

而在`forward()`函数中，使用了`relu`函数作为层激活函数。

【填坑relu】

#### 2. 智能体部分

从设计模式中，智能体与重放容器可以分开也可以合并，因为此处使用列表来表示该容器，故操作时较简单，因此与智能体合并。

回到智能体部分，智能体需要具备以下三个函数：

1. 初始化参数和重放容器
2. 存储历史数据至容器
3. 根据Q函数网络选取动作
4. 训练，更新参数

> 1. 初始化部分

初始函数`__init__()`代码如下：

```python
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size=100000, eps_end = 0.05, eps_dec = 5e-4, target_update_freq = 100):
        """
        initialize the agent
       	
       	Args:
       		gamma : the discounted factor in Bellman Equation
       		lr : learning rate
       		input_dims : the number of states
       		batch_size : the size of one batch
       		n_actions : the number of actions
       		max_mem_size : the size of replay buffer
       		eps_end : the minimum value of epislon in epsilon-greedy trick
       		eps_dec : the value of epsilon decay after each step(linear decay)
       		target_freq : the period for updating the parameters of the target network
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
		# get action space
        self.action_space = [i for i in range(n_actions)]   # for epsilon-greedy

        self.batch_size = batch_size
        self.mem_size = max_mem_size
        # the current index in replay buffer
        self.mem_cntr = 0
        # the counter of steps
        self.iter_cntr = 0
        self.target_update_freq = target_update_freq
		# evaluation network
        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, 
                            fc1_dims=256, fc2_dims=256, n_actions=n_actions)
		# target network
        self.Q_target = DeepQNetwork(self.lr, input_dims=input_dims,
                            fc1_dims=256, fc2_dims=256, n_actions=n_actions) 
		# replay buffer(use list for every element)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.terminate_memory = np.zeros(self.mem_size, dtype=np.bool)

```

由`__init()`函数可以看到：

1. 此智能体对于转移元组的每一元素都有一个重放容器，这样便于分开管理，而不用记录同一元组的下标顺序
2. 为了方便训练，存储了“完成标记”（done），用于定义终态的价值为0
3. 含有两个网络，即评估网络和目标网络
4. 含有两个计数指针，一个用于定位回放容器，另一个用于记录智能体与环境交互次数，以更新目标网络

> 2. 存储历史数据至容器

此函数`store_transitions()`代码如下：

```python
    def store_transition(self, state, action, reward, state_, terminate):
        """
        store tuple (s_t,a_t,r_t,s_{t+1},done) to replay buffer
        
        Args:
        	state : s_t
            action : a_t
            reward : r_t
            state_ : s_{t+1}
            terminate : done
        """
        index =  self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminate_memory[index] = terminate
        
        self.mem_cntr += 1
```

此函数实现标记简单，只需将容器计数指针`mem_cntr % mem_size`处写入数据即可，完成后，将指针后移一位。由于指针数值可能会超过容器大小，因此需要使用模`%`运算来覆盖掉之前写入的数据。

> 3. 根据Q函数网络选取动作

DQN是基于Q-learning，因此使用的是epsilon-greedy策略来选取动作，故此函数代码如下：

```python
    def choose_action(self, observation):   # obs = state
        """
        choose an action in current state by using epsilon-greedy
        
        Args:
        	observation : current state
        """
        if np.random.random() > self.epsilon:
            # choose the best action
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions =   self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
             action = np.random.choice(self.action_space)

        return action
```

由上述`choose_action()`函数，以$\varepsilon$的概率选择随机一个动作，$1-\varepsilon$的概率选取当前价值最高的价值。具体做法为：

1. 随机选择动作时，只需在动作空间采样即可
2. 选择最有动作时，首先将当前状态代入评估网络，得到每个动作价值，然后取输出值最大的动作即可

> 4. 训练，更新参数

此为智能体的核心函数，用于更新网络参数，以达到学习的效果，其代码如下所示：

```python
    def learn(self):
        """
        sample data of size batch_size from replay buffer, and the ues   them to update the network(s)
        """
        # when data in replay buffer less than batch, just ignore
        if self.mem_cntr < self.batch_size:
            return
		# release all old gradients at first
        self.Q_eval.optimizer.zero_grad()
		
        max_mem = min(self.mem_size, self.mem_cntr)
        # sample data
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # get batch space
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # get all the data from replay buffer
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]    # no need to conv
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminate_memory[batch]).to(self.Q_eval.device)
		
        # take these data to the evaluation network and get q(s,a)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]    # choose sampled action
        # get the target value using s_{t+1} 
        q_next = self.Q_target.forward(new_state_batch)
        # terminal states has a value of 0
        q_next[terminal_batch] = 0.0
		# build q_target : r + max_{a}(Q_target(s_{t+1}, a))
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
		# build q-error : q_target - q(s,a) 
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        # uodate q_eval network
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                            else self.eps_min
        
        # periodly update q_target network by q_eval      
        if self.iter_cntr % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
```

由上述`learn()`函数代码可以看到，首先，采样`batch`数据，然后根据以下更新式来更新网络：

$$
\begin{aligned}
\Delta{\omega}&={\alpha(\rm{TD\_target}-\hat{Q}(s,a,\omega))}\nabla_\omega\hat{Q(s,a,\omega)} \\
&={\alpha(r+\gamma\max_{a'}\hat{Q}(s',a',\textcolor{red}{\omega^-})-\hat{Q}(s,a,\omega))}\nabla_\omega\hat{Q(s,a,\omega)}
\end{aligned}
$$

Adam中已经使用了学习率$\alpha$，而梯度值由Pytroch自动计算，故我们只需要计算q-error即可。

#### 3. 环境测试

使用`gym`的`lunarlander-v2`环境来测试算法的正确性，在这之前，需要下载相应的包：

```python
pip install gym
pip install box2d
```

根据官方文档，该环境可以描述如下：

着陆板始终位于坐标（0,0）。坐标是状态向量中的前两个数字。从屏幕顶部移动到着陆点的奖励且速度为0时约为100..140点。如果着陆器远离着陆垫，它将失去奖励。如果着陆器坠毁或静止下来，则当前幕结束，并获得额外的-100或+100点。每条腿的接地触点为+10点。点火主引擎每帧为-0.3分。解决的奖励视作200分。可以在着陆垫外面着陆。燃料是无限的，因此特工可以学会飞行，然后首次尝试着陆。共有四个离散操作：不执行任何操作，左向发动机点火，主发动机点火，右发动机点火。

代码如下：

```python
import gym
from dqn import Agent
import numpy as np
from tensorboardX import SummaryWriter

env_name = 'LunarLander-v2'

if __name__ == '__main__':
    # initialize env, agent and some infomation
    env = gym.make(env_name)
    agent = Agent(gamma=0.99, epsilon=1.0, lr=1e-4, input_dims=env.observation_space.shape[0], batch_size=64,
                    n_actions=env.action_space.n, eps_end=0.01)
    writer = SummaryWriter(comment="-" + env_name)
    scores, eps_history = [], []
    target_score = 250
    avg_score = 0
    episode = 0
	# training
    while avg_score < target_score:
        score = 0
        done = False
        # get first state
        observation = env.reset()
        while not done:
            # choose action and interact with env
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            # store transitions
            agent.store_transition(observation, action, reward, 
                                        observation_, done)
            agent.learn()
            observation = observation_  # important
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        episode += 1
        writer.add_scalar("reward", score, episode)
        writer.add_scalar("avg_reward", avg_score, episode)

        print('episode:', episode, 'score %.2f' % score,
                'average score %.2f' %avg_score,
                'epsilon %.2f' %agent.epsilon)
    writer.close()

```

在主函数中，智能体每一步都选择一个动作，并于环境交互产生转移元组，同时更新网络参数，直到成功解决该环境。

训练的结果如下：

平均回报：

<center>
    <img src="/images/posts/blog/image-20210414153828838.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>
每一幕的回报：

![image-20210414153911852](/image-20210414153911852.png)
<center>
    <img src="/images/posts/blog/image-20210414153911852.png" alt="picture not found" style="zoom:100%;"/>
    <br>
</center>