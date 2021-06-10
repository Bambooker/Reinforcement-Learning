Introduction

What is reinforcement learning and why we care?

![image-20210606104107692](../Images/image-20210606104107692.png)

A computational approach to learning whereby an **agent** tries to **maximize** the total amount of **reward** it receives while interacting with a complex and uncertain **environment**.

一个agent如何在一个复杂不确定的环境里面，去极大化它能获得的收益，并且是一直在交互的。agent从环境中获取状态，然后输出一个action，放到环境中去，环境会按照动作进行下一步，然后输出下一个状态。agent的目的就是尽可能地从环境中获得收益哦。

Supervised Learning: Image Classification

• Annotated images, data follows i.i.d distribution.

• Learners are told what the labels are.

![image-20210607092551107](../Images/image-20210607092551107.png)

对于监督学习，在图片分类的任务中，需要很多标定好的images（1号图片标定为车，2号图片标定为飞机……）并且data要服从iid分布（独立同分布）。然后设计一个分类器，当输入车的图片缺返回飞机，就告诉分类器这是错误的，然后把这个错误写成一个loss function，通过back-propagation训练分类器。要求数据尽可能的没有关联，不然不是很好进行学习；并且已经告知了输入的正确标签。

Reinforcement Learning: Playing Breakout

• Data are not i.i.d. Instead, a correlated time series data

• No instant feedback or label for correct action

![image-20210607092502355](../Images/image-20210607092502355.png)

![image-20210607092535312](../Images/image-20210607092535312.png)

强化学习：在打砖块的任务中，数据并不是独立的，上一帧的数据和下一帧的数据有非常强的连续性。另外，在任务进行中，agent并没有获得立刻的反馈，比如我现在把杆往右移，只能知道球会往左上反弹，但之后的情况就不得而知了。这也是强化学习的难点，没有得到很好的反馈，但依旧希望这个agent可以在这个环境中学习。RL的training data相当于玩游戏的过程，比如第一步往右移，第二部往左，training data就是一个序列。把序列输入到网络中，在当前的状态应该输出一个action如往右移，但是没有一个明确的label，说明这个action是正确还是错误的，需要等到游戏结束，才能知道结果。这就是面临了一个问题叫delayed reward，使得网络的训练十分困难。

Difference between Reinforcement Learning and Supervised Learning

• Sequential data as input (not i.i.d)

输入是不独立的有连续性的序列。

• The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them.

学习器并未被告知正确的行为是什么，学习器不得不自己去发现哪些行为对最终结果是有利的。

• Trial-and-error exploration (balance between exploration and exploitation)

agent在获得自身学习能力是通过不断的试错，exploration指不断尝试新的行为，新的行为可能获得更高的收益，也可能一无所获；exploitation指已经知道这个行为可以获得一定的收益，进行重复就可以了。两者需要权衡

• There is no supervisor, only a reward signal, which is also delayed

有一个收益信号，环境会在一定延迟之后给出信号告诉这个行为是否有效，有益。



Features of Reinforcement Learning
• Trial-and-error exploration
• Delayed reward
• Time matters (sequential data, non i.i.d data)
• Agent’s actions affect the subsequent data it receives
(agent’s action changes the environment)

需要在环境里探索、试错来获取对环境的理解，agent会从环境获得延迟的reward，时间非常的重要，agent输出的action会影响随后它得到的数据，在训练agent的过程中采数据会根据agent和环境的交互而获得。如果agent在某个时刻crash了，那么整个学习就失败了，所以如何让agent可以一直稳定地学习也是强化学习的一个重点。

Big deal: Able to Achieve Superhuman Performance

为什么我们关注强化学习，非常重要的一点是它可以获得超人类的结果，监督学习的data是人为标定的，这个算法的上限就是人类的上限。对于强化学习，它自己在环境中探索，有非常大的潜力获得超越人类的能力，比如阿尔法狗。



Examples of reinforcement learning

• A chess player makes a move: the choice is informed
both by planning-anticipating possible replies and
counterreplies.
• A gazelle calf struggles to stand, 30 min later it runs 36
kilometers per hour.
• Portfolio management.
• Playing Atari game

下象棋，股票交易，玩游戏都是一个强化学习的过程，

RL example: Pong

![image-20210607095628441](../Images/image-20210607095628441.png)

Reinforcement Learning:
Sample actions (rollout), until game is over, then penalize each action

在强化学习中，通过让agent尝试完成游戏，再去说前面一系列action是正确还是错误的。

rollout指从当前帧生成很多局游戏，让当前的网络和环境进行交互，得到一堆观测，观测可以看成一个轨迹。也就是有一个当前帧，然后有一个策略，得到一个最终结果。通过最后的结果和观测到的序列来训练agent，使得agent获得能力，可以尽可能地得到最好的reward



Deep Reinforcement Learning:
Deep Learning + Reinforcement Learning

将强化学习和深度学习结合起来

Traditional CV and deep CV 

![image-20210607100714000](../Images/image-20210607100714000.png)

![image-20210607100721582](../Images/image-20210607100721582.png)

4

![image-20210607100841345](../Images/image-20210607100841345.png)

![image-20210607100843708](../Images/image-20210607100843708.png)



传统RL：需要手工设计一些特征，然后训练价值函数，然后做出决策。

深度RL：改进成端对端的神经网络，只需要输入状态，通过神经网络拟合整个过程得到最后的action。

Why RL works now?

• Computation power: many GPUs to do trial-and-error rollout
• Acquire the high degree of proficiency in domains governed by
simple, known rules
• End-to-end training, features and policy are jointly optimized toward
the end goal.

计算能力增加，可以从环境中获得大量信息，以至于取得很高的收益，而且有了端到端的训练，可以让特征提取和决策网络一起优化。





## Introduction to Sequential Decision Making

## 时序决策过程

The agent learns to interact with the environment

强化学习研究的问题就是agent和环境的交互。agent在个环境不断地交互，agent把输出的动作给环境，环境进行下一步，将观测和得到的收益返还给agent。agent的目的就要从这些观测中学到能获得极大化收益的策略。

![image-20210607102450201](../Images/image-20210607102450201.png)

### Rewards

- A reward is a scalar feedback signal
- Indicate how well agent is doing at step t
- Reinforcement Learning is based on the maximization of rewards: All goals of the agent can be described by the maximization of expected cumulative reward.

RL定义了奖励函数，收益是环境给的反馈信号，这个信号表明了agent在某一步采取了某个动作后是否得到了收益，RL就是想要极大化这个收益。

Examples of Rewards
• Chess players play to win:
+/- reward for wining or losing a game
• Manage stock investment
+/- reward for each profit or loss in $
• Play Atari games
+/- reward for increasing or decreasing scores



Sequential Decision Making

Objective of the agent: select a series of actions to maximize total future rewards

Actions may have long term consequences

Reward may be delayed

Trade-off between immediate reward and long-term reward

agent的目的就是得到一系列动作，使得最终结果的收益最大化。某个动作可能会有长期的影响，收益也是有延迟的。RL的一个重点就是近期收益和长期收益的权衡，如何获得更多的长期收益。



• The history is the sequence of observations, actions, rewards.

<img src="../Images/image-20210607105754375.png" alt="image-20210607105754375" style="zoom:33%;" />

What happens next depends on the history
State is the function used to determine what happens next：<img src="../Images/image-20210607111420599.png" alt="image-20210607111420599" style="zoom:33%;" />

agent在采取当前动作时，会依赖于之前的观测，动作和收益，所以可以把整个任务的状态看成是历史的函数。

Environment state and agent state

<img src="../Images/image-20210607110303487.png" alt="image-20210607110303487" style="zoom: 33%;" />

agent内部也有一个函数更新状态，当环境和agent的状态

• **Full observability**: agent directly observes the environment state, formally as Markov decision process (MDP)

<img src="../Images/image-20210607110436159.png" alt="image-20210607110436159" style="zoom: 33%;" />

Partial observability: agent indirectly observes the environment, formally as partially observable Markov decision process (POMDP)

• Black jack (only see public cards), Atari game with pixel observation

当环境的状态和agent的状态等价时，我们说环境可以完全被观测到。但有些情况下，agent不能完全获得环境的状态，比如有些时候agent只能获得像素上的信息，而无法获得环境全部的信息



Major Components of an RL Agent

• Policy: agent’s behavior function

策略函数：用来选取下一步的动作





Policy

• A policy is the agent’s behavior model
• It is a map function from state/observation to action.
• Stochastic policy: Probabilistic sample <img src="../Images/image-20210607113404157.png" alt="image-20210607113404157" style="zoom: 33%;" />
• Deterministic policy: <img src="../Images/image-20210607113903842.png" alt="image-20210607113903842" style="zoom:33%;" />

policy决定了agent的行为，这个函数的输入是状态，输出是动作。分为两种：随机策略和决定性策略。

Stochastic policy（随机策略）

是一个从状态集S 到动作集A 的条件概率分布。输入一个状态s，所有的动作都有一个概率（如向上的概率为0.3，向下的概率为0.7）；然后对这个分布进行采样（sample），最后获得实际采取的动作。

Deterministic policy（决定性策略）

是一个状态集S 到动作集A 的映射。输入一个状态s，采取极大化（最有可能的概率），提前决定好所有的动作中有一个概率为1，即一直采取这个动作。



Value function



Expected discounted sum of future rewards under a particular policy


• Discount factor weights immediate vs future rewards

• Used to quantify goodness/badness of states and actions

<img src="../Images/image-20210607120943678.png" alt="image-20210607120943678" style="zoom: 50%;" />

• Q-function (could be used to select among actions)

<img src="../Images/image-20210608154445157.png" alt="image-20210608154445157" style="zoom:50%;" />

价值函数：对当前状态进行估价，估计在某状态下采取某动作会对之后的结果带来多大的影响，价值大说明进入这个状态采取这个动作越有利。进一步说，就是选择了一个策略后，采取某个动作在未来会获得多大的价值。折价因子：希望在尽可能短的时间内得到更多的收益，现在获得100元和10年后获得100元最终得到的价值是不一样的。所以对于不同时间的收益需要乘上相应的折价因子，最后再将他们相加得到价值函数。价值函数实际就是期望。

Model
A model predicts what the environment will do next
Predict the next state: <img src="../Images/image-20210608154803882.png" alt="image-20210608154803882" style="zoom:33%;" />
Predict the next reward: <img src="../Images/image-20210608154819495.png" alt="image-20210608154819495" style="zoom:33%;" />

模型决定了下一个状态是什么样的。包括两部分：状态是怎么转移的，以及下一个收益是多大。



Markov Decision Processes (MDPs)

马尔科夫决策过程



Types of RL Agents based on What the Agent Learns

• Value-based agent:

<img src="../Images/image-20210608160353095.png" alt="image-20210608160353095" style="zoom:50%;" />

​	Explicit: Value function

​	Implicit: Policy (can derive a policy from value function)



基于价值函数的agent直接学习价值函数，隐式地学习策略，策略是从价值函数推算出来的。

Policy-based agent:

<img src="../Images/image-20210608160406820.png" alt="image-20210608160406820" style="zoom:50%;" />

• Explicit: policy
• No value function

基于策略的agent直接学习策略函数。

• Actor-Critic agent:
• Explicit: policy and value function



Types of RL Agents on if there is model
•Model-based
• Explicit: model
• May or may not have policy and/or value function
•Model-free
• Explicit: value function and/or policy function
• No model.

agent是否学习了环境模型也可以分为两类。



Exploration and Exploitation
• Agent only experiences what happens for the actions it tries!
• How should an RL agent balance its actions?



Exploration: trying new things that might enable the agent to make better decisions in the 	future.

Exploration就是不断的试错，判断是否会有更好的表现

• Exploitation: choosing actions that are expected to yield good reward given
the past experience

Exploitation就是直接采取已知的还不错的收益，

• Often there may be an exploration-exploitation trade-off
• May have to sacrifice reward in order to explore & learn about potentially
better policy

• Restaurant Selection
• Exploitation: Go to your favourite restaurant
• Exploration: Try a new restaurant
• Online Banner Advertisements
• Exploitation: Show the most successful advert
• Exploration: Show a different advert
• Oil Drilling
• Exploitation: Drill at the best-known location
• Exploration: Drill at a new location
• Game Playing
• Exploitation: Play the move you believe is
• Exploration: play an experimental move



