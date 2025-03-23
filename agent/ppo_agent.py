import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOMemory:
    """PPO记忆缓冲区"""
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.batch_size = batch_size

    def store(self, state, action, prob, val, reward, done, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.action_masks = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions)
        old_probs = torch.tensor(self.probs)
        vals = torch.tensor(self.vals)
        rewards = torch.tensor(self.rewards)
        dones = torch.tensor(self.dones)
        action_masks = torch.stack(self.action_masks)
        
        return states, actions, old_probs, vals, rewards, dones, action_masks, batches

class ActorNetwork(nn.Module):
    """策略网络（Actor）"""
    def __init__(self, input_channels, height, width, num_actions):
        super(ActorNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 计算CNN输出特征的维度
        cnn_output_size = 64 * height * width
        
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, state, action_mask):
        # 通过CNN提取特征
        x = self.cnn(state)
        
        # 策略网络输出动作概率
        action_logits = self.actor(x)
        
        # 应用动作掩码（极小值将使softmax后的概率接近0）
        action_logits = action_logits.masked_fill(action_mask == 0, float('-1e20'))
        
        # 使用softmax获取概率分布
        probs = nn.functional.softmax(action_logits, dim=-1)
        
        return probs

class CriticNetwork(nn.Module):
    """价值网络（Critic）"""
    def __init__(self, input_channels, height, width):
        super(CriticNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 计算CNN输出特征的维度
        cnn_output_size = 64 * height * width
        
        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        # 通过CNN提取特征
        x = self.cnn(state)
        
        # 价值网络输出状态价值
        value = self.critic(x)
        
        return value

class PPOAgent:
    """使用PPO算法的智能体"""
    def __init__(self, height, width, device, lr=0.0003, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, entropy_coef=0.01):
        # 游戏板尺寸
        self.height = height
        self.width = width
        self.board_size = height * width
        
        # 设备（GPU或CPU）
        self.device = device
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        
        # 记忆缓冲区
        self.memory = PPOMemory(batch_size)
        
        # 网络
        self.actor = ActorNetwork(1, height, width, self.board_size).to(device)
        self.critic = CriticNetwork(1, height, width).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        print(f"PPO智能体初始化完成，使用设备: {device}")
    
    def preprocess_state(self, obs):
        """预处理观察，返回状态张量和动作掩码"""
        # 提取观察中的游戏板
        board = obs['board']
        
        # 将游戏板转换为张量，添加通道维度
        state = torch.FloatTensor(board).unsqueeze(0).to(self.device)
        
        # 提取动作掩码
        action_mask = torch.FloatTensor(obs['action_mask']).to(self.device)
        
        return state, action_mask
    
    def select_action(self, state, action_mask):
        """选择动作"""
        # 确保网络处于评估模式
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            # 获取策略概率分布
            probs = self.actor(state, action_mask)
            
            # 获取状态价值
            value = self.critic(state)
            
            # 创建分类分布
            dist = Categorical(probs)
            
            # 采样动作
            action = dist.sample()
            
            # 获取动作的对数概率
            prob = dist.log_prob(action)
            
        # 返回到训练模式
        self.actor.train()
        self.critic.train()
        
        return action.item(), prob.item(), value.item()
    
    def store(self, state, action, prob, val, reward, done, action_mask):
        """存储转换到记忆中"""
        self.memory.store(state, action, prob, val, reward, done, action_mask)
    
    def learn(self):
        """学习"""
        # 如果记忆为空，无需学习
        if len(self.memory.states) == 0:
            return None, None
        
        # 生成批次
        states, actions, old_probs, vals, rewards, dones, action_masks, batches = self.memory.generate_batches()
        
        # 使用广义优势估计计算优势函数
        advantages = self._compute_advantages(rewards, vals, dones)
        
        # 将优势函数归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        # PPO更新
        actor_loss_total = 0
        critic_loss_total = 0
        
        # 多个回合的更新
        for _ in range(self.n_epochs):
            for batch in batches:
                # 获取当前批次数据
                batch_states = states[batch].to(self.device)
                batch_actions = actions[batch].to(self.device)
                batch_old_probs = old_probs[batch].to(self.device)
                batch_advantages = advantages[batch].to(self.device)
                batch_action_masks = action_masks[batch].to(self.device)
                
                # 获取新策略的动作概率
                probs = self.actor(batch_states, batch_action_masks)
                dist = Categorical(probs)
                new_probs = dist.log_prob(batch_actions)
                
                # 计算策略比例和裁剪目标
                prob_ratio = torch.exp(new_probs - batch_old_probs)
                weighted_probs = prob_ratio * batch_advantages
                clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * batch_advantages
                
                # 考虑熵正则化项
                entropy = dist.entropy().mean()
                
                # Actor损失
                actor_loss = -torch.min(weighted_probs, clipped_probs).mean() - self.entropy_coef * entropy
                
                # 清除梯度并反向传播
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()
                
                # 价值预测
                critic_value = self.critic(batch_states).squeeze()
                
                # 返回目标
                returns = batch_advantages + vals[batch].to(self.device)
                
                # Critic损失
                critic_loss = nn.functional.mse_loss(critic_value, returns)
                
                # 清除梯度并反向传播
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()
                
                actor_loss_total += actor_loss.item()
                critic_loss_total += critic_loss.item()
        
        # 清空记忆
        self.memory.clear()
        
        # 返回平均损失
        return actor_loss_total / (len(batches) * self.n_epochs), critic_loss_total / (len(batches) * self.n_epochs)
    
    def _compute_advantages(self, rewards, values, dones):
        """计算广义优势估计"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # 从后向前计算
        for t in reversed(range(len(rewards)-1)):
            next_value = values[t+1]
            mask = 1 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
            
        return advantages
    
    def save_model(self, path):
        """保存模型"""
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存状态字典
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
        
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"警告: 模型文件 {path} 不存在")
            return False
        
        # 加载状态字典
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        print(f"模型已从 {path} 加载")
        return True 