import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from agent.human_reasoning import HumanReasoning

class DQNNetwork(nn.Module):
    """
    用于扫雷游戏的深度Q网络模型
    输入：当前游戏板状态
    输出：每个可能动作的Q值
    """
    def __init__(self, height, width, hidden_size=256):  # 增加隐藏层大小
        super(DQNNetwork, self).__init__()
        self.height = height
        self.width = width
        input_channels = 4  # 修改为4通道输入
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 添加第三个卷积层
        
        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 计算卷积后的特征图大小
        conv_output_size = height * width * 64
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 添加第二个全连接层
        self.fc3 = nn.Linear(hidden_size // 2, height * width)  # 输出每个格子的Q值
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # 确保输入有合适的形状 [batch_size, channels, height, width]
        batch_size = x.size(0)
        x = x.view(batch_size, 4, self.height, self.width)  # 4通道输入
        
        # 卷积层+批归一化+激活
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 应用dropout
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class DuelingDQNNetwork(nn.Module):
    """
    基于Dueling架构的深度Q网络模型
    将Q值分解为状态值V(s)和优势函数A(s,a)
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    def __init__(self, height, width, hidden_size=256):
        super(DuelingDQNNetwork, self).__init__()
        self.height = height
        self.width = width
        input_channels = 4
        
        # 共享的特征提取层
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 计算卷积后的特征图大小
        conv_output_size = height * width * 64
        
        # 价值流 - 评估状态的价值V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # 输出单一状态值
        )
        
        # 优势流 - 评估每个动作的优势A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, height * width)  # 输出每个动作的优势
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 4, self.height, self.width)
        
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 计算状态值
        value = self.value_stream(x)
        
        # 计算动作优势
        advantage = self.advantage_stream(x)
        
        # 合并价值和优势得到Q值：Q = V(s) + A(s,a) - mean(A(s,a))
        # 减去平均优势以增强数值稳定性
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

# 定义优先经验回放缓冲区
class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    根据TD误差大小对经验进行优先级采样
    """
    def __init__(self, max_size, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """
        初始化优先经验回放缓冲区
        
        参数:
            max_size: 缓冲区最大容量
            alpha: 决定优先级影响程度的指数 (0表示均匀采样, 1表示完全按优先级)
            beta: 重要性采样的指数，用于修正采样偏差 (1表示完全修正)
            beta_increment: 每次采样后beta的增加量
            epsilon: 添加到TD误差以确保所有经验至少有一些采样概率
        """
        self.max_size = max_size
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.next_index = 0
        self._max_priority = 1.0  # 新经验的初始优先级
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, experience, error=None):
        """添加新经验到缓冲区"""
        priority = self._max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            if self.next_index >= len(self.buffer):
                self.next_index = 0
            self.buffer[self.next_index] = experience
            self.priorities[self.next_index] = priority
            
        self.next_index = (self.next_index + 1) % self.max_size
    
    def sample(self, batch_size):
        """从缓冲区中采样一批经验"""
        if len(self.buffer) < batch_size:
            return [], [], []
            
        # 计算采样概率 - 确保是一维数组
        priorities = np.array(self.priorities, dtype=np.float32).flatten()
        sum_priorities = np.sum(priorities)
        
        if sum_priorities == 0:
            # 如果所有优先级都是0，使用均匀分布
            probs = np.ones_like(priorities) / len(priorities)
        else:
            probs = priorities / sum_priorities
        
        # 按概率采样，确保probs是一维数组
        try:
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        except ValueError as e:
            print(f"采样错误: {e}")
            print(f"probs形状: {probs.shape}, 是否一维: {probs.ndim == 1}")
            print(f"probs总和: {np.sum(probs)}")
            # 使用均匀采样作为备选方案
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # 归一化权重
        
        # 增加beta以逐渐减少偏差
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, errors):
        """更新经验的优先级"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
     
class DQNAgent:
    """
    基于DQN的扫雷游戏智能体
    """
    def __init__(self, height, width, device=None, learning_rate=2e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997,
                 memory_size=20000, batch_size=128, target_update=5,
                 double_dqn=True, dueling_dqn=True, 
                 prioritized_replay=True, supervised_pretrain=True, 
                 use_human_reasoning=True):
        """
        初始化DQN智能体
        
        参数:
            height (int): 游戏板高度
            width (int): 游戏板宽度
            device: 计算设备 (CPU/GPU)
            learning_rate (float): 学习率
            gamma (float): 折扣因子
            epsilon_start (float): 初始探索率
            epsilon_end (float): 最小探索率
            epsilon_decay (float): 探索率衰减系数
            memory_size (int): 经验回放缓冲区大小
            batch_size (int): 批次大小
            target_update (int): 目标网络更新频率
            double_dqn (bool): 是否使用双重DQN
            dueling_dqn (bool): 是否使用Dueling DQN
            prioritized_replay (bool): 是否使用优先经验回放
            supervised_pretrain (bool): 是否使用自监督预训练
            use_human_reasoning (bool): 是否使用人类推理能力
        """
        self.height = height
        self.width = width
        self.state_size = height * width
        self.action_size = height * width
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.prioritized_replay = prioritized_replay
        self.supervised_pretrain = supervised_pretrain
        self.use_human_reasoning = use_human_reasoning
        
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 选择网络架构
        if dueling_dqn:
            # 使用Dueling DQN网络
            self.policy_net = DuelingDQNNetwork(height, width).to(self.device)
            self.target_net = DuelingDQNNetwork(height, width).to(self.device)
        else:
            # 使用标准DQN网络
            self.policy_net = DQNNetwork(height, width).to(self.device)
            self.target_net = DQNNetwork(height, width).to(self.device)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不需要训练
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # 经验回放缓冲区
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(max_size=memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
        
        # 记忆机制：记录明显安全的格子
        self.known_safe_cells = set()
        
        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 训练参数
        self.steps_done = 0
        
        # 如果启用预训练，初始化预训练数据集
        if supervised_pretrain:
            self.pretrain_memory = deque(maxlen=10000)
        
        # 添加人类推理模块
        if use_human_reasoning:
            self.human_reasoning = HumanReasoning(height, width)
        
        # 添加跟踪和调试信息
        self.reasoning_used_count = 0
        self.network_used_count = 0
        
    def select_action(self, state, action_mask):
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态（游戏板）
            action_mask: 可行动作掩码
            
        返回:
            选择的动作索引
        """
        # 首先检查是否有已知安全的格子
        safe_actions = self._get_safe_actions(state, action_mask)
        if safe_actions:
            # 如果有安全的格子，优先选择
            return random.choice(safe_actions)
        
        # 将action_mask转换为布尔掩码
        valid_actions = np.where(action_mask == 1)[0]
        
        if not len(valid_actions):
            # 如果没有有效动作，随机选择一个动作（这种情况不应该发生）
            return random.randint(0, self.action_size - 1)
        
        # 探索：随机选择动作
        if random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # 利用：选择Q值最大的动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            # 将无效动作的Q值设为很小的值
            mask_tensor = torch.FloatTensor(action_mask).to(self.device)
            masked_q_values = q_values.squeeze(0) - (1 - mask_tensor) * 1e9
            
            return masked_q_values.argmax().item()
    
    def _get_safe_actions(self, state, action_mask):
        """
        根据当前状态和逻辑推理识别确定安全的格子
        
        返回:
            安全动作的列表，如果没有确定安全的格子则为空列表
        """
        # 获取游戏板状态（第一个通道是主要状态）
        board_state = state[0]
        safe_actions = []
        
        # 首先检查已知安全格子集合
        for action in self.known_safe_cells:
            y, x = divmod(action, self.width)
            # 如果仍然有效（未挖掘）
            if action_mask[action] == 1:
                safe_actions.append(action)
        
        # 使用简单的扫雷逻辑识别安全格子
        for y in range(self.height):
            for x in range(self.width):
                pos = y * self.width + x
                
                # 只关注已挖掘的数字格子
                if board_state[y, x] > 0 and board_state[y, x] < 1.0:  # 归一化值在0-1之间的数字
                    # 获取周围未挖掘的格子
                    unknown_cells = []
                    flagged_mines = 0
                    
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.height and 0 <= nx < self.width:
                                npos = ny * self.width + nx
                                # 未挖掘的格子
                                if board_state[ny, nx] == -0.5 and action_mask[npos] == 1:
                                    unknown_cells.append(npos)
                                # 已标记的地雷
                                elif board_state[ny, nx] == 1.5:  # 归一化后的地雷值
                                    flagged_mines += 1
                    
                    # 如果已标记地雷数等于格子数字，则周围所有未挖掘格子安全
                    if round(board_state[y, x] * 8) == flagged_mines and unknown_cells:
                        for cell in unknown_cells:
                            if cell not in safe_actions:
                                safe_actions.append(cell)
                                self.known_safe_cells.add(cell)
        
        return safe_actions
    
    def update_safe_cells(self, state, action_mask):
        """
        更新已知安全格子集合
        """
        # 首先清除已经不可用的安全格子（已挖掘）
        self.known_safe_cells = {cell for cell in self.known_safe_cells if action_mask[cell] == 1}
        
        # 添加新的安全格子
        self._get_safe_actions(state, action_mask)
    
    def remember(self, state, action, reward, next_state, done, action_mask=None, next_action_mask=None):
        """
        存储经验到回放缓冲区，应用奖励塑形

        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的原始奖励
            next_state: 下一个状态
            done: 是否是终止状态
            action_mask: 当前状态的有效动作掩码
            next_action_mask: 下一个状态的有效动作掩码
        """
        # 预处理状态，如果它们是字典格式
        if isinstance(state, dict):
            board = state["board"]
            state_array = self.preprocess_state(board)
            if action_mask is None and "action_mask" in state:
                action_mask = state["action_mask"]
        else:
            state_array = state
            
        if isinstance(next_state, dict):
            next_board = next_state["board"]
            next_state_array = self.preprocess_state(next_board)
            if next_action_mask is None and "action_mask" in next_state:
                next_action_mask = next_state["action_mask"]
        else:
            next_state_array = next_state
        
        # 高级奖励塑形策略
        shaped_reward = self._shape_reward(reward, state, next_state, action, done)
        
        # 存储经验
        experience = (state_array, action, shaped_reward, next_state_array, done)
        
        # 根据使用的回放缓冲区类型不同处理方式不同
        if self.prioritized_replay:
            # 对于优先经验回放，先用最大优先级添加
            self.memory.add(experience)
        else:
            # 对于普通经验回放，直接添加
            self.memory.append(experience)
        
        # 如果使用预训练，也保存给自监督学习使用
        if self.supervised_pretrain and not done:
            self.pretrain_memory.append((state_array, action))
    
    def _shape_reward(self, original_reward, state, next_state, action, done):
        """
        应用高级奖励塑形策略
        
        参数:
            original_reward: 原始奖励
            state: 当前状态
            next_state: 下一个状态
            action: 执行的动作
            done: 是否是终止状态
            
        返回:
            shaped_reward: 塑形后的奖励
        """
        shaped_reward = original_reward
        
        # 1. 基本的奖励塑形: 非终止状态的连续性奖励
        if not done:
            shaped_reward += 0.1  # 鼓励探索
        
        # 2. 成功打开空白格的奖励（可能触发连锁反应）
        if isinstance(next_state, dict) and isinstance(state, dict):
            # 计算新打开的格子数量
            prev_opened = np.sum(state["board"] >= 0)
            curr_opened = np.sum(next_state["board"] >= 0)
            newly_opened = curr_opened - prev_opened
            
            # 奖励与新打开的格子数量成正比
            if newly_opened > 1:  # 如果打开了多个格子（触发连锁反应）
                shaped_reward += 0.1 * newly_opened
        
        # 3. 边缘探索奖励 - 鼓励在已知区域边缘探索
        if isinstance(state, dict):
            y, x = divmod(action, self.width)
            edge_reward = 0
            
            # 检查选择的格子是否在已知数字旁边
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        # 如果周围有数字格子，增加奖励
                        if 1 <= state["board"][ny, nx] <= 8:
                            edge_reward = 0.2
                            break
            
            shaped_reward += edge_reward
        
        # 4. 避免点开已知地雷的惩罚
        if done and original_reward < 0:  # 游戏失败
            # 检查是否可以通过逻辑推理避免这个错误
            if self.use_human_reasoning:
                board_state = self.preprocess_state(state["board"])
                action_mask = np.ones(self.action_size)
                for i in range(self.action_size):
                    y, x = divmod(i, self.width)
                    if state["board"][y, x] >= 0:  # 已经打开的格子
                        action_mask[i] = 0
                
                # 找出所有可能的地雷位置
                _, mine_actions = self.human_reasoning.solve_with_csp(board_state, action_mask)
                
                # 如果选择的动作在推理出的地雷位置中，额外惩罚
                if action in mine_actions:
                    shaped_reward -= 1.0  # 额外惩罚推理失误
        
        # 5. 对正确标记地雷或安全打开格子的奖励
        if self.use_human_reasoning:
            board_state = self.preprocess_state(state["board"]) if isinstance(state, dict) else state
            action_mask = np.ones(self.action_size)
            
            if isinstance(state, dict):
                for i in range(self.action_size):
                    y, x = divmod(i, self.width)
                    if state["board"][y, x] >= 0:  # 已经打开的格子
                        action_mask[i] = 0
            
            # 找出安全的格子
            safe_actions, _ = self.human_reasoning.solve_with_csp(board_state, action_mask)
            
            # 如果选择的动作在推理出的安全位置中，奖励正确决策
            if action in safe_actions and not done:
                shaped_reward += 0.5  # 奖励正确的推理决策
        
        # 6. 为游戏最终结果提供额外奖励/惩罚
        if done:
            if original_reward > 0:  # 游戏胜利
                shaped_reward += 2.0  # 额外胜利奖励
            else:  # 游戏失败
                # 根据游戏进度给予不同程度的惩罚
                if isinstance(state, dict):
                    # 计算已打开的格子百分比
                    total_cells = self.height * self.width
                    opened_cells = np.sum(state["board"] >= 0)
                    progress = opened_cells / total_cells
                    
                    # 根据进度减轻惩罚
                    if progress > 0.5:  # 如果已经完成一半以上
                        shaped_reward += 1.0  # 减轻失败惩罚
        
        return shaped_reward
    
    def replay(self):
        """从经验回放缓冲区中学习"""
        if self.prioritized_replay:
            # 优先经验回放
            if len(self.memory) < self.batch_size:
                return None
            
            # 按优先级采样
            samples, indices, weights = self.memory.sample(self.batch_size)
            if not samples:  # 如果采样失败
                return None
                
            # 解包样本
            states, actions, rewards, next_states, dones = zip(*samples)
            
            # 转换为张量
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
            
            # 计算当前Q值
            current_q_values = self.policy_net(states).gather(1, actions)
            
            # 使用Double DQN或标准DQN计算目标Q值
            if self.double_dqn:
                with torch.no_grad():
                    # 使用策略网络选择动作
                    next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                    # 使用目标网络评估动作
                    next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # 计算目标值
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
            # 计算TD误差 (用于更新优先级)
            td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
            
            # 带权重的Huber损失
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
            
            # 优化模型
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新经验优先级
            self.memory.update_priorities(indices, td_errors)
            
        else:
            # 标准经验回放
            if len(self.memory) < self.batch_size:
                return None
            
            # 随机采样一个批次
            minibatch = random.sample(self.memory, self.batch_size)
            
            # 准备批次数据
            states, actions, rewards, next_states, dones = zip(*minibatch)
            
            # 将列表转换为numpy数组，再转换为tensor
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            # 计算当前Q值
            current_q_values = self.policy_net(states).gather(1, actions)
            
            # 使用Double DQN计算目标Q值
            if self.double_dqn:
                with torch.no_grad():
                    # 使用策略网络确定最佳动作
                    next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                    # 使用目标网络评估这些动作的价值
                    max_next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # 标准DQN：直接选择目标网络的最大值
                with torch.no_grad():
                    max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # 贝尔曼方程计算目标值
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
            # 计算Huber损失（对异常值更稳健）
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # 优化模型
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target_network()
        
        # 返回损失值用于监控
        return loss.item()
    
    def preprocess_state(self, board):
        """
        预处理状态，将原始游戏板转换为神经网络输入格式
        
        参数:
            board: 原始游戏板状态
            
        返回:
            处理后的状态数组
        """
        # 将游戏板状态标准化到[-0.5, 1.5]范围
        # -1 (未挖掘) -> -0.5
        # 0-8 (数字) -> 0-1.0
        # 9 (地雷) -> 1.5
        normalized_board = np.zeros_like(board, dtype=np.float32)
        
        # 未挖掘的格子
        normalized_board[board == -1] = -0.5
        
        # 数字格子（将0-8映射到0-1范围）
        for i in range(9):
            normalized_board[board == i] = i / 8.0
        
        # 地雷格子
        normalized_board[board == 9] = 1.5
        
        # 创建多通道状态表示
        state = np.zeros((4, self.height, self.width), dtype=np.float32)
        
        # 通道0：标准化后的游戏板
        state[0] = normalized_board
        
        # 通道1：未挖掘的格子（1表示未挖掘，0表示已挖掘）
        state[1] = (board == -1).astype(np.float32)
        
        # 通道2：数字格子的标志（1表示是数字，0表示不是）
        state[2] = ((board >= 0) & (board <= 8)).astype(np.float32)
        
        # 通道3：旗帜/地雷标记（1表示有标记，0表示没有）
        state[3] = (board == 9).astype(np.float32)
        
        return state
    
    def pretrain(self, num_epochs=100, batch_size=64):
        """
        使用自监督学习预训练网络
        """
        if not self.supervised_pretrain or len(self.pretrain_memory) < batch_size:
            print("预训练数据不足或未启用预训练")
            return
        
        print(f"开始自监督预训练，数据集大小：{len(self.pretrain_memory)}")
        
        for epoch in range(num_epochs):
            # 从预训练数据集中随机采样
            batch = random.sample(self.pretrain_memory, min(batch_size, len(self.pretrain_memory)))
            states, actions = zip(*batch)
            
            # 确保所有状态有相同的形状
            # 检查批次中第一个状态的形状
            first_state = states[0]
            if isinstance(first_state, np.ndarray):
                # 获取第一个状态的形状
                state_shape = first_state.shape
                # 调整批次中所有状态为相同形状
                normalized_states = []
                for state in states:
                    if state.shape != state_shape:
                        # 如果形状不匹配，调整尺寸或跳过
                        print(f"警告：发现形状不匹配的状态 {state.shape} vs {state_shape}，已跳过")
                        continue
                    normalized_states.append(state)
                
                if len(normalized_states) < batch_size // 2:
                    print(f"警告：调整后批次大小太小 ({len(normalized_states)}), 跳过此次训练批次")
                    continue
                    
                states = normalized_states
                # 取状态对应的动作
                actions = actions[:len(states)]
            
            try:
                # 将数据转换为张量
                states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                actions_tensor = torch.LongTensor(actions).to(self.device)
                
                # 预测动作
                self.policy_net.train()
                q_values = self.policy_net(states_tensor)
                
                # 计算交叉熵损失
                loss = F.cross_entropy(q_values, actions_tensor)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"预训练周期 {epoch+1}/{num_epochs}, 损失: {loss.item():.4f}")
            except Exception as e:
                print(f"预训练批次处理错误: {e}")
                continue
        
        # 预训练完成后更新目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("预训练完成!")
    
    def save_model(self, filepath):
        """保存模型权重"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load_model(self, filepath):
        """加载模型权重"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
    
    def act(self, state, action_mask=None, deterministic=False):
        """
        根据当前状态选择动作
        
        参数:
            state: 当前环境状态（字典或数组格式）
            action_mask: 有效动作的掩码
            deterministic: 是否使用确定性策略(贪婪)
            
        返回:
            选择的动作索引
        """
        # 处理字典格式的状态
        if isinstance(state, dict):
            board = state["board"]
            state_array = self.preprocess_state(board)
            
            # 如果action_mask为None且字典中包含掩码，使用字典中的掩码
            if action_mask is None and "action_mask" in state:
                action_mask = state["action_mask"]
        else:
            # 如果已经是预处理过的数组状态，直接使用
            state_array = state
        
        # 如果启用了人类推理，并且不是确定要进行探索
        if self.use_human_reasoning and (deterministic or random.random() > self.epsilon):
            # 尝试使用人类推理模块找出安全的移动
            action = self.human_reasoning.find_safe_move(state_array, action_mask)
            
            # 如果找到了安全的移动，就使用它
            if action is not None:
                self.reasoning_used_count += 1
                return action
        
        # 如果人类推理没有找到安全移动，或者决定探索，就使用神经网络
        # 将状态调整为网络所需的形状
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        
        # 使用策略网络预测Q值
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        # 应用动作掩码（如果提供）
        if action_mask is not None:
            mask_tensor = torch.FloatTensor(action_mask).to(self.device)
            q_values = q_values * mask_tensor - 1e8 * (1 - mask_tensor)
        
        # 选择最大Q值的动作（贪婪策略）或随机动作（ε-贪婪探索）
        if deterministic or random.random() > self.epsilon:
            action = torch.argmax(q_values).item()
        else:
            # 随机选择有效动作
            if action_mask is not None:
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    # 如果没有有效动作，选择任意动作（应该不会发生）
                    action = random.randint(0, self.action_size - 1)
            else:
                action = random.randint(0, self.action_size - 1)
        
        self.network_used_count += 1
        return action
    
    def _soft_update_target_network(self, tau=0.01):
        """
        软更新目标网络
        
        参数:
            tau: 软更新系数，控制更新的速度
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
        
        # 更新探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1 