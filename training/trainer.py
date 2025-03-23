import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

class Trainer:
    """训练器基类"""
    def __init__(self, env, agent, models_dir="models", logs_dir="logs"):
        """
        初始化训练器
        
        参数:
            env: 游戏环境
            agent: 智能体
            models_dir: 模型保存目录
            logs_dir: 日志保存目录
        """
        self.env = env
        self.agent = agent
        
        # 创建目录
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # 记录训练指标
        self.train_rewards = []
        self.train_steps = []
        self.train_wins = []
        self.train_losses = []
        self.eval_rewards = []
        self.eval_steps = []
        self.eval_wins = []
        
        # 当前训练难度
        self.current_difficulty = 1
        
    def train(self, num_episodes, max_steps, eval_interval, save_interval):
        """训练智能体（由子类实现）"""
        raise NotImplementedError("子类必须实现train方法")
    
    def evaluate(self, num_episodes):
        """评估智能体（由子类实现）"""
        raise NotImplementedError("子类必须实现evaluate方法")
    
    def _update_env_difficulty(self, config):
        """更新环境难度"""
        from environment import MinesweeperEnv
        self.env = MinesweeperEnv(
            width=config['width'],
            height=config['height'],
            num_mines=config['num_mines'],
            render_mode=self.env.render_mode
        )
    
    def _create_new_agent_for_size(self, height, width):
        """创建适应新大小的智能体"""
        from agent import DQNAgent
        
        # 保存当前智能体的超参数
        if hasattr(self, 'agent'):
            epsilon = self.agent.epsilon
            epsilon_end = self.agent.epsilon_end
            epsilon_decay = self.agent.epsilon_decay
            gamma = self.agent.gamma
            batch_size = self.agent.batch_size
            target_update = self.agent.target_update
            device = self.agent.device
        else:
            # 默认值
            epsilon = 1.0
            epsilon_end = 0.1
            epsilon_decay = 0.995
            gamma = 0.99
            batch_size = 64
            target_update = 10
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建新智能体
        new_agent = DQNAgent(
            height=height,
            width=width,
            epsilon_start=epsilon,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            gamma=gamma,
            batch_size=batch_size,
            target_update=target_update,
            device=device
        )
        
        return new_agent
    
    def _plot_training_curves(self):
        """绘制训练曲线（由子类重写）"""
        raise NotImplementedError("子类必须实现_plot_training_curves方法")

    def _plot_training_curves(self):
        """绘制并保存训练曲线"""
        plt.figure(figsize=(15, 10))
        
        # 绘制奖励
        plt.subplot(2, 2, 1)
        plt.plot(self.train_rewards, label='训练奖励')
        plt.plot(self.eval_rewards, label='评估奖励')
        plt.xlabel('评估次数')
        plt.ylabel('平均奖励')
        plt.title('奖励曲线')
        plt.legend()
        
        # 绘制步数
        plt.subplot(2, 2, 2)
        plt.plot(self.train_steps, label='训练步数')
        plt.plot(self.eval_steps, label='评估步数')
        plt.xlabel('评估次数')
        plt.ylabel('平均步数')
        plt.title('步数曲线')
        plt.legend()
        
        # 绘制胜率
        plt.subplot(2, 2, 3)
        plt.plot(self.train_wins, label='训练胜率')
        plt.plot(self.eval_wins, label='评估胜率')
        plt.xlabel('评估次数')
        plt.ylabel('胜率')
        plt.title('胜率曲线')
        plt.axhline(y=0.7, color='r', linestyle='--', label='难度提升阈值')
        plt.legend()
        
        # 绘制损失
        plt.subplot(2, 2, 4)
        plt.plot(self.train_losses)
        plt.xlabel('评估次数')
        plt.ylabel('损失')
        plt.title('损失曲线')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, 'training_curves.png'))
        plt.close() 