import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

class PPOTrainer:
    """
    使用PPO算法训练扫雷AI
    """
    def __init__(self, env, agent, models_dir="models", logs_dir="logs"):
        """
        初始化PPO训练器
        
        参数:
            env: 扫雷游戏环境
            agent: PPO智能体
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
        self.train_actor_losses = []
        self.train_critic_losses = []
        self.eval_rewards = []
        self.eval_steps = []
        self.eval_wins = []
        
        # 当前训练难度
        self.current_difficulty = 1
        
    def train(self, num_episodes=10000, max_steps=100, n_update=128, eval_interval=100, 
              save_interval=500, difficulty_increase_threshold=0.7,
              curriculum_config=None):
        """
        训练智能体
        
        参数:
            num_episodes: 训练的总回合数
            max_steps: 每个回合的最大步数
            n_update: 更新网络的间隔步数
            eval_interval: 评估的间隔回合数
            save_interval: 保存模型的间隔回合数
            difficulty_increase_threshold: 增加难度的胜率阈值
            curriculum_config: 课程学习配置
        """
        # 默认课程学习配置
        if curriculum_config is None:
            curriculum_config = [
                {'width': 3, 'height': 3, 'num_mines': 1},  # 更简单的起点
                {'width': 5, 'height': 5, 'num_mines': 3},
                {'width': 5, 'height': 5, 'num_mines': 5},
                {'width': 8, 'height': 8, 'num_mines': 10},
                {'width': 10, 'height': 10, 'num_mines': 15},
                {'width': 16, 'height': 16, 'num_mines': 40}
            ]
        
        # 初始难度
        self.current_difficulty = 0
        self._update_env_difficulty(curriculum_config[self.current_difficulty])
        
        # 适应新环境的大小
        self.agent = self._create_new_agent_for_size(
            curriculum_config[self.current_difficulty]['height'],
            curriculum_config[self.current_difficulty]['width']
        )
        
        print(f"开始训练，初始难度：{curriculum_config[self.current_difficulty]}")
        
        # 训练循环
        episode_rewards = []
        episode_steps = []
        episode_wins = []
        running_actor_loss = 0.0
        running_critic_loss = 0.0
        steps_since_update = 0
        
        for episode in range(1, num_episodes + 1):
            # 重置环境
            obs, _ = self.env.reset()
            state, action_mask = self.agent.preprocess_state(obs)
            
            total_reward = 0
            done = False
            
            for step in range(max_steps):
                # 选择动作
                action_result = self.agent.select_action(state, action_mask)
                
                # 处理返回值
                if isinstance(action_result, tuple):
                    action, prob, val = action_result
                else:
                    action = action_result
                    prob, val = 0, 0  # 假值，不使用PPO时
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state, next_action_mask = self.agent.preprocess_state(next_obs)
                
                # 记录转换
                done = terminated or truncated
                
                # 对于PPO，存储转换
                self.agent.store(state, action, prob, val, reward, done, action_mask)
                steps_since_update += 1
                
                # 更新PPO
                if steps_since_update >= n_update or done:
                    actor_loss, critic_loss = self.agent.learn()
                    if actor_loss is not None and critic_loss is not None:
                        running_actor_loss += actor_loss
                        running_critic_loss += critic_loss
                    steps_since_update = 0
                
                total_reward += reward
                state = next_state
                action_mask = next_action_mask
                
                if done:
                    break
            
            # 记录本回合的结果
            episode_rewards.append(total_reward)
            episode_steps.append(step + 1)
            episode_wins.append(1 if info.get("win", False) else 0)
            
            # 每eval_interval回合评估一次
            if episode % eval_interval == 0:
                mean_reward = np.mean(episode_rewards[-eval_interval:])
                mean_steps = np.mean(episode_steps[-eval_interval:])
                win_rate = np.mean(episode_wins[-eval_interval:])
                mean_actor_loss = running_actor_loss / eval_interval if running_actor_loss > 0 else 0
                mean_critic_loss = running_critic_loss / eval_interval if running_critic_loss > 0 else 0
                
                self.train_rewards.append(mean_reward)
                self.train_steps.append(mean_steps)
                self.train_wins.append(win_rate)
                self.train_actor_losses.append(mean_actor_loss)
                self.train_critic_losses.append(mean_critic_loss)
                
                print(f"回合 {episode}/{num_episodes} - 奖励: {mean_reward:.2f}, 步数: {mean_steps:.2f}, 胜率: {win_rate:.2f}, Actor损失: {mean_actor_loss:.4f}, Critic损失: {mean_critic_loss:.4f}")
                
                # 评估当前模型
                eval_reward, eval_steps, eval_win_rate = self.evaluate(100)
                self.eval_rewards.append(eval_reward)
                self.eval_steps.append(eval_steps)
                self.eval_wins.append(eval_win_rate)
                
                # 检查是否需要增加难度
                if (eval_win_rate >= difficulty_increase_threshold and 
                    self.current_difficulty < len(curriculum_config) - 1):
                    self.current_difficulty += 1
                    new_config = curriculum_config[self.current_difficulty]
                    print(f"胜率 {eval_win_rate:.2f} 达到阈值，增加难度到 {new_config}")
                    
                    # 保存当前模型
                    model_path = os.path.join(self.models_dir, f"agent_difficulty_{self.current_difficulty-1}.pth")
                    self.agent.save_model(model_path)
                    
                    # 更新环境难度
                    self._update_env_difficulty(new_config)
                    
                    # 为新环境创建适应大小的新智能体
                    self.agent = self._create_new_agent_for_size(
                        new_config['height'], new_config['width']
                    )
                
                running_actor_loss = 0.0
                running_critic_loss = 0.0
                
                # 绘制并保存训练曲线
                self._plot_training_curves()
            
            # 定期保存模型
            if episode % save_interval == 0:
                model_path = os.path.join(self.models_dir, f"agent_episode_{episode}.pth")
                self.agent.save_model(model_path)
        
        # 训练结束，保存最终模型
        model_path = os.path.join(self.models_dir, "agent_final.pth")
        self.agent.save_model(model_path)
        
        return self.train_rewards, self.train_steps, self.train_wins
    
    def evaluate(self, num_episodes=100):
        """
        评估智能体性能
        
        参数:
            num_episodes: 评估的回合数
            
        返回:
            平均奖励，平均步数，胜率
        """
        total_rewards = []
        total_steps = []
        total_wins = 0
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            state, action_mask = self.agent.preprocess_state(obs)
            
            total_reward = 0
            for step in range(self.env.board_size * 2):  # 设置较大的最大步数
                # 处理PPO和DQN的不同接口
                action_result = self.agent.select_action(state, action_mask)
                
                # 处理返回值
                if isinstance(action_result, tuple):
                    action = action_result[0]  # PPO返回元组 (action, prob, val)
                else:
                    action = action_result  # DQN只返回action
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state, next_action_mask = self.agent.preprocess_state(next_obs)
                
                total_reward += reward
                state = next_state
                action_mask = next_action_mask
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
            total_steps.append(step + 1)
            if info.get("win", False):
                total_wins += 1
        
        # 计算平均指标
        mean_reward = np.mean(total_rewards)
        mean_steps = np.mean(total_steps)
        win_rate = total_wins / num_episodes
        
        print(f"评估结果: 奖励: {mean_reward:.2f}, 步数: {mean_steps:.2f}, 胜率: {win_rate:.2f}")
        
        return mean_reward, mean_steps, win_rate
    
    def _update_env_difficulty(self, config):
        """更新环境难度"""
        # 创建新环境实例
        from environment import MinesweeperEnv
        self.env = MinesweeperEnv(
            width=config['width'],
            height=config['height'],
            num_mines=config['num_mines'],
            render_mode=self.env.render_mode
        )
    
    def _create_new_agent_for_size(self, height, width):
        """创建适应新大小的智能体"""
        from agent import PPOAgent
        
        # 创建新智能体
        new_agent = PPOAgent(
            height=height,
            width=width,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr=0.0002,  # 较低的学习率
            gamma=0.99,
            gae_lambda=0.95,
            policy_clip=0.2,
            batch_size=64,
            n_epochs=5,
            entropy_coef=0.01  # 鼓励探索
        )
        
        return new_agent
    
    def _plot_training_curves(self):
        """绘制并保存训练曲线"""
        plt.figure(figsize=(15, 15))
        
        # 绘制奖励
        plt.subplot(3, 2, 1)
        plt.plot(self.train_rewards, label='训练奖励')
        plt.plot(self.eval_rewards, label='评估奖励')
        plt.xlabel('评估次数')
        plt.ylabel('平均奖励')
        plt.title('奖励曲线')
        plt.legend()
        
        # 绘制步数
        plt.subplot(3, 2, 2)
        plt.plot(self.train_steps, label='训练步数')
        plt.plot(self.eval_steps, label='评估步数')
        plt.xlabel('评估次数')
        plt.ylabel('平均步数')
        plt.title('步数曲线')
        plt.legend()
        
        # 绘制胜率
        plt.subplot(3, 2, 3)
        plt.plot(self.train_wins, label='训练胜率')
        plt.plot(self.eval_wins, label='评估胜率')
        plt.xlabel('评估次数')
        plt.ylabel('胜率')
        plt.title('胜率曲线')
        plt.axhline(y=0.7, color='r', linestyle='--', label='难度提升阈值')
        plt.legend()
        
        # 绘制Actor损失
        plt.subplot(3, 2, 4)
        plt.plot(self.train_actor_losses)
        plt.xlabel('评估次数')
        plt.ylabel('Actor损失')
        plt.title('Actor损失曲线')
        
        # 绘制Critic损失
        plt.subplot(3, 2, 5)
        plt.plot(self.train_critic_losses)
        plt.xlabel('评估次数')
        plt.ylabel('Critic损失')
        plt.title('Critic损失曲线')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, 'ppo_training_curves.png'))
        plt.close() 