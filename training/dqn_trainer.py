import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from .trainer import Trainer
from agent import DQNAgent
import random
from environment.minesweeper import MinesweeperEnv as Minesweeper
from environment.minesweeper_solver import MinesweeperSolver
from tqdm import tqdm
from collections import deque

class DQNTrainer(Trainer):
    """
    使用DQN算法训练扫雷AI
    """
    def __init__(self, env, agent, models_dir="models", logs_dir="logs"):
        """
        初始化DQN训练器
        
        参数:
            env: 扫雷游戏环境
            agent: DQN智能体
            models_dir: 模型保存目录
            logs_dir: 日志保存目录
        """
        super().__init__(env, agent, models_dir, logs_dir)
        
        # 记录训练指标
        self.train_rewards = []
        self.train_steps = []
        self.train_wins = []
        self.losses = []
        self.eval_rewards = []
        self.eval_steps = []
        self.eval_wins = []
        
        # 额外添加损失指标
        self.train_losses = []
        
        # 当前训练难度
        self.current_difficulty = 1
        
    def train(self, num_episodes=10000, initial_difficulty=3, max_difficulty=10, 
              difficulty_increase_threshold=0.4, eval_freq=100, save_freq=500,
              render=False, verbose=True, enable_human_reasoning=True):
        """
        使用DQN算法训练智能体
        
        参数:
            num_episodes: 训练的总回合数
            initial_difficulty: 初始的游戏难度(地雷数)
            max_difficulty: 最大游戏难度
            difficulty_increase_threshold: 增加难度的阈值(胜率)
            eval_freq: 评估的频率(每eval_freq个回合进行一次评估)
            save_freq: 保存模型的频率
            render: 是否渲染游戏画面
            verbose: 是否打印详细信息
            enable_human_reasoning: 是否启用人类推理
        """
        
        # 设置环境和智能体
        difficulty = initial_difficulty
        self.env.set_difficulty(difficulty)
        
        # 创建结果记录
        results = {
            'episode_rewards': [],
            'episode_steps': [],
            'episode_wins': [],
            'episode_losses': [],
            'eval_rewards': [],
            'eval_win_rates': [],
            'difficulties': []
        }
        
        # 启用或禁用智能体的人类推理功能
        if hasattr(self.agent, 'use_human_reasoning'):
            self.agent.use_human_reasoning = enable_human_reasoning
            
        # 跟踪最近的胜率，用于动态调整难度
        win_history = deque(maxlen=100)
        
        # 训练循环
        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # 单回合循环
            while not done:
                # 获取有效动作掩码
                action_mask = self.env.get_valid_actions()
                
                # 选择动作（使用act方法，整合了人类推理和神经网络）
                action = self.agent.act(state, action_mask)
                
                # 执行动作
                next_state, reward, done, _, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                # 存储经验
                self.agent.remember(state, action, reward, next_state, done, action_mask)
                
                # 智能体学习
                loss = self.agent.replay()
                
                # 更新状态
                state = next_state
                
                # 如果需要，渲染环境
                if render:
                    self.env.render(disable_text_output=True)
            
            # 记录回合结果
            win = info.get('win', False)
            win_history.append(1 if win else 0)
            
            results['episode_rewards'].append(total_reward)
            results['episode_steps'].append(steps)
            results['episode_wins'].append(1 if win else 0)
            results['episode_losses'].append(loss if loss is not None else 0)
            results['difficulties'].append(difficulty)
            
            # 每隔一定回合数打印进度
            if verbose and episode % 100 == 0:
                recent_rewards = results['episode_rewards'][-100:]
                recent_wins = results['episode_wins'][-100:]
                recent_steps = results['episode_steps'][-100:]
                win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
                
                print(f"回合: {episode}/{num_episodes}, " 
                      f"难度: {difficulty}, "
                      f"平均奖励: {np.mean(recent_rewards):.2f}, "
                      f"平均步数: {np.mean(recent_steps):.2f}, "
                      f"胜率: {win_rate:.2f}, "
                      f"探索率: {self.agent.epsilon:.3f}")
                
                # 如果启用了人类推理，显示使用情况统计
                if enable_human_reasoning and hasattr(self.agent, 'reasoning_used_count'):
                    total_decisions = self.agent.reasoning_used_count + self.agent.network_used_count
                    reasoning_ratio = self.agent.reasoning_used_count / total_decisions if total_decisions > 0 else 0
                    print(f"人类推理使用率: {reasoning_ratio:.2f} ({self.agent.reasoning_used_count}/{total_decisions})")
                    # 重置计数器以便跟踪最近的使用情况
                    self.agent.reasoning_used_count = 0
                    self.agent.network_used_count = 0
            
            # 周期性评估
            if episode % eval_freq == 0:
                avg_reward, win_rate = self.evaluate(num_episodes=20, difficulty=difficulty)
                results['eval_rewards'].append(avg_reward)
                results['eval_win_rates'].append(win_rate)
                
                if verbose:
                    print(f"评估 - 平均奖励: {avg_reward:.2f}, 胜率: {win_rate:.2f}")
                
                # 根据评估结果调整难度
                if win_rate >= difficulty_increase_threshold and difficulty < max_difficulty:
                    difficulty += 1
                    self.env.set_difficulty(difficulty)
                    print(f"难度增加至 {difficulty}")
            
            # 周期性保存模型
            if episode % save_freq == 0:
                self.save_model(f"dqn_model_ep{episode}_diff{difficulty}")
        
        return results
    
    def evaluate(self, num_episodes=100, difficulty=None, render=False, deterministic=True, use_human_reasoning=True):
        """
        评估智能体性能
        
        参数:
            num_episodes: 评估的回合数
            difficulty: 评估的难度级别，如果为None则使用当前环境难度
            render: 是否渲染游戏
            deterministic: 是否使用确定性策略
            use_human_reasoning: 是否在评估中使用人类推理
            
        返回:
            avg_reward: 平均奖励
            win_rate: 胜率
        """
        # 如果指定了难度，暂时更改环境难度
        original_difficulty = None
        if difficulty is not None:
            original_difficulty = self.env.get_difficulty()
            self.env.set_difficulty(difficulty)
        
        # 暂存原始的人类推理设置
        original_reasoning = False
        if hasattr(self.agent, 'use_human_reasoning'):
            original_reasoning = self.agent.use_human_reasoning
            self.agent.use_human_reasoning = use_human_reasoning
        
        total_rewards = []
        wins = 0
        
        for i in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action_mask = self.env.get_valid_actions()
                
                # 使用act方法，可能会使用人类推理
                action = self.agent.act(state, action_mask, deterministic=deterministic)
                
                next_state, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if render:
                    self.env.render(disable_text_output=True)
            
            total_rewards.append(episode_reward)
            if info.get('win', False):
                wins += 1
        
        # 恢复原始难度
        if original_difficulty is not None:
            self.env.set_difficulty(original_difficulty)
        
        # 恢复原始人类推理设置
        if hasattr(self.agent, 'use_human_reasoning'):
            self.agent.use_human_reasoning = original_reasoning
        
        avg_reward = np.mean(total_rewards)
        win_rate = wins / num_episodes
        
        return avg_reward, win_rate
    
    def _update_env_difficulty(self, config):
        """更新环境难度配置"""
        self.env.width = config['width']
        self.env.height = config['height']
        self.env.num_mines = config['num_mines']
        print(f"环境难度更新为: 宽度={config['width']}, 高度={config['height']}, 地雷数={config['num_mines']}")
        
    def _create_new_agent_for_size(self, height, width):
        """
        为新的游戏板大小创建新的智能体
        """
        return DQNAgent(height, width)
    
    def _plot_training_curves(self, rewards, wins, steps):
        """绘制训练曲线"""
        window_size = min(100, len(rewards))
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_wins = np.convolve(wins, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # 奖励曲线
        ax1.plot(smoothed_rewards)
        ax1.set_title('Smoothed Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        
        # 胜率曲线
        ax2.plot(smoothed_wins)
        ax2.set_title('Smoothed Win Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim(0, 1)
        
        # 步数曲线
        ax3.plot(smoothed_steps)
        ax3.set_title('Smoothed Steps')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, 'training_curves.png'))
        plt.close()
    
    def _collect_pretrain_data_with_solver(self, num_samples=2000, games_per_difficulty=50):
        """
        使用专业求解器收集预训练数据
        """
        print(f"使用扫雷求解器收集预训练数据，目标样本数: {num_samples}")
        solver = MinesweeperSolver()
        collected_samples = 0
        total_games = 0
        
        # 使用当前环境的尺寸
        current_height = self.agent.height
        current_width = self.agent.width
        
        # 定义不同难度的游戏配置，但保持与当前尺寸一致
        game_configs = [
            {'width': current_width, 'height': current_height, 'num_mines': max(1, current_height * current_width // 10)},
            {'width': current_width, 'height': current_height, 'num_mines': max(2, current_height * current_width // 8)},
            {'width': current_width, 'height': current_height, 'num_mines': max(3, current_height * current_width // 6)}
        ]
        
        # 使用进度条
        pbar = tqdm(total=num_samples)
        
        # 对每个难度收集样本
        for config in game_configs:
            if collected_samples >= num_samples:
                break
                
            width, height, num_mines = config['width'], config['height'], config['num_mines']
            if width != current_width or height != current_height:
                print(f"警告：配置尺寸 {width}x{height} 与当前智能体尺寸 {current_width}x{current_height} 不匹配，已跳过")
                continue
                
            env = Minesweeper(width=width, height=height, num_mines=num_mines)
            
            for _ in range(games_per_difficulty):
                total_games += 1
                if collected_samples >= num_samples:
                    break
                    
                # 重置环境
                obs, _ = env.reset()
                board = obs["board"]
                action_mask = obs["action_mask"]
                
                # 使用求解器生成移动序列
                move_sequence = solver.generate_moves_sequence(board, action_mask, max_moves=50)
                
                # 收集数据
                for board_state, action in move_sequence:
                    # 确保状态表示使用正确的尺寸
                    if board_state.shape != (height, width):
                        print(f"警告：跳过形状不匹配的棋盘状态 {board_state.shape} != {(height, width)}")
                        continue
                        
                    # 预处理状态
                    state_dict = {"board": board_state, "action_mask": action_mask}
                    state, _ = self.agent.preprocess_state(state_dict)
                    
                    # 检查处理后的状态形状
                    expected_shape = (2, height, width)  # 对于双通道状态
                    if state.shape != expected_shape:
                        print(f"警告：预处理后状态形状不匹配 {state.shape} != {expected_shape}")
                        continue
                    
                    # 添加到预训练记忆
                    if hasattr(self.agent, 'pretrain_memory'):
                        self.agent.pretrain_memory.append((state, action))
                        collected_samples += 1
                        pbar.update(1)
                        
                        if collected_samples >= num_samples:
                            break
        
        pbar.close()
        
        # 验证收集的数据
        if hasattr(self.agent, 'pretrain_memory') and self.agent.pretrain_memory:
            first_shape = self.agent.pretrain_memory[0][0].shape
            inconsistent = False
            for i, (state, _) in enumerate(self.agent.pretrain_memory):
                if state.shape != first_shape:
                    print(f"警告：第 {i} 个样本的形状与首个样本不一致: {state.shape} vs {first_shape}")
                    inconsistent = True
                    break
            if not inconsistent:
                print(f"验证通过：所有 {len(self.agent.pretrain_memory)} 个样本具有相同的形状 {first_shape}")
        
        print(f"预训练数据收集完成：{len(self.agent.pretrain_memory)} 样本（来自 {total_games} 局游戏）")
    
    def save_model(self, filename):
        """
        保存模型到文件
        
        参数:
            filename: 保存的文件名（不包含路径）
        """
        # 确保models_dir目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 构建完整的文件路径
        file_path = os.path.join(self.models_dir, f"{filename}.pth")
        
        # 使用智能体的save_model方法保存模型
        self.agent.save_model(file_path)
        print(f"模型已保存到: {file_path}")
        
    def load_model(self, model_path):
        """
        从文件加载模型
        
        参数:
            model_path: 模型文件的完整路径
        """
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            return False
            
        try:
            self.agent.load_model(model_path)
            print(f"模型已从 {model_path} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False
    
    def compare_methods(self, num_episodes=100, difficulty=None, verbose=True):
        """
        对比不同方法的效果：纯人类逻辑、纯DQN和二者结合
        
        参数:
            num_episodes: 每种方法评估的回合数
            difficulty: 评估的难度级别
            verbose: 是否输出详细信息
            
        返回:
            对比结果的字典
        """
        if difficulty is not None:
            original_difficulty = self.env.get_difficulty()
            self.env.set_difficulty(difficulty)
            
        print(f"开始对比评估（每种方法 {num_episodes} 回合，难度: {self.env.get_difficulty()}）...")
        
        results = {}
        
        # 1. 评估纯人类逻辑
        print("评估纯人类逻辑...")
        if hasattr(self.agent, 'use_human_reasoning'):
            # 保存原始设置
            original_epsilon = self.agent.epsilon
            
            # 设置为总是使用人类推理（将探索率设为0，并强制使用人类推理）
            self.agent.epsilon = 0
            self.agent.use_human_reasoning = True
            self.agent.reasoning_used_count = 0
            self.agent.network_used_count = 0
            
            pure_logic_rewards, pure_logic_win_rate = self.evaluate(
                num_episodes=num_episodes, 
                use_human_reasoning=True,
                deterministic=True
            )
            
            # 计算人类推理使用率
            total_decisions = self.agent.reasoning_used_count + self.agent.network_used_count
            logic_usage_rate = self.agent.reasoning_used_count / total_decisions if total_decisions > 0 else 0
            
            results['pure_logic'] = {
                'avg_reward': pure_logic_rewards,
                'win_rate': pure_logic_win_rate,
                'logic_usage': logic_usage_rate
            }
            
            if verbose:
                print(f"纯人类逻辑 - 平均奖励: {pure_logic_rewards:.2f}, 胜率: {pure_logic_win_rate:.2f}")
                print(f"人类推理使用率: {logic_usage_rate:.2f} ({self.agent.reasoning_used_count}/{total_decisions})")
            
            # 重置计数器
            self.agent.reasoning_used_count = 0
            self.agent.network_used_count = 0
        
        # 2. 评估纯DQN
        print("评估纯DQN...")
        if hasattr(self.agent, 'use_human_reasoning'):
            # 禁用人类推理
            self.agent.use_human_reasoning = False
            self.agent.epsilon = 0  # 确定性策略
            
            pure_dqn_rewards, pure_dqn_win_rate = self.evaluate(
                num_episodes=num_episodes, 
                use_human_reasoning=False,
                deterministic=True
            )
            
            results['pure_dqn'] = {
                'avg_reward': pure_dqn_rewards,
                'win_rate': pure_dqn_win_rate
            }
            
            if verbose:
                print(f"纯DQN - 平均奖励: {pure_dqn_rewards:.2f}, 胜率: {pure_dqn_win_rate:.2f}")
        
        # 3. 评估二者结合
        print("评估人类逻辑 + DQN结合...")
        if hasattr(self.agent, 'use_human_reasoning'):
            # 启用人类推理并保持较低的探索率
            self.agent.use_human_reasoning = True
            self.agent.epsilon = 0.1  # 一定程度的探索
            self.agent.reasoning_used_count = 0
            self.agent.network_used_count = 0
            
            combined_rewards, combined_win_rate = self.evaluate(
                num_episodes=num_episodes, 
                use_human_reasoning=True,
                deterministic=False
            )
            
            # 计算人类推理使用率
            total_decisions = self.agent.reasoning_used_count + self.agent.network_used_count
            combined_logic_usage = self.agent.reasoning_used_count / total_decisions if total_decisions > 0 else 0
            
            results['combined'] = {
                'avg_reward': combined_rewards,
                'win_rate': combined_win_rate,
                'logic_usage': combined_logic_usage
            }
            
            if verbose:
                print(f"结合方法 - 平均奖励: {combined_rewards:.2f}, 胜率: {combined_win_rate:.2f}")
                print(f"人类推理使用率: {combined_logic_usage:.2f} ({self.agent.reasoning_used_count}/{total_decisions})")
            
            # 恢复原始设置
            self.agent.epsilon = original_epsilon
        
        # 恢复原始难度
        if difficulty is not None:
            self.env.set_difficulty(original_difficulty)
        
        # 打印对比总结
        if verbose:
            print("\n方法对比总结:")
            print(f"{'方法':<15} {'平均奖励':<12} {'胜率':<10} {'推理使用率':<12}")
            print("-" * 50)
            
            if 'pure_logic' in results:
                print(f"{'纯人类逻辑':<15} {results['pure_logic']['avg_reward']:<12.2f} "
                      f"{results['pure_logic']['win_rate']:<10.2f} "
                      f"{results['pure_logic']['logic_usage']:<12.2f}")
                
            if 'pure_dqn' in results:
                print(f"{'纯DQN':<15} {results['pure_dqn']['avg_reward']:<12.2f} "
                      f"{results['pure_dqn']['win_rate']:<10.2f} {'N/A':<12}")
                
            if 'combined' in results:
                print(f"{'结合方法':<15} {results['combined']['avg_reward']:<12.2f} "
                      f"{results['combined']['win_rate']:<10.2f} "
                      f"{results['combined']['logic_usage']:<12.2f}")
        
        return results 