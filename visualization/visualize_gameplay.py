import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

class VisualizeGameplay:
    """可视化智能体游戏过程"""
    def __init__(self, env, agent, save_dir="visualizations"):
        """
        初始化可视化器
        
        参数:
            env: 游戏环境
            agent: 智能体
            save_dir: 保存可视化结果的目录
        """
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
    def run(self, num_games=5, delay=0.5, save_animation=False):
        """
        可视化多局游戏
        
        参数:
            num_games: 游戏局数
            delay: 每步之间的延迟（秒）
            save_animation: 是否保存动画
        """
        for game_idx in range(num_games):
            print(f"游戏 {game_idx+1}/{num_games}")
            self.visualize_game(game_idx, delay, save_animation)
            time.sleep(1.0)  # 游戏之间的间隔
            
    def visualize_game(self, game_idx, delay=0.5, save_animation=False):
        """
        可视化单局游戏
        
        参数:
            game_idx: 游戏索引
            delay: 每步之间的延迟（秒）
            save_animation: 是否保存动画
        """
        # 初始化数据收集
        states = []
        actions = []
        rewards = []
        total_reward = 0
        
        # 重置环境
        obs, _ = self.env.reset()
        state, action_mask = self.agent.preprocess_state(obs)
        states.append(obs['board'].copy())
        
        # 首次渲染
        if hasattr(self.env, 'render'):
            self.env.render()
        else:
            self._render_board(states[-1], title=f"游戏 {game_idx+1} - 开始")
        
        # 游戏循环
        done = False
        step = 0
        
        while not done and step < 100:  # 限制最大步数
            # 智能体选择动作
            if hasattr(self.agent, 'select_action'):
                action_result = self.agent.select_action(state, action_mask)
                
                # 处理不同类型的返回值
                if isinstance(action_result, tuple):
                    action = action_result[0]  # PPO返回(action, prob, val)
                else:
                    action = action_result  # DQN只返回action
            else:
                # 随机动作作为后备
                valid_actions = np.where(obs['action_mask'] == 1)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            
            # 记录动作
            actions.append(action)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 记录数据
            rewards.append(reward)
            total_reward += reward
            states.append(next_obs['board'].copy())
            
            # 获取下一个状态
            next_state, next_action_mask = self.agent.preprocess_state(next_obs)
            
            # 确定游戏是否结束
            done = terminated or truncated
            
            # 渲染环境
            if hasattr(self.env, 'render'):
                self.env.render()
            else:
                title = f"游戏 {game_idx+1} - 步骤 {step+1} - 奖励: {reward:.2f}"
                self._render_board(states[-1], title=title)
            
            # 延迟以便观察
            time.sleep(delay)
            
            # 更新状态
            state = next_state
            action_mask = next_action_mask
            step += 1
            
            # 如果游戏结束，显示结果
            if done:
                result = "胜利" if info.get("win", False) else "失败"
                print(f"游戏 {game_idx+1} {result}! 步数: {step}, 总奖励: {total_reward:.2f}")
                
                if not hasattr(self.env, 'render'):
                    title = f"游戏 {game_idx+1} - {result}! 步数: {step}, 总奖励: {total_reward:.2f}"
                    self._render_board(states[-1], title=title)
                    plt.pause(2.0)  # 游戏结束时显示更长时间
        
        # 保存动画
        if save_animation and len(states) > 1:
            self._save_animation(states, actions, rewards, game_idx)
            
        return total_reward, step, info.get("win", False)
    
    def _render_board(self, board, title=""):
        """
        使用Matplotlib渲染游戏板
        
        参数:
            board: 游戏板
            title: 图形标题
        """
        # 创建颜色映射
        cmap = plt.cm.tab10
        norm = colors.Normalize(vmin=-1, vmax=9)
        
        # 创建图形
        plt.figure(figsize=(8, 8))
        plt.imshow(board, cmap=cmap, norm=norm)
        
        # 添加网格
        plt.grid(color='black', linestyle='-', linewidth=1.5)
        
        # 在每个格子中添加数字
        height, width = board.shape
        for i in range(height):
            for j in range(width):
                if board[i, j] == -1:
                    plt.text(j, i, "?", fontsize=14, ha='center', va='center')
                elif board[i, j] == 0:
                    plt.text(j, i, "", fontsize=14, ha='center', va='center')
                elif board[i, j] == 9:
                    plt.text(j, i, "💣", fontsize=14, ha='center', va='center')
                else:
                    plt.text(j, i, str(int(board[i, j])), fontsize=14, ha='center', va='center')
        
        # 设置标题和轴
        plt.title(title)
        plt.xticks(np.arange(-0.5, width, 1), [])
        plt.yticks(np.arange(-0.5, height, 1), [])
        
        # 显示图形
        plt.pause(0.01)
        plt.clf()
        plt.close()
        
    def _save_animation(self, states, actions, rewards, game_idx):
        """
        保存游戏动画
        
        参数:
            states: 游戏状态列表
            actions: 动作列表
            rewards: 奖励列表
            game_idx: 游戏索引
        """
        height, width = states[0].shape
        fig, ax = plt.subplots(figsize=(10, 10))
        
        cmap = plt.cm.tab10
        norm = colors.Normalize(vmin=-1, vmax=9)
        
        # 初始化图形
        img = ax.imshow(states[0], cmap=cmap, norm=norm)
        ax.grid(color='black', linestyle='-', linewidth=1.5)
        title = ax.set_title("开始游戏")
        
        def update(frame):
            # 更新图像数据
            img.set_array(states[frame])
            
            # 更新标题
            if frame > 0:
                action = actions[frame-1]
                reward = rewards[frame-1]
                action_y, action_x = action // width, action % width
                title.set_text(f"步骤 {frame} - 动作: ({action_y},{action_x}) - 奖励: {reward:.2f}")
            else:
                title.set_text(f"开始游戏")
                
            # 添加数字
            for i in range(height):
                for j in range(width):
                    for txt in ax.texts:
                        ax.texts.remove(txt)
                        break
            
            for i in range(height):
                for j in range(width):
                    if states[frame][i, j] == -1:
                        ax.text(j, i, "?", fontsize=14, ha='center', va='center')
                    elif states[frame][i, j] == 0:
                        ax.text(j, i, "", fontsize=14, ha='center', va='center')
                    elif states[frame][i, j] == 9:
                        ax.text(j, i, "💣", fontsize=14, ha='center', va='center')
                    else:
                        ax.text(j, i, str(int(states[frame][i, j])), fontsize=14, ha='center', va='center')
            
            return img, title
        
        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=len(states),
                                     interval=500, blit=False)
        
        # 保存动画
        filename = os.path.join(self.save_dir, f"game_{game_idx+1}.mp4")
        ani.save(filename, writer='ffmpeg', fps=2)
        plt.close() 