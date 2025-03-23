import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MatplotlibRenderer:
    """
    使用Matplotlib渲染扫雷游戏和AI决策过程
    """
    def __init__(self, env, agent=None, figsize=(8, 8), interval=500):
        """
        初始化Matplotlib渲染器
        
        参数:
            env: 扫雷游戏环境
            agent: DQN智能体（可选，用于可视化AI决策）
            figsize: 图的大小
            interval: 动画更新间隔（毫秒）
        """
        self.env = env
        self.agent = agent
        self.figsize = figsize
        self.interval = interval
        
        # 用于存储游戏状态历史
        self.history = []
        self.highlight_actions = []
        
        # 颜色映射
        self.cmap = self._create_colormap()
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.im = None
        self.highlight = None
        self.animation = None
    
    def _create_colormap(self):
        """创建自定义颜色映射"""
        # 创建基本颜色映射（用于-1到9的值）
        colors = [
            (0.8, 0.8, 0.8),  # -1: 未挖掘（灰色）
            (1.0, 1.0, 1.0),  # 0: 空白（白色）
            (0.0, 0.0, 1.0),  # 1: 蓝色
            (0.0, 0.5, 0.0),  # 2: 绿色
            (1.0, 0.0, 0.0),  # 3: 红色
            (0.0, 0.0, 0.5),  # 4: 深蓝
            (0.5, 0.0, 0.0),  # 5: 深红
            (0.0, 0.5, 0.5),  # 6: 青色
            (0.0, 0.0, 0.0),  # 7: 黑色
            (0.5, 0.5, 0.5),  # 8: 灰色
            (1.0, 0.0, 0.0),  # 9: 地雷（红色）
        ]
        
        # 创建ListedColormap
        return mcolors.ListedColormap(colors)
    
    def render_static(self, observation=None, highlight_action=None):
        """
        静态渲染游戏状态
        
        参数:
            observation: 游戏环境的观察（如果为None，则使用env.board）
            highlight_action: 要高亮显示的动作（格子索引）
        """
        if observation is None:
            board = self.env.board
        else:
            board = observation["board"]
        
        # 创建图形
        plt.figure(figsize=self.figsize)
        
        # 绘制游戏板
        plt.imshow(board, cmap=self.cmap, vmin=-1, vmax=9)
        
        # 添加网格
        plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        plt.xticks(np.arange(-.5, self.env.width, 1), [])
        plt.yticks(np.arange(-.5, self.env.height, 1), [])
        
        # 在格子中显示数字
        for y in range(self.env.height):
            for x in range(self.env.width):
                if 1 <= board[y, x] <= 8:
                    plt.text(x, y, str(board[y, x]), ha='center', va='center', fontsize=20)
                elif board[y, x] == 9:
                    plt.text(x, y, 'X', ha='center', va='center', fontsize=20, color='white')
        
        # 高亮显示AI选择的动作
        if highlight_action is not None:
            y, x = divmod(highlight_action, self.env.width)
            highlight = plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=False, edgecolor='lime', linewidth=3)
            plt.gca().add_patch(highlight)
        
        plt.tight_layout()
        plt.show()
    
    def record_step(self, observation, highlight_action=None):
        """
        记录一步游戏状态，用于之后的动画
        
        参数:
            observation: 游戏环境的观察
            highlight_action: 要高亮显示的动作（格子索引）
        """
        self.history.append(observation["board"].copy())
        self.highlight_actions.append(highlight_action)
    
    def _update_animation(self, frame):
        """动画帧更新函数"""
        # 清除之前的高亮
        if self.highlight is not None:
            self.highlight.remove()
            self.highlight = None
        
        # 更新图像
        self.im.set_array(self.history[frame])
        
        # 添加高亮
        if self.highlight_actions[frame] is not None:
            y, x = divmod(self.highlight_actions[frame], self.env.width)
            self.highlight = plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=False, edgecolor='lime', linewidth=3)
            self.ax.add_patch(self.highlight)
        
        # 更新格子中的数字
        for item in self.ax.texts:
            item.remove()
        
        board = self.history[frame]
        for y in range(self.env.height):
            for x in range(self.env.width):
                if 1 <= board[y, x] <= 8:
                    self.ax.text(x, y, str(board[y, x]), ha='center', va='center', fontsize=20)
                elif board[y, x] == 9:
                    self.ax.text(x, y, 'X', ha='center', va='center', fontsize=20, color='white')
        
        return [self.im] + ([] if self.highlight is None else [self.highlight])
    
    def create_animation(self):
        """
        根据记录的历史创建动画
        
        返回:
            动画对象
        """
        if not self.history:
            raise ValueError("没有记录的游戏历史，请先调用 record_step")
        
        # 重置图形
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_title("扫雷AI决策过程")
        
        # 初始状态
        self.im = self.ax.imshow(self.history[0], cmap=self.cmap, vmin=-1, vmax=9)
        
        # 添加网格
        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        self.ax.set_xticks(np.arange(-.5, self.env.width, 1))
        self.ax.set_yticks(np.arange(-.5, self.env.height, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # 创建动画
        self.animation = FuncAnimation(
            self.fig, self._update_animation, frames=len(self.history),
            interval=self.interval, blit=True, repeat=True
        )
        
        plt.tight_layout()
        return self.animation
    
    def save_animation(self, filename="minesweeper_ai.gif"):
        """保存动画为GIF文件"""
        if self.animation is None:
            self.create_animation()
        
        self.animation.save(filename, writer='pillow', fps=1000/self.interval)
        print(f"动画已保存到 {filename}")
    
    def show_animation(self):
        """显示动画"""
        if self.animation is None:
            self.create_animation()
        
        plt.show()
    
    def run_game_with_ai(self, num_episodes=1, max_steps=100, save_animation=False):
        """
        运行游戏，由AI控制，并记录过程用于创建动画
        
        参数:
            num_episodes: 运行的回合数
            max_steps: 每回合的最大步数
            save_animation: 是否保存动画到文件
        """
        if self.agent is None:
            raise ValueError("需要提供AI智能体才能运行AI决策模式")
        
        for episode in range(num_episodes):
            print(f"开始回合 {episode+1}/{num_episodes}")
            
            # 重置历史记录和环境
            self.history = []
            self.highlight_actions = []
            obs, _ = self.env.reset()
            state, action_mask = self.agent.preprocess_state(obs)
            
            # 记录初始状态
            self.record_step(obs)
            
            total_reward = 0
            for step in range(max_steps):
                # AI选择动作
                action = self.agent.select_action(state, action_mask)
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state, next_action_mask = self.agent.preprocess_state(next_obs)
                
                # 记录状态
                self.record_step(next_obs, action)
                
                # 更新状态
                total_reward += reward
                state = next_state
                action_mask = next_action_mask
                
                if terminated or truncated:
                    if info.get("win", False):
                        print(f"回合 {episode+1} 胜利！奖励: {total_reward:.2f}, 步数: {step+1}")
                    else:
                        print(f"回合 {episode+1} 失败！奖励: {total_reward:.2f}, 步数: {step+1}")
                    break
            
            # 创建并显示/保存动画
            animation = self.create_animation()
            if save_animation:
                self.save_animation(f"minesweeper_episode_{episode+1}.gif")
            else:
                self.show_animation() 