import pygame
import numpy as np
import sys
import time

class PygameRenderer:
    """
    使用Pygame渲染扫雷游戏和AI决策过程
    """
    # 颜色常量
    COLORS = {
        'background': (220, 220, 220),
        'grid': (180, 180, 180),
        'covered': (200, 200, 200),
        'uncovered': (255, 255, 255),
        'mine': (255, 0, 0),
        'flag': (255, 165, 0),
        'highlight': (100, 255, 100),
        'text': [(0, 0, 0),          # 0 - 黑色（空）
                 (0, 0, 255),        # 1 - 蓝色
                 (0, 128, 0),        # 2 - 绿色
                 (255, 0, 0),        # 3 - 红色
                 (0, 0, 128),        # 4 - 深蓝
                 (128, 0, 0),        # 5 - 深红
                 (0, 128, 128),      # 6 - 青色
                 (0, 0, 0),          # 7 - 黑色
                 (128, 128, 128)]    # 8 - 灰色
    }
    
    def __init__(self, env, agent=None, cell_size=50, fps=10, delay=0.5):
        """
        初始化Pygame渲染器
        
        参数:
            env: 扫雷游戏环境
            agent: DQN智能体（可选，用于可视化AI决策）
            cell_size: 格子大小（像素）
            fps: 帧率
            delay: AI决策间的延迟（秒）
        """
        self.env = env
        self.agent = agent
        self.cell_size = cell_size
        self.fps = fps
        self.delay = delay
        
        # 计算窗口大小
        self.width = env.width * cell_size
        self.height = env.height * cell_size
        
        # 初始化Pygame
        pygame.init()
        pygame.display.set_caption("扫雷AI")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # 加载字体
        self.font = pygame.font.SysFont('Arial', cell_size // 2)
        
        # 上次AI决策的格子位置
        self.last_action = None
    
    def render(self, observation=None, highlight_action=None):
        """
        渲染当前游戏状态
        
        参数:
            observation: 游戏环境的观察（如果为None，则使用env.board）
            highlight_action: 要高亮显示的动作（格子索引）
        """
        if observation is None:
            board = self.env.board
            mines = self.env.mines if self.env.done else None
        else:
            board = observation["board"]
            mines = observation["mines"] if np.any(observation["mines"]) else None
            
        # 填充背景
        self.screen.fill(self.COLORS['background'])
        
        # 绘制格子
        for y in range(self.env.height):
            for x in range(self.env.width):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                # 计算格子的1D索引
                cell_idx = y * self.env.width + x
                
                # 绘制格子背景
                if board[y, x] == -1:  # 未挖掘
                    pygame.draw.rect(self.screen, self.COLORS['covered'], rect)
                else:  # 已挖掘
                    pygame.draw.rect(self.screen, self.COLORS['uncovered'], rect)
                
                # 高亮显示最后一个AI操作
                if highlight_action is not None and cell_idx == highlight_action:
                    highlight_rect = pygame.Rect(
                        x * self.cell_size + 2, 
                        y * self.cell_size + 2, 
                        self.cell_size - 4, 
                        self.cell_size - 4
                    )
                    pygame.draw.rect(self.screen, self.COLORS['highlight'], highlight_rect)
                
                # 绘制格子内容
                if board[y, x] == 9 or (mines is not None and mines[y, x] == 1):
                    # 绘制地雷
                    pygame.draw.circle(
                        self.screen, 
                        self.COLORS['mine'],
                        (x * self.cell_size + self.cell_size // 2, 
                         y * self.cell_size + self.cell_size // 2),
                        self.cell_size // 3
                    )
                elif 1 <= board[y, x] <= 8:
                    # 绘制数字
                    text = self.font.render(str(board[y, x]), True, self.COLORS['text'][board[y, x]])
                    text_rect = text.get_rect(center=(
                        x * self.cell_size + self.cell_size // 2,
                        y * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text, text_rect)
                
                # 绘制格子边框
                pygame.draw.rect(self.screen, self.COLORS['grid'], rect, 1)
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def run_game_with_ai(self, num_episodes=1, max_steps=100):
        """
        运行游戏，由AI控制
        
        参数:
            num_episodes: 运行的回合数
            max_steps: 每回合的最大步数
        """
        if self.agent is None:
            raise ValueError("需要提供AI智能体才能运行AI决策模式")
        
        for episode in range(num_episodes):
            print(f"开始回合 {episode+1}/{num_episodes}")
            
            # 重置环境
            obs, _ = self.env.reset()
            state, action_mask = self.agent.preprocess_state(obs)
            
            # 绘制初始状态
            self.render(obs)
            
            total_reward = 0
            for step in range(max_steps):
                # 检查事件（如关闭窗口）
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                # AI选择动作
                action = self.agent.select_action(state, action_mask)
                
                # 高亮显示AI选择的动作
                self.last_action = action
                self.render(obs, highlight_action=action)
                
                # 等待一段时间，以便观察
                time.sleep(self.delay)
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state, next_action_mask = self.agent.preprocess_state(next_obs)
                
                # 更新状态
                total_reward += reward
                state = next_state
                action_mask = next_action_mask
                obs = next_obs
                
                # 渲染新状态
                self.render(obs, highlight_action=action)
                
                if terminated or truncated:
                    time.sleep(1.0)  # 游戏结束时多等待一会
                    if info.get("win", False):
                        print(f"回合 {episode+1} 胜利！奖励: {total_reward:.2f}, 步数: {step+1}")
                    else:
                        print(f"回合 {episode+1} 失败！奖励: {total_reward:.2f}, 步数: {step+1}")
                    break
            
            # 如果回合没有正常结束
            else:
                print(f"回合 {episode+1} 达到最大步数限制。奖励: {total_reward:.2f}")
    
    def run_game_manual(self):
        """
        手动玩游戏（点击格子）
        """
        # 重置环境
        self.env.reset()
        
        running = True
        while running:
            # 绘制当前状态
            self.render()
            
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # 左键点击
                    if event.button == 1:
                        # 获取点击位置
                        pos = pygame.mouse.get_pos()
                        x = pos[0] // self.cell_size
                        y = pos[1] // self.cell_size
                        
                        # 计算动作索引
                        action = y * self.env.width + x
                        
                        # 执行动作
                        _, reward, terminated, truncated, info = self.env.step(action)
                        
                        if terminated:
                            if info.get("win", False):
                                print("你赢了！")
                            else:
                                print("游戏结束！")
                            # 绘制最终状态
                            self.render()
                            time.sleep(2)
                            # 重置游戏
                            self.env.reset()
            
            # 控制帧率
            self.clock.tick(self.fps)
        
        pygame.quit()
        sys.exit() 