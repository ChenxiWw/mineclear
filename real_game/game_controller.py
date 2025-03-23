import os
import time
import numpy as np
from .screen_capture import ScreenCapture
from .mouse_controller import MouseController

class RealGameController:
    """实际扫雷游戏控制器"""
    def __init__(self, agent, args):
        """
        初始化游戏控制器
        
        参数:
            agent: 智能体
            args: 命令行参数
        """
        self.agent = agent
        self.args = args
        self.cell_size = args.cell_size if hasattr(args, 'cell_size') else 16
        self.num_games = args.num_games if hasattr(args, 'num_games') else 5
        self.delay = args.delay if hasattr(args, 'delay') else 0.5
        
        # 创建屏幕捕获器
        self.screen_capture = ScreenCapture(cell_size=self.cell_size)
        
        # 创建鼠标控制器
        self.mouse_controller = None  # 将在校准后初始化
    
    def run(self):
        """运行游戏控制"""
        print("请确保扫雷游戏已打开")
        print("此模式将尝试捕获屏幕上的扫雷游戏并自动操作")
        print("按Enter键继续...")
        input()
        
        # 选择游戏区域
        self.screen_capture.select_game_region()
        
        # 校准游戏网格
        print("校准游戏网格...")
        self.screen_capture.calibrate()
        
        # 获取游戏大小
        width = self.screen_capture.grid_width
        height = self.screen_capture.grid_height
        print(f"检测到游戏大小: {width}x{height}")
        
        # 创建鼠标控制器
        self.mouse_controller = MouseController(
            self.screen_capture, 
            move_speed=0.1,
            click_delay=0.2
        )
        
        print("准备完毕，将在3秒后开始操作...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # 游戏循环
        try:
            for game in range(self.num_games):
                self._play_game(game)
                time.sleep(2.0)  # 游戏间的间隔
                
        except KeyboardInterrupt:
            print("用户中断，退出程序")
        except Exception as e:
            print(f"发生错误: {e}")
        
        print("实际游戏控制结束")
        return True
    
    def _play_game(self, game_idx):
        """
        玩一局游戏
        
        参数:
            game_idx: 游戏索引
        """
        print(f"开始游戏 {game_idx+1}/{self.num_games}")
        
        # 捕获屏幕并识别游戏板
        board, _ = self.screen_capture.recognize_board()
        
        # 游戏宽度和高度
        width = self.screen_capture.grid_width
        height = self.screen_capture.grid_height
        
        # 创建观察
        observation = {
            "board": board,
            "mines": np.zeros_like(board),
            "action_mask": np.ones(width * height, dtype=np.int8)
        }
        
        # 更新动作掩码，已挖掘格子不能再点
        for y in range(height):
            for x in range(width):
                if board[y, x] != -1:
                    action_idx = y * width + x
                    observation["action_mask"][action_idx] = 0
        
        # 智能体选择动作
        state, action_mask = self.agent.preprocess_state(observation)
        action_result = self.agent.select_action(state, action_mask)
        
        # 处理返回值
        if isinstance(action_result, tuple):
            action = action_result[0]  # PPO返回(action, prob, val)
        else:
            action = action_result  # DQN只返回action
        
        # 执行动作
        success = self.mouse_controller.perform_action(action, width)
        
        if not success:
            print("执行动作失败，退出游戏")
            return
        
        # 等待一段时间让游戏状态更新
        time.sleep(0.5)
        
        # 继续直到游戏结束或最大步数
        for step in range(50):
            # 再次捕获屏幕
            board, _ = self.screen_capture.recognize_board()
            
            # 检查游戏是否结束
            if np.any(board == 9):
                print(f"游戏 {game_idx+1} 失败！步数: {step+1}")
                time.sleep(1.0)
                break
            
            # 计算已挖掘的格子数
            num_mines = getattr(self.args, 'num_mines', width * height // 5)  # 默认估计地雷数
            revealed = np.sum(board >= 0)
            if revealed >= (width * height - num_mines):
                print(f"游戏 {game_idx+1} 胜利！步数: {step+1}")
                time.sleep(1.0)
                break
            
            # 更新观察和动作掩码
            observation["board"] = board
            for y in range(height):
                for x in range(width):
                    if board[y, x] != -1:
                        action_idx = y * width + x
                        observation["action_mask"][action_idx] = 0
            
            # 智能体选择下一个动作
            state, action_mask = self.agent.preprocess_state(observation)
            action_result = self.agent.select_action(state, action_mask)
            
            # 处理返回值
            if isinstance(action_result, tuple):
                action = action_result[0]
            else:
                action = action_result
            
            # 执行动作
            success = self.mouse_controller.perform_action(action, width)
            
            if not success:
                print("执行动作失败，退出游戏")
                break
            
            # 等待一段时间
            time.sleep(self.delay) 