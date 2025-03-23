import os
import numpy as np
import time

# 条件导入GUI相关库
try:
    import cv2
    import pyautogui
    HAS_GUI = True
except ImportError as e:
    import warnings
    warnings.warn(f"无法导入GUI相关库 ({e})，ScreenCapture将不可用")
    HAS_GUI = False

class ScreenCapture:
    """屏幕捕获，从实际游戏屏幕中获取图像"""
    def __init__(self, cell_size=16, debug=False):
        """
        初始化屏幕捕获器
        
        参数:
            cell_size: 游戏中每个格子的像素大小
            debug: 是否启用调试模式
        """
        if not HAS_GUI:
            raise RuntimeError("ScreenCapture无法在无GUI环境中使用")
            
        self.cell_size = cell_size
        self.debug = debug
        
        # 游戏界面坐标（将通过校准设置）
        self.game_region = None  # (left, top, width, height)
        
        # 列举所有可能的单元格图案
        self.patterns = {
            'unknown': None,  # 未挖掘的格子
            'empty': None,    # 空格子（周围无地雷）
            '1': None,        # 数字1格子
            '2': None,        # 数字2格子
            # ... 其他数字
            'flag': None,     # 标记为地雷的格子
            'mine': None      # 地雷
        }
    
    def calibrate(self):
        """
        校准游戏界面位置
        手动操作：用户需要点击游戏区域的左上角和右下角
        """
        print("请点击游戏区域的左上角...")
        time.sleep(2)  # 给用户时间移动鼠标
        left_top = pyautogui.position()
        
        print("请点击游戏区域的右下角...")
        time.sleep(2)
        right_bottom = pyautogui.position()
        
        # 计算游戏区域
        width = right_bottom[0] - left_top[0]
        height = right_bottom[1] - left_top[1]
        self.game_region = (left_top[0], left_top[1], width, height)
        
        # 计算棋盘尺寸
        self.board_width = int(width / self.cell_size)
        self.board_height = int(height / self.cell_size)
        
        print(f"校准完成。游戏区域: {self.game_region}")
        print(f"棋盘尺寸: {self.board_width}x{self.board_height}")
        
        return self.board_width, self.board_height
    
    def capture_board(self):
        """
        捕获当前游戏棋盘状态
        
        返回:
            board: 游戏棋盘状态的numpy数组
            action_mask: 可执行动作的掩码
        """
        if self.game_region is None:
            raise ValueError("请先校准游戏区域")
        
        # 截取游戏区域
        screenshot = pyautogui.screenshot(region=self.game_region)
        image = np.array(screenshot)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 初始化棋盘和掩码
        board = np.full((self.board_height, self.board_width), -1, dtype=np.int8)  # 全部初始化为未挖掘
        action_mask = np.ones(self.board_height * self.board_width, dtype=np.int8)  # 所有动作可用
        
        # 遍历每个格子，识别其状态
        for y in range(self.board_height):
            for x in range(self.board_width):
                # 计算格子的像素区域
                cell_x = x * self.cell_size
                cell_y = y * self.cell_size
                cell_img = image[cell_y:cell_y+self.cell_size, cell_x:cell_x+self.cell_size]
                
                # 识别格子状态
                cell_state = self._recognize_cell(cell_img)
                
                # 更新棋盘
                if cell_state == 'unknown':
                    board[y, x] = -1  # 未挖掘
                elif cell_state == 'empty':
                    board[y, x] = 0   # 空格子
                elif cell_state == 'flag':
                    board[y, x] = 9   # 标记为地雷
                elif cell_state == 'mine':
                    board[y, x] = 9   # 地雷
                elif cell_state in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    board[y, x] = int(cell_state)  # 数字格子
                
                # 更新动作掩码
                if cell_state != 'unknown':
                    action_mask[y * self.board_width + x] = 0  # 已挖掘或标记的格子不能再点击
        
        # 返回状态
        return {
            "board": board,
            "action_mask": action_mask
        }
    
    def _recognize_cell(self, cell_img):
        """
        识别单元格状态
        
        参数:
            cell_img: 单元格图像
            
        返回:
            状态: unknown, empty, 1-8, flag, mine
        """
        # 这里需要实现图像识别逻辑
        # 简单示例：基于颜色判断
        # 实际项目中应该使用模板匹配或机器学习方法
        
        # 转换为灰度图
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        
        # 检测各种状态
        # 注意：以下仅为示例，实际情况需要根据游戏界面调整阈值
        
        # 检测未挖掘格子（通常颜色较深）
        if np.mean(gray) < 100:
            return 'unknown'
        
        # 检测空格子（通常颜色较亮）
        if np.mean(gray) > 220:
            return 'empty'
        
        # 检测数字（可以通过颜色或模板匹配）
        # 简化示例，实际应使用OCR或颜色分析
        blue = np.mean(cell_img[:,:,0])
        green = np.mean(cell_img[:,:,1])
        red = np.mean(cell_img[:,:,2])
        
        if blue > red and blue > green:
            return '1'  # 蓝色通常是数字1
        if green > red and green > blue:
            return '2'  # 绿色通常是数字2
        if red > blue and red > green:
            return '3'  # 红色通常是数字3
        
        # 其他状态检测...
        
        # 默认返回未知
        return 'unknown'
    
    def get_cell_position(self, action):
        """
        获取动作对应格子的屏幕坐标
        
        参数:
            action: 动作索引
            
        返回:
            (x, y): 屏幕坐标
        """
        if self.game_region is None:
            raise ValueError("请先校准游戏区域")
        
        row, col = divmod(action, self.board_width)
        
        # 计算格子中心点的屏幕坐标
        screen_x = self.game_region[0] + col * self.cell_size + self.cell_size // 2
        screen_y = self.game_region[1] + row * self.cell_size + self.cell_size // 2
        
        return (screen_x, screen_y)