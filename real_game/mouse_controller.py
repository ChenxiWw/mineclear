import pyautogui
import time
import random

class MouseController:
    """
    控制鼠标操作真实扫雷游戏
    """
    def __init__(self, screen_capture, move_speed=0.1, click_delay=0.1, human_like=True):
        """
        初始化鼠标控制器
        
        参数:
            screen_capture: ScreenCapture实例，用于获取游戏信息
            move_speed: 鼠标移动速度（秒）
            click_delay: 点击后的延迟（秒）
            human_like: 是否模拟人类鼠标操作（添加随机性）
        """
        self.screen_capture = screen_capture
        self.move_speed = move_speed
        self.click_delay = click_delay
        self.human_like = human_like
        
        # 确保pyautogui的失败保护
        pyautogui.FAILSAFE = True
    
    def move_to_cell(self, cell_x, cell_y):
        """
        将鼠标移动到指定格子
        
        参数:
            cell_x: 格子的x坐标
            cell_y: 格子的y坐标
        """
        # 获取格子在屏幕上的位置
        screen_x, screen_y = self.screen_capture.get_cell_screen_position(cell_x, cell_y)
        
        # 添加随机性，使移动更像人类
        if self.human_like:
            # 添加微小的随机偏移
            offset_x = random.uniform(-2, 2)
            offset_y = random.uniform(-2, 2)
            screen_x += offset_x
            screen_y += offset_y
            
            # 随机化移动速度
            move_speed = self.move_speed * random.uniform(0.8, 1.2)
        else:
            move_speed = self.move_speed
        
        # 移动鼠标
        pyautogui.moveTo(screen_x, screen_y, duration=move_speed)
    
    def left_click(self):
        """执行左键点击"""
        pyautogui.click(button='left')
        
        # 随机化点击后延迟
        if self.human_like:
            delay = self.click_delay * random.uniform(0.8, 1.5)
        else:
            delay = self.click_delay
            
        time.sleep(delay)
    
    def right_click(self):
        """执行右键点击（标记旗子）"""
        pyautogui.click(button='right')
        
        # 随机化点击后延迟
        if self.human_like:
            delay = self.click_delay * random.uniform(0.8, 1.5)
        else:
            delay = self.click_delay
            
        time.sleep(delay)
    
    def click_cell(self, cell_x, cell_y, right_click=False):
        """
        点击指定格子
        
        参数:
            cell_x: 格子的x坐标
            cell_y: 格子的y坐标
            right_click: 是否使用右键（标记旗子）
        """
        # 移动到格子
        self.move_to_cell(cell_x, cell_y)
        
        # 点击
        if right_click:
            self.right_click()
        else:
            self.left_click()
    
    def perform_action(self, action, board_width):
        """
        执行智能体选择的动作
        
        参数:
            action: 动作索引 (0 到 board_size-1)
            board_width: 游戏板宽度
            
        返回:
            是否成功执行动作
        """
        try:
            # 转换动作索引为二维坐标
            cell_y, cell_x = divmod(action, board_width)
            
            # 点击格子
            self.click_cell(cell_x, cell_y)
            
            return True
        except Exception as e:
            print(f"执行动作时出错: {e}")
            return False 