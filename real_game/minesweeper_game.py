#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pygame
from pygame.locals import *

class MinesweeperGame:
    """
    使用Pygame实现的扫雷游戏，允许AI或人类玩家交互
    """
    # 颜色定义
    COLORS = {
        'background': (220, 220, 220),
        'grid': (180, 180, 180),
        'closed': (200, 200, 200),
        'opened': (255, 255, 255),
        'mine': (255, 0, 0),
        'flag': (255, 165, 0),
        'text': [(0, 0, 0), (0, 0, 255), (0, 128, 0), (255, 0, 0), 
                (0, 0, 128), (128, 0, 0), (0, 128, 128), (0, 0, 0), (128, 128, 128)]
    }
    
    # 单元格状态
    CELL_CLOSED = 0
    CELL_OPENED = 1
    CELL_FLAGGED = 2
    
    def __init__(self, width=9, height=9, num_mines=10, cell_size=40, 
                 ai_agent=None, use_human_reasoning=True, control_keys=None):
        """
        初始化扫雷游戏
        
        参数:
            width (int): 游戏板宽度
            height (int): 游戏板高度
            num_mines (int): 地雷数量
            cell_size (int): 单元格像素大小
            ai_agent: 可选的AI智能体
            use_human_reasoning (bool): 是否使用人类推理能力
            control_keys (dict): 自定义控制键，格式为 {'ai': 'a', 'reset': 'r', 'quit': 'q'}
        """
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.cell_size = cell_size
        self.ai_agent = ai_agent
        self.use_human_reasoning = use_human_reasoning
        
        # 设置控制键
        default_keys = {'ai': 'a', 'reset': 'r', 'quit': 'q'}
        self.control_keys = control_keys if control_keys else default_keys
        
        # 将控制键转换为小写，并映射到ASCII码和pygame键值
        self.key_mappings = {}
        for action, key in self.control_keys.items():
            key = key.lower()
            # 手动映射pygame键值
            pygame_key = None
            if key == 'a':
                pygame_key = pygame.K_a
            elif key == 'r':
                pygame_key = pygame.K_r
            elif key == 'q':
                pygame_key = pygame.K_q
            elif key == 's':
                pygame_key = pygame.K_s
            elif key == 'n':
                pygame_key = pygame.K_n
            elif key == 'e':
                pygame_key = pygame.K_e
                
            self.key_mappings[action] = {
                'key': key,
                'ascii': ord(key),
                'pygame_key': pygame_key
            }
        
        # 确保pygame已初始化
        if not pygame.get_init():
            pygame.init()
        
        # 确保显示模块已初始化
        if not pygame.display.get_init():
            pygame.display.init()
            
        # 确保事件模块已初始化    
        pygame.event.set_allowed([QUIT, KEYDOWN, KEYUP, MOUSEBUTTONDOWN, MOUSEBUTTONUP])
        
        self.screen_width = width * cell_size + 1
        self.screen_height = height * cell_size + 1
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('MINECLEAR')
        
        # 使用支持中文的字体
        try:
            # 尝试使用系统中文字体
            font_options = [
                "SimHei", "Microsoft YaHei", "SimSun", "NSimSun", "FangSong", 
                "KaiTi", "Arial Unicode MS", "WenQuanYi Micro Hei"
            ]
            
            self.font = None
            for font_name in font_options:
                try:
                    self.font = pygame.font.SysFont(font_name, int(cell_size * 0.75))
                    # 测试是否能正确渲染中文
                    test_text = self.font.render("测试", True, (0, 0, 0))
                    if test_text.get_width() > 10:  # 如果宽度合理，说明字体有效
                        break
                except:
                    continue
                    
            # 如果所有系统字体都失败，回退到默认字体
            if self.font is None:
                self.font = pygame.font.Font(None, int(cell_size * 0.75))
                print("警告: 未找到支持中文的字体，界面可能会显示异常")
        except Exception as e:
            print(f"字体加载错误: {str(e)}")
            self.font = pygame.font.Font(None, int(cell_size * 0.75))
        
        # 创建一个专用于数字显示的字体
        self.number_font = pygame.font.Font(None, int(cell_size * 0.75))
        
        # 游戏状态
        self.game_over = False
        self.win = False
        self.first_click = True
        
        # 初始化游戏板
        self.reset_game()
    
    def reset_game(self):
        """重置游戏状态并创建新的游戏板"""
        # 游戏板初始化
        self.board = np.zeros((self.height, self.width), dtype=np.int8)  # 储存地雷位置
        self.visible = np.zeros((self.height, self.width), dtype=np.int8)  # 储存单元格状态
        self.game_over = False
        self.win = False
        self.first_click = True
        self.mines_placed = False
        self.mines_left = self.num_mines
        self.start_time = time.time()
    
    def place_mines(self, first_x, first_y):
        """放置地雷，确保第一次点击不会触发地雷"""
        # 创建所有可能的位置
        positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        
        # 移除第一次点击的位置及其周围的位置
        safe_positions = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = first_x + dx, first_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    safe_positions.append((nx, ny))
        
        # 从位置列表中移除安全位置
        for pos in safe_positions:
            if pos in positions:
                positions.remove(pos)
        
        # 随机选择地雷位置
        mine_positions = np.random.choice(len(positions), self.num_mines, replace=False)
        
        # 放置地雷
        for i in mine_positions:
            x, y = positions[i]
            self.board[y, x] = -1  # -1表示地雷
    
    def count_adjacent_mines(self, x, y):
        """计算周围的地雷数量"""
        count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.board[ny, nx] == -1:
                    count += 1
        return count
    
    def open_cell(self, x, y):
        """打开一个单元格，如果是空的，则递归打开周围的单元格"""
        if not (0 <= x < self.width and 0 <= y < self.height) or self.visible[y, x] != self.CELL_CLOSED:
            return False
        
        # 如果是第一次点击，放置地雷
        if self.first_click:
            print(f"首次点击: ({x}, {y})，放置地雷中...")
            self.place_mines(x, y)
            self.first_click = False
            
            # 填充周围地雷数量
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y, x] != -1:  # 不是地雷
                        self.board[y, x] = self.count_adjacent_mines(x, y)
        
        # 如果点到地雷，游戏结束
        if self.board[y, x] == -1:
            self.visible[y, x] = self.CELL_OPENED
            self.game_over = True
            print("踩到地雷！游戏结束")
            return True
        
        # 打开单元格
        self.visible[y, x] = self.CELL_OPENED
        
        # 如果是空格（周围没有地雷），递归打开周围的单元格
        if self.board[y, x] == 0:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    self.open_cell(x + dx, y + dy)
        
        # 检查是否获胜
        if self.check_win():
            print("恭喜获胜！")
        
        return True
    
    def toggle_flag(self, x, y):
        """标记或取消标记一个单元格"""
        if not (0 <= x < self.width and 0 <= y < self.height) or self.visible[y, x] == self.CELL_OPENED:
            return False
        
        if self.visible[y, x] == self.CELL_CLOSED:
            self.visible[y, x] = self.CELL_FLAGGED
            self.mines_left -= 1
        elif self.visible[y, x] == self.CELL_FLAGGED:
            self.visible[y, x] = self.CELL_CLOSED
            self.mines_left += 1
        
        return True
    
    def check_win(self):
        """检查是否获胜（所有非地雷的单元格都已打开）"""
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y, x] != -1 and self.visible[y, x] != self.CELL_OPENED:
                    return False
        
        self.win = True
        self.game_over = True
        print("所有安全格子都已打开，游戏胜利！")
        return True
    
    def get_game_state(self):
        """获取当前游戏状态"""
        state = np.zeros((4, self.height, self.width), dtype=np.float32)
        
        # 第一通道：已开启的数字格子（归一化到0-1之间）
        # 第二通道：未打开的格子
        # 第三通道：已标记的格子
        # 第四通道：游戏板边缘信息
        
        for y in range(self.height):
            for x in range(self.width):
                if self.visible[y, x] == self.CELL_OPENED:
                    if self.board[y, x] >= 0:  # 不是地雷
                        state[0, y, x] = self.board[y, x] / 8.0  # 归一化
                elif self.visible[y, x] == self.CELL_CLOSED:
                    state[1, y, x] = 1.0
                elif self.visible[y, x] == self.CELL_FLAGGED:
                    state[2, y, x] = 1.0
                
                # 边缘信息
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
                    state[3, y, x] = 1.0
        
        return state
    
    def get_valid_actions(self):
        """获取有效动作的掩码"""
        action_mask = np.zeros(self.width * self.height, dtype=np.int8)
        
        for y in range(self.height):
            for x in range(self.width):
                pos = y * self.width + x
                if self.visible[y, x] == self.CELL_CLOSED:
                    action_mask[pos] = 1
        
        return action_mask
    
    def ai_move(self):
        """让AI进行一步操作"""
        if self.ai_agent and not self.game_over:
            print("AI正在思考...")
            state = {'board': self.get_game_state()}
            action_mask = self.get_valid_actions()
            
            try:
                action = self.ai_agent.act(state, action_mask)
                x, y = action % self.width, action // self.width
                print(f"AI选择位置: ({x}, {y})")
                return self.open_cell(x, y)
            except Exception as e:
                print(f"AI操作出错: {str(e)}")
                return False
        else:
            if not self.ai_agent:
                print("错误: 未加载AI模型，请加载模型后再尝试AI辅助")
                
                # 没有AI模型时，用随机策略代替
                action_mask = self.get_valid_actions()
                valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                if valid_actions:
                    action = np.random.choice(valid_actions)
                    x, y = action % self.width, action // self.width
                    print(f"使用随机策略选择位置: ({x}, {y})")
                    return self.open_cell(x, y)
            elif self.game_over:
                print("游戏已结束，请按R键重新开始")
            return False
    
    def draw_board(self):
        """绘制游戏板"""
        self.screen.fill(self.COLORS['background'])
        
        # 绘制每个单元格
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)
                
                # 根据单元格状态绘制
                if self.visible[y, x] == self.CELL_CLOSED:
                    pygame.draw.rect(self.screen, self.COLORS['closed'], rect)
                elif self.visible[y, x] == self.CELL_OPENED:
                    pygame.draw.rect(self.screen, self.COLORS['opened'], rect)
                    
                    # 如果是地雷
                    if self.board[y, x] == -1:
                        pygame.draw.circle(self.screen, self.COLORS['mine'],
                                        rect.center, self.cell_size // 3)
                    # 如果是数字
                    elif self.board[y, x] > 0:
                        # 使用专用的数字字体
                        text = self.number_font.render(str(self.board[y, x]), True, 
                                             self.COLORS['text'][self.board[y, x]])
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                        
                elif self.visible[y, x] == self.CELL_FLAGGED:
                    pygame.draw.rect(self.screen, self.COLORS['closed'], rect)
                    pygame.draw.polygon(self.screen, self.COLORS['flag'],
                                     [(x * self.cell_size + self.cell_size // 4, y * self.cell_size + self.cell_size // 4),
                                      (x * self.cell_size + self.cell_size * 3 // 4, y * self.cell_size + self.cell_size // 2),
                                      (x * self.cell_size + self.cell_size // 4, y * self.cell_size + self.cell_size * 3 // 4)])
                
                # 绘制网格
                pygame.draw.rect(self.screen, self.COLORS['grid'], rect, 1)
        
        # 如果游戏结束，显示所有地雷
        if self.game_over:
            # 显示所有地雷
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y, x] == -1 and self.visible[y, x] != self.CELL_OPENED:
                        rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                        self.cell_size, self.cell_size)
                        pygame.draw.rect(self.screen, self.COLORS['opened'], rect)
                        pygame.draw.circle(self.screen, self.COLORS['mine'],
                                        rect.center, self.cell_size // 3)
                        pygame.draw.rect(self.screen, self.COLORS['grid'], rect, 1)
            
            # 显示游戏结果文本 - 使用英文文本避免字体问题
            font = pygame.font.Font(None, 36)
            if self.win:
                text = font.render(f"WIN! Press {self.key_mappings['reset']['key'].upper()} to restart", True, (0, 128, 0))
            else:
                text = font.render(f"GAME OVER! Press {self.key_mappings['reset']['key'].upper()} to restart", True, (255, 0, 0))
            
            # 创建半透明背景
            text_bg = pygame.Surface((self.screen_width, 40), pygame.SRCALPHA)
            text_bg.fill((200, 200, 200, 180))  # RGBA，最后一个值是透明度
            self.screen.blit(text_bg, (0, self.screen_height//2 - 20))
            
            # 显示文本
            text_rect = text.get_rect(center=(self.screen_width//2, self.screen_height//2))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def run_game(self):
        """运行游戏主循环"""
        clock = pygame.time.Clock()
        running = True
        last_key_time = 0
        
        # 配置按键信息
        ai_key = self.key_mappings['ai']['key'].upper()
        reset_key = self.key_mappings['reset']['key'].upper()
        quit_key = self.key_mappings['quit']['key'].upper()
        
        print(f"游戏启动! 按{ai_key}键随机移动，{reset_key}键重置游戏，{quit_key}键退出。")
        print("请点击游戏窗口确保获得焦点")
        
        # 调试信息
        print("按键映射:")
        for action, mapping in self.key_mappings.items():
            print(f"{action}: key={mapping['key']}, ascii={mapping['ascii']}, pygame_key={mapping['pygame_key']}")
        
        while running:
            # 绘制游戏板
            self.draw_board()
            
            # 绘制状态和信息
            status_font = pygame.font.Font(None, 20)
            status_text = f"地雷: {self.mines_left} | 按{ai_key}:随机移动 | 按{reset_key}:重置 | 按{quit_key}:退出"
            
            # 创建状态栏背景
            status_bg = pygame.Surface((self.screen_width, 20), pygame.SRCALPHA)
            status_bg.fill((50, 50, 50, 200))
            self.screen.blit(status_bg, (0, 0))
            
            # 尝试用英文渲染文本避免中文显示问题
            text = status_font.render(status_text, True, (255, 255, 255))
            self.screen.blit(text, (10, 2))
            
            pygame.display.flip()
            
            # 1. 完全重置事件队列以避免积压
            pygame.event.pump()
            
            # 2. 直接检测键盘状态 - 这是最可靠的方法
            keys = pygame.key.get_pressed()
            
            # 3. 添加节流以防止按键重复触发太快
            current_time = time.time()
            if current_time - last_key_time > 0.25:  # 250ms延迟
                # 直接检查具体按键
                if keys[pygame.K_q]:  # 退出键
                    print("检测到Q键，退出游戏")
                    running = False
                    last_key_time = current_time
                
                elif keys[pygame.K_r]:  # 重置键
                    print("检测到R键，重置游戏")
                    self.reset_game()
                    last_key_time = current_time
                
                elif keys[pygame.K_a]:  # AI/随机移动键
                    print("检测到A键，执行随机移动")
                    self.ai_move()  # 使用随机策略
                    last_key_time = current_time
                
                # 检查自定义按键 (如果与默认不同)
                for action, mapping in self.key_mappings.items():
                    pygame_key = mapping['pygame_key']
                    if pygame_key and pygame_key not in [pygame.K_q, pygame.K_r, pygame.K_a] and keys[pygame_key]:
                        print(f"检测到自定义{mapping['key'].upper()}键")
                        if action == 'quit':
                            running = False
                        elif action == 'reset':
                            self.reset_game()
                        elif action == 'ai':
                            self.ai_move()
                        last_key_time = current_time
            
            # 4. 处理标准事件
            for event in pygame.event.get():
                # 调试每个接收到的事件
                if event.type == KEYDOWN:
                    print(f"接收到键盘事件: {pygame.key.name(event.key)} (键值: {event.key})")
                
                if event.type == QUIT:
                    running = False
                
                # 鼠标点击
                elif event.type == MOUSEBUTTONDOWN:
                    x, y = event.pos[0] // self.cell_size, event.pos[1] // self.cell_size
                    
                    if event.button == 1:  # 左键
                        if not self.game_over:
                            self.open_cell(x, y)
                    elif event.button == 3:  # 右键
                        if not self.game_over:
                            self.toggle_flag(x, y)
            
            clock.tick(30)
        
        pygame.quit()
    
    def get_game_result(self):
        """获取游戏结果"""
        if not self.game_over:
            return None
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'win': self.win,
            'time': elapsed_time,
            'mines_left': self.mines_left
        }


def main():
    """主函数"""
    game = MinesweeperGame(width=9, height=9, num_mines=10)
    game.run_game()
    
    result = game.get_game_result()
    if result:
        print(f"游戏结束！{'胜利！' if result['win'] else '失败！'}")
        print(f"用时: {result['time']:.2f}秒")
        
        # 等待用户按键退出
        print("按任意键退出...")
        try:
            import msvcrt
            msvcrt.getch()
        except ImportError:
            pass


if __name__ == "__main__":
    main() 