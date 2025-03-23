import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class MinesweeperEnv(gym.Env):
    """
    扫雷游戏环境，符合OpenAI Gym规范
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, width=5, height=5, num_mines=3, render_mode=None):
        """
        初始化扫雷环境
        
        参数:
            width (int): 游戏板宽度
            height (int): 游戏板高度
            num_mines (int): 地雷数量
            render_mode (str): 渲染模式，可选 "human" 或 "rgb_array"
        """
        self.width = width
        self.height = height
        self.num_mines = min(num_mines, width * height - 1)  # 确保地雷数量不超过总格子数-1
        self.board_size = width * height
        self.render_mode = render_mode
        
        # 定义动作空间（选择一个格子）
        self.action_space = spaces.Discrete(self.board_size)
        
        # 定义观察空间
        # -1: 未挖掘, 0-8: 周围地雷数, 9: 已知地雷
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=9, shape=(height, width), dtype=np.int8),
            "mines": spaces.Box(low=0, high=1, shape=(height, width), dtype=np.int8),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.board_size,), dtype=np.int8)
        })
        
        # 游戏状态
        self.board = None  # 玩家看到的游戏板
        self.mines = None  # 地雷位置
        self.counts = None  # 每个格子周围的地雷数
        self.action_mask = None  # 可用动作掩码
        self.done = False
        self.steps = 0
        
        # 用于计算奖励
        self.prev_revealed_count = 0
        self.safe_cells = None  # 安全的格子（不含地雷）
        
        # 新增：用于跟踪已经揭示的安全区域
        self.known_safe_area = None
        self.first_action = True  # 标记是否是首次行动

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 重置游戏状态
        self.board = np.full((self.height, self.width), -1, dtype=np.int8)  # 全部未挖掘
        self.action_mask = np.ones(self.board_size, dtype=np.int8)  # 所有动作可用
        self.done = False
        self.steps = 0
        self.prev_revealed_count = 0
        self.first_action = True
        
        # 随机放置地雷
        self.mines = np.zeros((self.height, self.width), dtype=np.int8)
        mine_positions = self.np_random.choice(self.board_size, self.num_mines, replace=False)
        for pos in mine_positions:
            y, x = divmod(pos, self.width)
            self.mines[y, x] = 1
            
        # 计算每个格子周围的地雷数
        self.counts = np.zeros((self.height, self.width), dtype=np.int8)
        for y in range(self.height):
            for x in range(self.width):
                if self.mines[y, x] == 1:
                    continue  # 地雷格子不需要计算
                
                # 计算周围8个格子的地雷数
                mine_count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width and self.mines[ny, nx] == 1:
                            mine_count += 1
                self.counts[y, x] = mine_count
        
        # 计算安全格子数量
        self.safe_cells = self.board_size - self.num_mines
        
        # 初始化已知安全区域
        self.known_safe_area = np.zeros((self.height, self.width), dtype=np.int8)
                
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """执行一个动作并返回结果"""
        # 检查动作是否有效
        if action < 0 or action >= self.board_size:
            # 无效动作
            reward = -5.0
            terminated = False
            truncated = False
            info = {"invalid_action": True}
            observation = self._get_observation()
            return observation, reward, terminated, truncated, info
        
        y, x = divmod(action, self.width)
        terminated = False
        truncated = False
        info = {}
        
        # 第一次点击保证不会踩到地雷
        if self.first_action:
            # 如果第一次点击的是地雷，重新放置地雷
            if self.mines[y, x] == 1:
                # 把这个位置的地雷移到别的地方
                self.mines[y, x] = 0
                
                # 随机找一个没有地雷的位置放置新地雷
                while True:
                    new_y, new_x = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
                    if self.mines[new_y, new_x] == 0 and (new_y != y or new_x != x):
                        self.mines[new_y, new_x] = 1
                        break
            
            # 计算每个格子周围的地雷数
            for cy in range(self.height):
                for cx in range(self.width):
                    if self.mines[cy, cx] == 0:
                        mine_count = 0
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < self.height and 0 <= nx < self.width and self.mines[ny, nx] == 1:
                                    mine_count += 1
                        self.counts[cy, cx] = mine_count
            
        self.first_action = False
        
        # 检查是否是有效动作（未挖掘的格子）
        if self.board[y, x] != -1 or self.action_mask[action] == 0:
            # 无效动作，给予惩罚
            reward = -2.0  # 增加对无效动作的惩罚
            info["invalid_action"] = True
        elif self.mines[y, x] == 1:
            # 点到地雷，游戏失败
            self.board[y, x] = 9  # 标记为地雷
            self.action_mask[action] = 0  # 更新动作掩码
            
            # 新的惩罚机制：根据游戏进度增加惩罚
            progress = self.prev_revealed_count / self.safe_cells
            reward = -10.0 - 20.0 * progress  # 游戏越接近完成踩雷惩罚越大
            
            terminated = True
            info["game_over"] = True
        else:
            # 保存当前状态用于计算信息增益
            old_board = self.board.copy()
            old_known_safe = self.known_safe_area.copy()
            old_revealed_count = self.prev_revealed_count
            
            # 有效的安全格子
            self._reveal_cell(y, x)
            
            # 更新已知安全区域
            for i in range(self.height):
                for j in range(self.width):
                    if self.board[i, j] >= 0 and self.board[i, j] < 9:
                        self.known_safe_area[i, j] = 1
            
            # 计算新挖掘的格子数量
            revealed_count = np.sum(self.board >= 0)
            newly_revealed = revealed_count - self.prev_revealed_count
            self.prev_revealed_count = revealed_count
            
            # 1. 基本奖励：每挖掘一个安全格子给予递增奖励
            progress_factor = revealed_count / self.safe_cells  # 游戏进度因子
            base_reward = 0.5 * newly_revealed * (1 + progress_factor)  # 随游戏进度增加奖励
            
            # 2. 信息增益奖励：揭示更多信息的动作获得更高奖励
            # 计算信息增益：揭示的新安全区域大小
            info_gain = np.sum(self.known_safe_area - old_known_safe)
            info_reward = 0.8 * info_gain  # 增加信息增益奖励
            
            # 3. 空白区域奖励：点击到0（没有周围地雷）的格子给予额外奖励
            if self.counts[y, x] == 0:
                blank_reward = 2.0  # 增加空白格子的奖励
            else:
                blank_reward = 0.0
            
            # 4. 明智决策奖励：选择周围已知数字附近的格子
            wise_choice_reward = 0.0
            has_adjacent_clue = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if old_board[ny, nx] > 0 and old_board[ny, nx] < 9:
                            has_adjacent_clue = True
                            break
                if has_adjacent_clue:
                    break
            
            if has_adjacent_clue:
                wise_choice_reward = 1.0  # 增加基于策略的奖励
            
            # 5. 高风险高回报：如果周围有大量未揭示格子，给予额外奖励
            unknown_neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if old_board[ny, nx] == -1:  # 未揭示格子
                            unknown_neighbors += 1
            
            risk_reward = 0.1 * unknown_neighbors  # 周围的未知格子越多，奖励越高
            
            # 6. 进度奖励：每完成10%的游戏进度给予额外奖励
            old_progress_tier = int(old_revealed_count / self.safe_cells * 10)
            new_progress_tier = int(revealed_count / self.safe_cells * 10)
            progress_reward = 0.0
            if new_progress_tier > old_progress_tier:
                progress_reward = 3.0  # 每突破一个10%进度点给予额外奖励
            
            # 7. 胜利奖励
            victory_reward = 0.0
            if revealed_count == self.safe_cells:
                victory_reward = 50.0  # 大幅增加胜利奖励
                terminated = True
                info["win"] = True
            
            # 总奖励
            reward = base_reward + info_reward + blank_reward + wise_choice_reward + risk_reward + progress_reward + victory_reward
            
            # 奖励指标记录
            info["base_reward"] = base_reward
            info["info_reward"] = info_reward
            info["blank_reward"] = blank_reward
            info["wise_choice_reward"] = wise_choice_reward
            info["risk_reward"] = risk_reward
            info["progress_reward"] = progress_reward
            info["victory_reward"] = victory_reward
            info["newly_revealed"] = newly_revealed
        
        self.steps += 1
        self.done = terminated
        
        observation = self._get_observation()
        
        # 更新可用动作掩码
        for i in range(self.board_size):
            y, x = divmod(i, self.width)
            if self.board[y, x] != -1:  # 已挖掘格子不能再点击
                self.action_mask[i] = 0
        
        return observation, reward, terminated, truncated, info
    
    def _reveal_cell(self, y, x):
        """
        挖掘特定格子，如果是0则自动挖掘周围格子
        """
        if not (0 <= y < self.height and 0 <= x < self.width):
            return
        
        # 如果格子已经挖掘或标记为地雷，则跳过
        if self.board[y, x] != -1:
            return
        
        # 如果是地雷，不要挖掘（应该在step函数中处理）
        if self.mines[y, x] == 1:
            return
        
        # 挖掘当前格子
        self.board[y, x] = self.counts[y, x]
        action_idx = y * self.width + x
        self.action_mask[action_idx] = 0
        
        # 如果是0（周围没有地雷），则自动挖掘周围格子
        if self.counts[y, x] == 0:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    self._reveal_cell(y + dy, x + dx)
    
    def _get_observation(self):
        """
        获取当前状态的观察
        """
        return {
            "board": self.board.copy(),
            "mines": self.mines.copy() if self.done else np.zeros_like(self.mines),  # 只在游戏结束时显示地雷
            "action_mask": self.action_mask.copy()
        }
    
    def render(self, mode="human", disable_text_output=False):
        """
        渲染当前游戏状态
        
        参数:
            mode: 渲染模式，"human" 或 "rgb_array"
            disable_text_output: 是否禁用文本输出
        """
        if not disable_text_output:
            # 文本模式渲染（控制台输出）
            output = ""
            for y in range(self.height):
                output += "|"
                for x in range(self.width):
                    cell_value = self.board[y, x]
                    if cell_value == -1:
                        output += "?|"
                    elif cell_value == 9:
                        output += "F|"
                    elif cell_value == 0:
                        output += " |"
                    else:
                        output += str(cell_value) + "|"
                if y < self.height - 1:
                    output += "\n" + "-" * (self.width * 2 + 1) + "\n"
            
            print(output)
        
        # 如果需要，还可以添加GUI渲染代码
        if mode == "human" and self.render_mode == "human":
            # GUI渲染代码将在这里实现
            pass
        elif mode == "rgb_array" and self.render_mode == "rgb_array":
            # 返回RGB数组表示
            # 将在这里实现
            pass
    
    def close(self):
        """关闭环境，释放资源"""
        pass
    
    def seed(self, seed=None):
        """设置随机种子"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def set_difficulty(self, num_mines):
        """
        设置游戏难度（地雷数量）
        
        参数:
            num_mines (int): 地雷数量
        """
        self.num_mines = min(num_mines, self.width * self.height - 1)
        print(f"难度已设置为 {self.num_mines} 个地雷")
        
    def get_difficulty(self):
        """
        获取当前难度（地雷数量）
        
        返回:
            int: 当前地雷数量
        """
        return self.num_mines
        
    def get_valid_actions(self):
        """
        获取当前有效的动作（未挖掘的格子）
        
        返回:
            np.array: 动作掩码，1表示有效，0表示无效
        """
        obs = self._get_observation()
        return obs["action_mask"] 