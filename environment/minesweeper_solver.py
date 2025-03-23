import numpy as np
import random
from collections import deque

class MinesweeperSolver:
    """
    扫雷游戏求解器，使用各种推理规则识别安全格子和地雷
    用于生成高质量的预训练数据
    """
    def __init__(self):
        self.height = 0
        self.width = 0
        self.board = None
        self.unknown_cells = set()
        self.safe_cells = set()
        self.mine_cells = set()
        self.opened_cells = set()
        self.flagged_cells = set()
        
    def setup(self, board):
        """
        设置游戏板状态
        """
        self.board = np.copy(board)
        self.height, self.width = board.shape
        self.unknown_cells = set()
        self.safe_cells = set()
        self.mine_cells = set()
        self.opened_cells = set()
        self.flagged_cells = set()
        
        # 初始化各种格子集合
        for y in range(self.height):
            for x in range(self.width):
                pos = y * self.width + x
                if board[y, x] == -1:  # 未挖掘
                    self.unknown_cells.add(pos)
                elif board[y, x] == 9:  # 已标记地雷
                    self.flagged_cells.add(pos)
                    self.mine_cells.add(pos)
                elif 0 <= board[y, x] <= 8:  # 已打开的数字或空格
                    self.opened_cells.add(pos)
    
    def get_neighbors(self, pos):
        """
        获取一个位置周围的8个相邻格子
        """
        y, x = divmod(pos, self.width)
        neighbors = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    neighbors.append(ny * self.width + nx)
        
        return neighbors
    
    def get_unknown_neighbors(self, pos):
        """
        获取一个位置周围未知的格子
        """
        return [n for n in self.get_neighbors(pos) if n in self.unknown_cells]
    
    def get_flagged_neighbors(self, pos):
        """
        获取一个位置周围已标记的地雷格子
        """
        return [n for n in self.get_neighbors(pos) if n in self.flagged_cells]
    
    def single_point_strategy(self):
        """
        单点策略：如果一个数字周围的未知格子数等于数字值减去周围已标记的地雷数，
        那么所有未知格子都是地雷；
        如果一个数字周围已标记的地雷数等于数字值，那么所有未知格子都是安全的。
        """
        made_progress = False
        
        for pos in self.opened_cells:
            y, x = divmod(pos, self.width)
            if self.board[y, x] == 0:
                continue  # 忽略空格（数字0）
                
            unknown_neighbors = self.get_unknown_neighbors(pos)
            flagged_neighbors = self.get_flagged_neighbors(pos)
            
            # 如果周围的已标记地雷数等于数字值，那么所有未知格子都是安全的
            if len(flagged_neighbors) == self.board[y, x]:
                for n in unknown_neighbors:
                    if n not in self.safe_cells:
                        self.safe_cells.add(n)
                        made_progress = True
            
            # 如果未知格子数等于数字值减去已标记地雷数，那么所有未知格子都是地雷
            remaining_mines = self.board[y, x] - len(flagged_neighbors)
            if len(unknown_neighbors) == remaining_mines and remaining_mines > 0:
                for n in unknown_neighbors:
                    if n not in self.mine_cells:
                        self.mine_cells.add(n)
                        made_progress = True
        
        return made_progress
    
    def solve_with_csp(self):
        """
        使用约束满足问题(CSP)解决方法
        考虑多个数字格子的共同约束，可以解决更复杂的情况
        """
        made_progress = False
        
        # 构建重叠区域的约束集
        constraints = {}  # {pos: (未知邻居集合, 剩余地雷数)}
        
        for pos in self.opened_cells:
            y, x = divmod(pos, self.width)
            if self.board[y, x] == 0:
                continue
                
            unknown_neighbors = set(self.get_unknown_neighbors(pos))
            if not unknown_neighbors:
                continue
                
            flagged_neighbors = self.get_flagged_neighbors(pos)
            remaining_mines = self.board[y, x] - len(flagged_neighbors)
            
            constraints[pos] = (unknown_neighbors, remaining_mines)
        
        # 查找重叠约束
        for pos1, (cells1, mines1) in list(constraints.items()):
            for pos2, (cells2, mines2) in list(constraints.items()):
                if pos1 == pos2 or not cells1 or not cells2:
                    continue
                
                # 检查第一个集合是否是第二个的子集
                if cells1.issubset(cells2):
                    # cells2 - cells1 肯定有 mines2 - mines1 个地雷
                    diff_cells = cells2 - cells1
                    diff_mines = mines2 - mines1
                    
                    if diff_mines == 0 and diff_cells:
                        # 差集中都是安全的
                        for cell in diff_cells:
                            if cell not in self.safe_cells:
                                self.safe_cells.add(cell)
                                made_progress = True
                    
                    elif diff_mines == len(diff_cells) and diff_cells:
                        # 差集中都是地雷
                        for cell in diff_cells:
                            if cell not in self.mine_cells:
                                self.mine_cells.add(cell)
                                made_progress = True
                
                # 检查第二个集合是否是第一个的子集
                if cells2.issubset(cells1):
                    # cells1 - cells2 肯定有 mines1 - mines2 个地雷
                    diff_cells = cells1 - cells2
                    diff_mines = mines1 - mines2
                    
                    if diff_mines == 0 and diff_cells:
                        # 差集中都是安全的
                        for cell in diff_cells:
                            if cell not in self.safe_cells:
                                self.safe_cells.add(cell)
                                made_progress = True
                    
                    elif diff_mines == len(diff_cells) and diff_cells:
                        # 差集中都是地雷
                        for cell in diff_cells:
                            if cell not in self.mine_cells:
                                self.mine_cells.add(cell)
                                made_progress = True
        
        return made_progress
    
    def probability_analysis(self):
        """
        概率分析：计算每个未知格子是地雷的概率，选择概率最低的格子作为安全格子
        这是一个启发式方法，不保证100%准确，但在没有确定安全格子时很有用
        """
        # 如果已经有确定的安全格子，直接返回
        if self.safe_cells:
            return False
            
        # 每个未知格子的地雷概率
        mine_probabilities = {pos: 0.0 for pos in self.unknown_cells}
        cell_constraints = {pos: [] for pos in self.unknown_cells}
        
        # 收集每个未知格子相关的约束
        for pos in self.opened_cells:
            y, x = divmod(pos, self.width)
            if self.board[y, x] == 0:
                continue
                
            unknown_neighbors = self.get_unknown_neighbors(pos)
            flagged_neighbors = self.get_flagged_neighbors(pos)
            remaining_mines = self.board[y, x] - len(flagged_neighbors)
            
            if not unknown_neighbors:
                continue
                
            # 这个约束影响的所有未知格子
            for n in unknown_neighbors:
                cell_constraints[n].append((set(unknown_neighbors), remaining_mines))
        
        # 计算每个未知格子的地雷概率
        for pos in self.unknown_cells:
            constraints = cell_constraints[pos]
            if not constraints:
                # 如果没有相关约束，使用默认概率
                mine_probabilities[pos] = 0.5
                continue
                
            # 使用最严格的约束（概率最低或最高的情况）
            min_prob = 1.0
            for cells, mines in constraints:
                if mines == 0:
                    # 确定是安全的
                    min_prob = 0.0
                    break
                elif len(cells) == mines:
                    # 确定是地雷
                    min_prob = 1.0
                    break
                else:
                    # 计算概率
                    prob = mines / len(cells)
                    min_prob = min(min_prob, prob)
            
            mine_probabilities[pos] = min_prob
        
        # 如果有确定的安全格子（概率为0），添加到安全格子集合
        safe_candidates = [pos for pos, prob in mine_probabilities.items() if prob == 0.0]
        if safe_candidates:
            for pos in safe_candidates:
                self.safe_cells.add(pos)
            return True
            
        # 如果有确定的地雷（概率为1），添加到地雷集合
        mine_candidates = [pos for pos, prob in mine_probabilities.items() if prob == 1.0]
        if mine_candidates:
            for pos in mine_candidates:
                self.mine_cells.add(pos)
            return True
        
        # 如果没有确定的安全格子或地雷，选择概率最低的作为"可能安全"
        if mine_probabilities:
            safest_pos = min(mine_probabilities, key=mine_probabilities.get)
            if mine_probabilities[safest_pos] < 0.3:  # 阈值可以调整
                self.safe_cells.add(safest_pos)
                return True
        
        return False
    
    def get_safe_move(self, action_mask=None):
        """
        获取一个安全的移动，如果有多个，随机选择一个
        """
        if not self.safe_cells:
            return None
            
        valid_safe_cells = list(self.safe_cells)
        if action_mask is not None:
            valid_safe_cells = [pos for pos in valid_safe_cells if action_mask[pos] == 1]
            
        if not valid_safe_cells:
            return None
            
        return random.choice(valid_safe_cells)
    
    def solve(self, board, action_mask=None):
        """
        解析游戏板，返回一个安全的动作
        如果找不到确定安全的动作，使用概率分析选择最可能安全的格子
        
        参数:
            board: 游戏板状态
            action_mask: 有效动作掩码
            
        返回:
            安全的动作索引，如果没有找到则返回None
        """
        self.setup(board)
        
        # 如果是第一步，选择中心位置
        if np.all(board == -1):
            center_y, center_x = self.height // 2, self.width // 2
            return center_y * self.width + center_x
        
        # 迭代应用求解策略，直到没有进展
        made_progress = True
        while made_progress:
            made_progress = False
            made_progress |= self.single_point_strategy()
            made_progress |= self.solve_with_csp()
            
        # 如果基本策略无法找到安全格子，尝试概率分析
        if not self.safe_cells:
            self.probability_analysis()
        
        # 获取安全的移动
        return self.get_safe_move(action_mask)
    
    def solve_board(self, board, action_mask):
        """
        完整求解游戏板，返回所有能够确定的安全格子和地雷
        
        返回:
            (safe_moves, mine_positions)
        """
        self.setup(board)
        
        # 迭代应用求解策略，直到没有进展
        made_progress = True
        iterations = 0
        max_iterations = 10  # 防止无限循环
        
        while made_progress and iterations < max_iterations:
            iterations += 1
            made_progress = False
            made_progress |= self.single_point_strategy()
            made_progress |= self.solve_with_csp()
        
        # 应用概率分析
        self.probability_analysis()
        
        # 过滤有效的安全格子
        valid_safe_moves = []
        if action_mask is not None:
            valid_safe_moves = [pos for pos in self.safe_cells if action_mask[pos] == 1]
        else:
            valid_safe_moves = list(self.safe_cells)
            
        return valid_safe_moves, list(self.mine_cells)
    
    def generate_moves_sequence(self, board, action_mask, max_moves=20):
        """
        生成一个移动序列，用于训练
        
        参数:
            board: 初始游戏板状态
            action_mask: 有效动作掩码
            max_moves: 最大移动次数
            
        返回:
            移动序列列表
        """
        from copy import deepcopy
        
        # 模拟环境的简化版本
        class SimpleMinesweeper:
            def __init__(self, board, mines, action_mask):
                self.board = deepcopy(board)
                self.mines = deepcopy(mines)
                self.action_mask = deepcopy(action_mask)
                self.height, self.width = board.shape
                
            def step(self, action):
                y, x = divmod(action, self.width)
                
                # 检查是否有效
                if self.action_mask[action] == 0:
                    return deepcopy(self.board), 0, True, False
                
                # 更新action_mask
                self.action_mask[action] = 0
                
                # 如果是地雷，游戏结束
                if (y, x) in self.mines:
                    self.board[y, x] = 9  # 标记为地雷
                    return deepcopy(self.board), -1, True, True
                
                # 计算周围地雷数
                mine_count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            if (ny, nx) in self.mines:
                                mine_count += 1
                
                # 更新格子
                self.board[y, x] = mine_count
                
                # 如果是0，自动打开周围的格子
                if mine_count == 0:
                    queue = deque([(y, x)])
                    while queue:
                        cy, cx = queue.popleft()
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < self.height and 0 <= nx < self.width:
                                    pos = ny * self.width + nx
                                    if self.board[ny, nx] == -1 and self.action_mask[pos] == 1:
                                        self.action_mask[pos] = 0
                                        
                                        # 计算周围地雷
                                        nmine_count = 0
                                        for ndy in [-1, 0, 1]:
                                            for ndx in [-1, 0, 1]:
                                                if ndy == 0 and ndx == 0:
                                                    continue
                                                nny, nnx = ny + ndy, nx + ndx
                                                if 0 <= nny < self.height and 0 <= nnx < self.width:
                                                    if (nny, nnx) in self.mines:
                                                        nmine_count += 1
                                        
                                        self.board[ny, nx] = nmine_count
                                        if nmine_count == 0:
                                            queue.append((ny, nx))
                
                # 检查是否获胜
                if np.sum(self.action_mask) == len(self.mines):
                    return deepcopy(self.board), 1, True, False
                
                return deepcopy(self.board), 0.5, False, False
        
        # 创建简化的扫雷环境
        mines = []
        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
                if board[y, x] == 9:
                    mines.append((y, x))
        
        env = SimpleMinesweeper(board, mines, action_mask)
        moves_sequence = []
        
        for _ in range(max_moves):
            # 求解当前状态
            action = self.solve(env.board, env.action_mask)
            if action is None:
                break
                
            # 执行动作
            new_board, reward, done, is_mine = env.step(action)
            
            # 记录成功的动作
            if not is_mine:
                moves_sequence.append((deepcopy(env.board), action))
            
            if done:
                break
        
        return moves_sequence 