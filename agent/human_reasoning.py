import numpy as np
import random
from collections import deque, defaultdict

class HumanReasoning:
    """
    模拟人类扫雷推理的类
    整合逻辑推理与概率分析能力
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.board_size = height * width
        
        # 状态追踪
        self.unknown_cells = set()
        self.safe_cells = set()
        self.mine_cells = set()
        self.flagged_cells = set()
        self.opened_cells = set()
        
    def update_state(self, board_state, action_mask):
        """
        更新当前棋盘状态
        
        参数:
            board_state: 当前棋盘状态 (多通道数组)
            action_mask: 有效动作掩码
        """
        self.unknown_cells = set()
        self.safe_cells = set()
        self.mine_cells = set()
        self.flagged_cells = set()
        self.opened_cells = set()
        
        # 扫描棋盘更新各类格子集合
        normalized_board = board_state[0]  # 使用第一个通道作为主要状态
        
        for y in range(self.height):
            for x in range(self.width):
                pos = y * self.width + x
                
                # 检查是否是合法位置
                if not (0 <= y < self.height and 0 <= x < self.width):
                    continue
                    
                if normalized_board[y, x] == -0.5:  # 未挖掘
                    if action_mask[pos] == 1:  # 仍然有效
                        self.unknown_cells.add(pos)
                elif normalized_board[y, x] == 1.5:  # 标记为地雷
                    self.flagged_cells.add(pos)
                    self.mine_cells.add(pos)
                elif 0 <= normalized_board[y, x] <= 1.0:  # 已打开的数字或空格 (归一化后在0-1之间)
                    self.opened_cells.add(pos)
    
    def get_neighbors(self, pos):
        """获取位置周围的8个相邻格子"""
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
    
    def get_unknown_neighbors(self, pos, board_state):
        """获取位置周围未知的格子"""
        y, x = divmod(pos, self.width)
        unknown = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    npos = ny * self.width + nx
                    if board_state[0][ny, nx] == -0.5 and npos in self.unknown_cells:
                        unknown.append(npos)
        
        return unknown
    
    def get_flagged_neighbors(self, pos, board_state):
        """获取位置周围已标记的地雷格子"""
        y, x = divmod(pos, self.width)
        flagged = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    npos = ny * self.width + nx
                    if board_state[0][ny, nx] == 1.5:  # 标记为地雷
                        flagged.append(npos)
        
        return flagged
        
    def identify_safe_cells(self, board_state, action_mask):
        """
        使用单格策略寻找确定安全的格子
        
        返回:
            safe_actions: 确定安全的动作列表
        """
        self.update_state(board_state, action_mask)
        
        safe_actions = []
        
        # 扫描所有已打开的数字格子
        for pos in self.opened_cells:
            y, x = divmod(pos, self.width)
            if board_state[0][y, x] == 0:  # 跳过数字为0的格子
                continue
                
            # 获取周围未知格子和已标记地雷
            unknown_neighbors = self.get_unknown_neighbors(pos, board_state)
            flagged_neighbors = self.get_flagged_neighbors(pos, board_state)
            
            # 将数字转换回1-8范围
            cell_number = round(board_state[0][y, x] * 8)
            
            # 规则1: 如果已标记地雷数等于格子数字，则周围所有未知格子安全
            if len(flagged_neighbors) == cell_number and unknown_neighbors:
                for neighbor in unknown_neighbors:
                    if neighbor not in safe_actions and action_mask[neighbor] == 1:
                        safe_actions.append(neighbor)
                        self.safe_cells.add(neighbor)
        
        # 如果没有找到确定安全的格子，返回空列表
        return safe_actions
    
    def identify_mines(self, board_state, action_mask):
        """
        识别确定是地雷的格子
        
        返回:
            mine_actions: 确定是地雷的动作列表
        """
        self.update_state(board_state, action_mask)
        
        mine_actions = []
        
        # 扫描所有已打开的数字格子
        for pos in self.opened_cells:
            y, x = divmod(pos, self.width)
            if board_state[0][y, x] == 0:  # 跳过数字为0的格子
                continue
                
            # 获取周围未知格子和已标记地雷
            unknown_neighbors = self.get_unknown_neighbors(pos, board_state)
            flagged_neighbors = self.get_flagged_neighbors(pos, board_state)
            
            # 将数字转换回1-8范围
            cell_number = round(board_state[0][y, x] * 8)
            
            # 规则2: 如果未知格子数量+已标记地雷数=格子数字，则所有未知格子都是地雷
            remaining_mines = cell_number - len(flagged_neighbors)
            if len(unknown_neighbors) == remaining_mines and remaining_mines > 0:
                for neighbor in unknown_neighbors:
                    if neighbor not in mine_actions:
                        mine_actions.append(neighbor)
                        self.mine_cells.add(neighbor)
        
        return mine_actions
    
    def solve_with_csp(self, board_state, action_mask):
        """
        使用约束满足问题(CSP)解决方法
        考虑多个数字格子的共同约束
        
        返回:
            (safe_actions, mine_actions): 安全动作和地雷动作的元组
        """
        self.update_state(board_state, action_mask)
        
        safe_actions = []
        mine_actions = []
        
        # 构建约束集
        constraints = {}  # {pos: (未知邻居集合, 剩余地雷数)}
        
        for pos in self.opened_cells:
            y, x = divmod(pos, self.width)
            if board_state[0][y, x] == 0:
                continue
                
            unknown_neighbors = set(self.get_unknown_neighbors(pos, board_state))
            if not unknown_neighbors:
                continue
                
            flagged_neighbors = self.get_flagged_neighbors(pos, board_state)
            
            # 将数字转换回1-8范围
            cell_number = round(board_state[0][y, x] * 8)
            remaining_mines = cell_number - len(flagged_neighbors)
            
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
                            if cell not in safe_actions and action_mask[cell] == 1:
                                safe_actions.append(cell)
                                self.safe_cells.add(cell)
                    
                    elif diff_mines == len(diff_cells) and diff_cells:
                        # 差集中都是地雷
                        for cell in diff_cells:
                            if cell not in mine_actions:
                                mine_actions.append(cell)
                                self.mine_cells.add(cell)
                
                # 检查第二个集合是否是第一个的子集
                if cells2.issubset(cells1):
                    # cells1 - cells2 肯定有 mines1 - mines2 个地雷
                    diff_cells = cells1 - cells2
                    diff_mines = mines1 - mines2
                    
                    if diff_mines == 0 and diff_cells:
                        # 差集中都是安全的
                        for cell in diff_cells:
                            if cell not in safe_actions and action_mask[cell] == 1:
                                safe_actions.append(cell)
                                self.safe_cells.add(cell)
                    
                    elif diff_mines == len(diff_cells) and diff_cells:
                        # 差集中都是地雷
                        for cell in diff_cells:
                            if cell not in mine_actions:
                                mine_actions.append(cell)
                                self.mine_cells.add(cell)
        
        return safe_actions, mine_actions
    
    def calculate_probabilities(self, board_state, action_mask):
        """
        计算每个未知格子是地雷的概率
        
        返回:
            低风险动作列表
        """
        self.update_state(board_state, action_mask)
        
        # 如果已经有确定安全的格子，就不需要计算概率了
        if self.safe_cells:
            return list(self.safe_cells)
            
        # 每个未知格子的地雷概率
        mine_probabilities = {pos: 0.0 for pos in self.unknown_cells}
        cell_constraints = defaultdict(list)
        
        # 收集每个未知格子相关的约束
        for pos in self.opened_cells:
            y, x = divmod(pos, self.width)
            if board_state[0][y, x] == 0:
                continue
                
            unknown_neighbors = self.get_unknown_neighbors(pos, board_state)
            flagged_neighbors = self.get_flagged_neighbors(pos, board_state)
            
            # 将数字转换回1-8范围
            cell_number = round(board_state[0][y, x] * 8)
            remaining_mines = cell_number - len(flagged_neighbors)
            
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
        
        # 找出概率最低的格子
        if mine_probabilities:
            min_prob = min(mine_probabilities.values())
            lowest_risk_cells = [pos for pos, prob in mine_probabilities.items() 
                                if prob == min_prob and action_mask[pos] == 1]
            
            if lowest_risk_cells and min_prob < 0.5:  # 只返回概率较低的格子
                return lowest_risk_cells
        
        # 如果找不到低风险格子，尝试选择边缘格子
        edge_cells = self.find_edge_cells(board_state, action_mask)
        if edge_cells:
            return edge_cells
            
        # 实在没有策略时，随机选择
        valid_actions = [pos for pos in self.unknown_cells if action_mask[pos] == 1]
        return valid_actions if valid_actions else []
    
    def find_edge_cells(self, board_state, action_mask):
        """寻找边缘格子（数字旁边的未知格子）"""
        edge_cells = []
        
        for pos in self.unknown_cells:
            if action_mask[pos] != 1:
                continue
                
            y, x = divmod(pos, self.width)
            
            # 检查是否在数字旁边
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        # 检查是否是数字格子(归一化后在0-1之间的数值)
                        if 0 < board_state[0][ny, nx] < 1.0:
                            edge_cells.append(pos)
                            break
                else:
                    continue
                break
        
        return edge_cells
    
    def find_safe_move(self, board_state, action_mask):
        """
        使用人类推理找出安全的移动
        
        返回:
            action: 推荐的安全动作索引，如果找不到确定安全的则返回概率最低的
        """
        # 如果是第一步，选择中心位置
        if np.all(board_state[0] == -0.5):
            center_y, center_x = self.height // 2, self.width // 2
            return center_y * self.width + center_x
        
        # 1. 使用单格策略寻找确定安全的格子
        safe_actions = self.identify_safe_cells(board_state, action_mask)
        if safe_actions:
            return random.choice(safe_actions)
        
        # 2. 使用约束满足问题解决更复杂的情况
        csp_safe, _ = self.solve_with_csp(board_state, action_mask)
        if csp_safe:
            return random.choice(csp_safe)
        
        # 3. 使用概率分析
        low_risk_actions = self.calculate_probabilities(board_state, action_mask)
        if low_risk_actions:
            return random.choice(low_risk_actions)
        
        # 4. 如果还找不到，随机选择一个有效动作
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) > 0:
            return random.choice(valid_actions)
        
        # 如果没有有效动作，返回None
        return None 