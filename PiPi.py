from copy import deepcopy

class ChessGame:
    def __init__(self, is_copy=False):

        self.pieces = {
            'yellow': set([(0, 1), (1, 0), (5, 4), (4, 5)]),
            'blue': set([(0, 4), (1, 5), (5, 1), (4, 0)])
        }

        # 禁走区标记
        self.blocked_pieces = {
            'yellow': set(),
            'blue': set()
        }

        # self.pieces['yellow'].add((0, 1))
        # self.pieces['yellow'].add((1, 0))
        # self.pieces['yellow'].add((5, 4))
        # self.pieces['yellow'].add((4, 5))
        # self.pieces['blue'].add((0, 4))
        # self.pieces['blue'].add((1, 5))
        # self.pieces['blue'].add((5, 1))
        # self.pieces['blue'].add((4, 0))
        self.center = {(2,2), (2,3), (3,3), (3,2)}
        
        # 当前玩家，黄方先手
        self.current_player = 'yellow'
        self.selected_piece = None  # 记录选中的棋子位置
        self.winner = None
        self.max_turns = 150  # 新增最大回合数
        self.turn = 0

        self.is_copy = is_copy
        
    def _sign(self, x):
        if x < 0: return -1
        else: return 1
    
    def change_player(self):
        self.turn += 1
        self.current_player = 'blue' if self.current_player == 'yellow' else 'yellow'
        
    def _is_homo(self, start_pos, target_pos): #判断是否同色
        if start_pos in self.pieces['yellow'] and target_pos in self.pieces['yellow']: return True
        if start_pos in self.pieces['blue'] and target_pos in self.pieces['blue']: return True
        return False

    def move_piece(self, piece_pos, direction):
        # 检查目标位置是否在棋盘内
        x, y = piece_pos
        if direction == '↑':
            new_x, new_y = x, y-1
        elif direction == '↓':
            new_x, new_y = x, y+1
        elif direction == '←':
            new_x, new_y = x-1, y
        elif direction == '→':
            new_x, new_y = x+1, y
        else:
            return False  # 无效方向
        
        if 0 <= new_x < 6 and 0 <= new_y < 6:
            # 检查目标位置是否为空或有敌方棋子
            if (new_x, new_y) not in (self.pieces['yellow'] | self.pieces['blue']) and not self.is_blocked_area((new_x, new_y), self.current_player):
                # 更新棋子位置
                if piece_pos not in self.pieces[self.current_player]:
                    return False
                if self.current_player == 'yellow':
                    self.blocked_pieces['yellow'].clear()
                    self.blocked_pieces['yellow'].add((x, y))
                    self.pieces['yellow'].remove((x, y))
                    self.pieces['yellow'].add((new_x, new_y))
                else:
                    self.blocked_pieces['blue'].clear()
                    self.blocked_pieces['blue'].add((x, y))
                    self.pieces['blue'].remove((x, y))
                    self.pieces['blue'].add((new_x, new_y))
                return True
            else:
                return False
        return False
    
    def is_valid_shoot(self, start_pos, direction):
        shoot_player = self.current_player
        operation_valid = False
        x_flag, y_flag = start_pos
        while 0 <= x_flag <= 5 and 0 <= y_flag <= 5:
            if direction in ('↙', '←', '↖'): x_flag -= 1
            elif direction in ('↘', '→', '↗'): x_flag += 1
            if direction in ('↖', '↑', '↗'): y_flag -= 1
            elif direction in ('↙', '↓', '↘'): y_flag += 1
            if shoot_player == self.current_player and (x_flag, y_flag) in self.blocked_pieces[self.current_player]: return False
            if (x_flag, y_flag) in self.pieces[self.current_player]: 
                operation_valid = True
                shoot_player = self.current_player
            elif (x_flag, y_flag) in self.pieces[set(self.pieces.keys() - {self.current_player}).pop()]:
                shoot_player = set(self.pieces.keys() - {self.current_player}).pop()
        return operation_valid
        
    def shoot_decide(self, start_pos, direction):
        self.blocked_pieces[self.current_player].clear() 
        x, y = start_pos
        x_flag, y_flag = start_pos
        if direction in ('↙', '←', '↖'): x_flag -= 1
        elif direction in ('↘', '→', '↗'): x_flag += 1
        if direction in ('↖', '↑', '↗'): y_flag -= 1
        elif direction in ('↙', '↓', '↘'): y_flag += 1
        while True:
            if not 0 <= x_flag <= 5  or not 0 <= y_flag <= 5:
                if (x, y) not in self.pieces[self.current_player]:
                    self.pieces[set(self.pieces.keys() - {self.current_player}).pop()].remove((x, y)) #将另一名玩家的棋子踢出后break
                break
            if (x_flag, y_flag) not in (self.pieces['yellow'] | self.pieces['blue']):
                if (x, y) in self.pieces['yellow']: 
                    if self.current_player == 'yellow': self.blocked_pieces['yellow'].add((x, y))
                    self.pieces['yellow'].remove((x, y))
                    self.pieces['yellow'].add((x_flag, y_flag))
                else:
                    if self.current_player == 'blue': self.blocked_pieces['blue'].add((x, y))
                    self.pieces['blue'].remove((x, y))
                    self.pieces['blue'].add((x_flag, y_flag))
            x, y = x_flag, y_flag
            if direction in ('↙', '←', '↖'): x_flag -= 1
            elif direction in ('↘', '→', '↗'): x_flag += 1
            if direction in ('↖', '↑', '↗'): y_flag -= 1
            elif direction in ('↙', '↓', '↘'): y_flag += 1

    def shoot_piece(self, start_pos, target_direction): #点击方向键射
        if start_pos not in self.pieces[self.current_player]: return False
        if self.is_valid_shoot(start_pos, target_direction): 
            self.shoot_decide(start_pos, target_direction)
            return True
        return False
        
    def is_move_valid(self, piece_pos, direction):
        """验证移动是否合法（不修改状态）"""
        x, y = piece_pos
        # 计算新位置
        if direction == '↑':
            new_x, new_y = x, y-1
        elif direction == '↓':
            new_x, new_y = x, y+1
        elif direction == '←':
            new_x, new_y = x-1, y
        elif direction == '→':
            new_x, new_y = x+1, y
        else:
            return False  # 无效方向
        
        # 检查边界
        if not (0 <= new_x < 6 and 0 <= new_y < 6):
            return False
        
        # 目标位置是否为空且未被阻塞
        target_pos = (new_x, new_y)
        all_pieces = self.pieces['yellow'] | self.pieces['blue']
        return (
            target_pos not in all_pieces and
            not self.is_blocked_area(target_pos, self.current_player)
        )

    def is_shoot_valid(self, start_pos, direction):
        """验证射击是否合法（不修改状态）"""
        x, y = start_pos
        if start_pos not in self.pieces[self.current_player]:
            return False
        
        current_player = self.current_player
        opponent = 'blue' if current_player == 'yellow' else 'yellow'
        x_flag, y_flag = x, y
        has_valid_target = False
        
        while True:
            # 根据方向移动
            if direction in ('↙', '←', '↖'):
                x_flag -= 1
            elif direction in ('↘', '→', '↗'):
                x_flag += 1
            if direction in ('↖', '↑', '↗'):
                y_flag -= 1
            elif direction in ('↙', '↓', '↘'):
                y_flag += 1
            
            # 越界检查
            if not (0 <= x_flag < 6 and 0 <= y_flag < 6): break

            # 禁走区
            if (x_flag, y_flag) in self.blocked_pieces[current_player]: 
                has_valid_target = False
                break

            # 检查是否命中棋子
            if (x_flag, y_flag) in self.pieces[current_player]: has_valid_target = True
        return has_valid_target

    def mark_blocked_areas(self, piece_pos, current_player):
        self.blocked_pieces[current_player].add(piece_pos)
    
    def is_blocked_area(self, piece_pos, current_player):
        if piece_pos in self.blocked_pieces[current_player]:
            return True
        return False
        
    def get_valid_actions(self):
        valid_actions = []
        for piece in self.pieces[self.current_player]:
            # 移动动作优化
            x, y = piece
            possible_dirs = []
            if y > 0: possible_dirs.append('↑')
            if y < 5: possible_dirs.append('↓')
            if x > 0: possible_dirs.append('←')
            if x < 5: possible_dirs.append('→')
            for dir in possible_dirs:
                if self.is_move_valid(piece, dir):
                    valid_actions.append(('move', piece, dir))
            # 射击动作优化
            shoot_dirs = ['↖', '↑', '↗', '←', '→', '↙', '↓', '↘']
            for dir in shoot_dirs:
                if self.is_shoot_valid(piece, dir):
                    valid_actions.append(('shoot', piece, dir))
        return valid_actions
    
    def is_victory(self):
        if self.pieces['yellow'].issubset(self.center): return 'yellow', True, 'Occupy'
        if self.pieces['blue'].issubset(self.center): return 'blue', True, 'Occupy'
        if len(self.pieces['yellow']) <= 1: return 'blue', True, 'Destroy'
        if len(self.pieces['blue']) <= 1: return 'yellow', True, 'Destroy'
        if self.turn >= self.max_turns: return 'draw', True, None
        if self.get_valid_actions() == None: 
            return 'blue' if self.current_player == 'yellow' else 'yellow'
        return None, False, None

    def copy(self):
        new_game = ChessGame(is_copy=True)
        new_game.pieces = {
            'yellow': set(self.pieces['yellow']),
            'blue': set(self.pieces['blue'])
        }
        new_game.blocked_pieces = {
            'yellow': set(self.blocked_pieces['yellow']),
            'blue': set(self.blocked_pieces['blue'])
        }
        new_game.current_player = self.current_player  # 确保复制当前玩家
        new_game.selected_piece = self.selected_piece
        new_game.winner = self.winner
        new_game.turn = self.turn
        return new_game
        
    def __deepcopy__(self, memo):
        new_game = ChessGame(is_copy=True)
        new_game.pieces = deepcopy(self.pieces)
        new_game.blocked_pieces = deepcopy(self.blocked_pieces)
        new_game.current_player = self.current_player
        new_game.turn = self.turn
        return new_game