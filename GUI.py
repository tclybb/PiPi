import pygame
from datetime import datetime
import numpy as np
from AI import MCTS, AlphaZeroModel, PiPiGameWrapper
import tensorflow as tf
from PiPi import ChessGame

class ChessGameGUI:
    def __init__(self, game):
        # 初始化Pygame
        pygame.init()
        
        # 设置窗口大小
        self.screen = pygame.display.set_mode((500, 300))
        pygame.display.set_caption("PiPi Chess")

        
        # 游戏实例
        self.game = game
        self.start_time = datetime.now()
        
        # 每个格子的宽度和高度
        self.grid_size = 50
        self.width = self.height = 6 * self.grid_size
        
        # 颜色定义
        self.WHITE = (255, 255, 255)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW_B = (120, 120, 0)
        self.BLUE_B = (0, 0, 80)
        self.BACKBOARD = (159, 245, 205)

        self.keyboard_dir = {pygame.K_UP: '↑', pygame.K_DOWN: '↓', pygame.K_LEFT: '←', pygame.K_RIGHT: '→'}
        self.numboard_dir = {pygame.K_KP1: '↙', pygame.K_KP2: '↓', pygame.K_KP3: '↘', pygame.K_KP4: '←', 
                             pygame.K_KP6: '→', pygame.K_KP7: '↖', pygame.K_KP8: '↑', pygame.K_KP9: '↗',
                             pygame.K_UP: '↑', pygame.K_DOWN: '↓', pygame.K_LEFT: '←', pygame.K_RIGHT: '→'}

        self.font = pygame.font.Font(None, 32)
        self.shoot_flag = False # 弹射标记
        self.start_piece = None

        # self.ai_player = AIPlayer('blue')  # 创建AI玩家
        self.ai_mode = False  # AI模式开关
        self.ai_vs_ai_mode = False  # AI模式开关

        # self.ai_player = RLAgent('blue')
        # self.ai_player.load_model("blue_ai_final.pkl")
        # self.ai_mode = True  # 启用AI模式

        self.ai_agent = None  # AI实例
        self.ai_agents = {}  # AI实例
        self.training_mode = False  # 训练模式标志
        self.running = True  # 新增游戏运行状态标记
        self.ai_move_delay = 500  # AI动作间隔(毫秒)

    def draw_board(self): # ////////需调整中心框格
        # 绘制棋盘背景           
        for i in range(6):
            for j in range(6):
                rect = pygame.Rect(j*self.grid_size, i*self.grid_size,
                                  self.grid_size, self.grid_size)
                if (i, j) in self.game.center: color = (255, 224, 232)
                elif (i + j) % 2 == 0: color = self.WHITE
                else: color = (150, 150, 150)
                pygame.draw.rect(self.screen, color, rect)

        # 绘制信息区
        pygame.draw.rect(self.screen, self.BACKBOARD, (300, 0, 500, 300)) 
        text_current_player = self.font.render(f"Current Player:", True, (0, 0, 0))
        self.screen.blit(text_current_player, (320, 20))
        if self.game.current_player == 'yellow':
            pygame.draw.circle(self.screen, self.YELLOW,
                              (400, 100), self.grid_size//2.5)
        else: pygame.draw.circle(self.screen, self.BLUE,
                              (400, 100), self.grid_size//2.5)

        elapsed = datetime.now() - self.start_time
        time_str = f"Time: {elapsed.seconds//60}:{elapsed.seconds%60:02d}"
        text = self.font.render(time_str, True, (0, 0, 0))
        self.screen.blit(text, (320, 200))
        if self.shoot_flag:
            text = self.font.render(f"Shooting Time", True, (0, 0, 0))
            self.screen.blit(text, (320, 250))
                
        # 绘制棋子
        for pos in self.game.pieces['yellow']:
            x, y = pos
            center_x = x * self.grid_size + self.grid_size // 2
            center_y = y * self.grid_size + self.grid_size // 2
            pygame.draw.circle(self.screen, self.YELLOW,
                              (center_x, center_y), self.grid_size//2.5)
            
        for pos in self.game.pieces['blue']:
            x, y = pos
            center_x = x * self.grid_size + self.grid_size // 2
            center_y = y * self.grid_size + self.grid_size // 2
            pygame.draw.circle(self.screen, self.BLUE,
                              (center_x, center_y), self.grid_size//2.5)
        
        for pos in self.game.blocked_pieces['yellow']:
            x, y = pos
            center_x = x * self.grid_size + self.grid_size // 2
            center_y = y * self.grid_size + self.grid_size // 2
            pygame.draw.circle(self.screen, self.YELLOW_B,
                              (center_x, center_y), self.grid_size//2.5)
            
        for pos in self.game.blocked_pieces['blue']:
            x, y = pos
            center_x = x * self.grid_size + self.grid_size // 2
            center_y = y * self.grid_size + self.grid_size // 2
            pygame.draw.circle(self.screen, self.BLUE_B,
                              (center_x, center_y), self.grid_size//2.5)
            
        # 高亮选中格
        if (not self.shoot_flag) and self.game.selected_piece:
            x, y = self.game.selected_piece
            pygame.draw.rect(self.screen, (0, 255, 0), 
                            (x * self.grid_size, y * self.grid_size,
                            self.grid_size, self.grid_size), 2)
            
        # 高亮弹射棋子
        if self.shoot_flag and self.start_piece in self.game.pieces[self.game.current_player]:
            x, y = self.start_piece
            pygame.draw.rect(self.screen, (0, 255, 255), 
                            (x * self.grid_size, y * self.grid_size,
                            self.grid_size, self.grid_size), 2)
            
        (winner, is_game_over, _) = self.game.is_victory()
        if is_game_over:
            font = pygame.font.Font(None, 48)
            text = font.render(f"Winner: {winner}", True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(text, text_rect)
    
    def handle_input(self):
        if self.ai_agent and self.game.current_player == self.ai_agent.color:
            self.handle_ai_turn()
            return True
        if self.ai_mode and self.game.current_player == self.ai_player.color:
            self.handle_ai_turn()
            return True
        if self.ai_vs_ai_mode:
            self.handle_ai_turn()
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 获取鼠标位置
                mouse_pos = pygame.mouse.get_pos()
                x = mouse_pos[0] // self.grid_size
                y = mouse_pos[1] // self.grid_size
                
                # 检查该位置是否有当前玩家的棋子
                if (x, y) in self.game.pieces[self.game.current_player]:
                    self.game.selected_piece = (x, y)
            
            elif event.type == pygame.KEYDOWN:
                if self.game.selected_piece not in self.game.pieces[self.game.current_player]: break #执行操作后跳出
                if self.game.selected_piece is not None:
                    if (not self.shoot_flag) and event.key in self.keyboard_dir.keys(): #移动
                        if self.game.move_piece(self.game.selected_piece, self.keyboard_dir[event.key]): #判断是否合法移动
                            self.game.change_player()
                            self.selected_piece = None
                    elif event.key == pygame.K_SPACE:  # 切换弹射模式
                        self.shoot_flag = not self.shoot_flag
                        self.start_piece = self.game.selected_piece
                    elif self.shoot_flag and event.key in self.numboard_dir.keys():
                        if self.game.shoot_piece(self.start_piece, self.numboard_dir[event.key]): #判断是否合法移动
                            self.game.change_player()
                            self.selected_piece = None
                            self.shoot_flag = False
    
        return True
        
    def set_ai_mode(self, mode, model_paths=None):
        """完整AI模式设置"""
        from AI import MCTS, PiPiGameWrapper, AlphaZeroModel
        import tensorflow as tf
        
        self.ai_agents = {}
        if mode == 'ai_vs_ai':
            for color in ['yellow', 'blue']:
                # 加载对应颜色模型
                model = AlphaZeroModel()
                model.model = tf.keras.models.load_model(model_paths[color])
                self.ai_agents[color] = MCTS(model)
                
        elif mode == 'ai_vs_human':
            color = list(model_paths.keys())[0]  # 获取配置的AI颜色
            model = AlphaZeroModel()
            model.model = tf.keras.models.load_model(model_paths[color])
            self.ai_agents[color] = MCTS(model)

    def ai_turn_loop(self):
        """AI对战主循环"""
        while self.running:
            # 检查游戏状态
            winner, is_over, _ = self.game.is_victory()
            if is_over:
                print(f"游戏结束! 胜利方: {winner}")
                self.running = False
                break
                
            # 获取当前AI
            current_color = self.game.current_player
            mcts = self.ai_agents.get(current_color)
            if not mcts:
                pygame.time.wait(100)  # 人类玩家回合不处理
                continue

            # AI决策流程
            wrapper = PiPiGameWrapper(self.game.copy())
            mcts.search(wrapper)  # MCTS搜索
            action_probs = mcts.get_action_probs(wrapper, temp=0)
            action_idx = np.argmax(action_probs)
            
            # 执行动作并同步状态
            
            wrapper.do_action(action_idx)
            mcts.update_with_move(action_idx)
            self.game = wrapper.game

            # 更新界面
            self.draw_board()
            pygame.display.flip()
            pygame.time.wait(self.ai_move_delay)  # 控制AI速度

    def run(self):
        """启动游戏主循环"""
        clock = pygame.time.Clock()
        while self.running:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
            # AI对战模式自动运行
            if hasattr(self, 'ai_agents') and len(self.ai_agents) > 0:
                self.ai_turn_loop()
            else:  # 人类玩家模式
                self.handle_input()
                self.draw_board()
                pygame.display.flip()
                
            clock.tick(60)  # 限制60FPS
        pygame.quit()
        
    def handle_ai_turn(self):
        """处理AI回合"""
        current_color = self.game.current_player
        mcts = self.ai_agents.get(current_color)
        
        if mcts:
            # 获取当前游戏状态
            wrapper = PiPiGameWrapper(self.game.copy())
            
            # MCTS搜索
            mcts.search(wrapper)
            action_probs = mcts.get_action_probs(wrapper, temp=0)
            action_idx = np.argmax(action_probs)
            
            # 执行动作
            wrapper.do_action(action_idx)
            
            # 同步游戏状态
            self.game = wrapper.game
            mcts.update_with_move(action_idx)
        
    def update(self):
        self.draw_board()
        pygame.display.flip()
        winner, is_over, _ = self.game.is_victory()
        while is_over: pygame.time.wait(5000000) 


# terminal_chess.py
from PiPi import ChessGame
from AI import MCTS, AlphaZeroModel, PiPiGameWrapper
import numpy as np

class TerminalChess:
    def __init__(self, ai_models=None):
        self.game = ChessGame()
        self.ai_agents = {}
        if ai_models:
            for color, model_path in ai_models.items():
                model = AlphaZeroModel()
                model.model = tf.keras.models.load_model(model_path)
                self.ai_agents[color] = MCTS(model)
                
    def print_board(self):
        """ASCII方式打印棋盘"""
        print("\n" + "="*40)
        print(f"当前玩家: {self.game.current_player}")
        print(f"当前回合: {self.game.turn}/{self.game.max_turns}")
        print("禁走区标记: Y-黄色禁走, B-蓝色禁走")
        print("棋子标记: ◎-黄色棋子, ●-蓝色棋子\n")
        
        for y in range(6):
            row = []
            for x in range(6):
                pos = (x, y)
                if pos in self.game.pieces['yellow']:
                    sym = '◎  ' 
                elif pos in self.game.pieces['blue']:
                    sym = '●  '
                else:
                    sym = '.  '
                
                # 添加禁走区标记
                if pos in self.game.blocked_pieces['yellow']:
                    sym = 'Y  '
                elif pos in self.game.blocked_pieces['blue']:
                    sym = 'B  '

                row.append(sym)
            print("".join(row))
        print()
        
    def human_turn(self):
        """处理人类玩家输入"""
        print("可用操作：")
        print("1. 移动棋子")
        print("2. 射击棋子")
        choice = input("请选择操作类型(1/2): ").strip()
        
        # 获取棋子位置
        while True:
            try:
                x = int(input("输入棋子X坐标(0-5): "))
                y = int(input("输入棋子Y坐标(0-5): "))
                if (x, y) not in self.game.pieces[self.game.current_player]:
                    raise ValueError
                break
            except:
                print("无效位置，请重新输入")

        # 移动操作
        if choice == '1':
            dir_map = {'w':'↑','s':'↓','a':'←','d':'→'}
            while True:
                direction = input("输入方向(w=↑,s=↓,a=←,d=→): ").lower()
                if direction in dir_map and self.game.move_piece((x,y), dir_map[direction]):
                    self.game.change_player()
                    return
                print("无效方向，请重新输入")
                
        # 射击操作
        elif choice == '2':
            dir_map = {
                '7':'↖', '8':'↑', '9':'↗',
                '4':'←',          '6':'→',
                '1':'↙', '2':'↓', '3':'↘'
            }
            while True:
                print("小键盘方向对应：")
                print("7 8 9\n4   6\n1 2 3")
                direction = input("输入数字方向: ")
                if direction in dir_map and self.game.shoot_piece((x,y), dir_map[direction]):
                    self.game.change_player()
                    return
                print("无效方向，请重新输入")

    def ai_turn(self):
        """AI自动决策"""
        current_color = self.game.current_player
        opponent_color = 'yellow' if current_color == 'blue' else 'blue'
        mcts = self.ai_agents.get(current_color)
        mcts_op = self.ai_agents.get(opponent_color)
        wrapper = PiPiGameWrapper(self.game.copy())
        mcts.search(wrapper)  # MCTS搜索
        action_probs = mcts.get_action_probs(wrapper, temp=0)
        action_idx = np.argmax(action_probs)
        wrapper.do_action(action_idx)
        self.game = wrapper.game
        mcts.update_with_move(action_idx)
        # mcts_op.update_with_move(action_idx)
        print(f"AI选择了动作 {action_idx}")

    def run(self):
        while True:
            self.print_board()
            winner, ended, _ = self.game.is_victory()
            if ended:
                print(f"游戏结束！胜利方：{winner}")
                break
            if self.game.current_player in self.ai_agents:
                print("AI正在思考...")
                self.ai_turn()
            else:
                self.human_turn()

if __name__ == "__main__":
    # 配置示例：双AI对战
    game = TerminalChess(ai_models={
        'yellow': 'models/ai_8.h5',
        'blue': 'models/ai_8.h5'
    })
    # game = TerminalChess(ai_models={
    #     'yellow': 'models/ai_8.h5'
    # })
    game.run()
