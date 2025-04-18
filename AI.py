import numpy as np
import tensorflow as tf
from PiPi import ChessGame
from collections import deque
import random
import matplotlib.pyplot as plt
from joblib import Parallel
import math
from threading import Thread


def build_action_space():
    """构建动作空间：[(位置, 方向类型)]"""
    actions = []
    shoot_dirs = ['↖','↑','↗','←','→','↙','↓','↘']
    move_dirs = ['↑','↓','←','→']
    for x in range(6):
        for y in range(6):
            for d in move_dirs:
                actions.append(((x, y), 'move', d))
            for d in shoot_dirs:
                actions.append(((x, y), 'shoot', d))
    return actions
init_actions = build_action_space()

# 配置参数
CONFIG = {
    "n_playout": 200,      # 增加模拟次数
    "c_puct": 1.5,        # 降低探索系数
    "batch_size": 256,
    "buffer_size": 1000000, # 增大经验池
    "epochs": 50,          # 增加训练轮次
    "train_steps": 5000,   # 延长总训练步数
    "learning_rate": 0.003,
    "lr_decay": 0.95,      # 新增学习率衰减
    "temp_threshold": 150,    # 延长探索阶段
    "max_turn": 300,
    "actions": init_actions,
    "y_sample": 2,
    "b_sample": 2           # 前期策略学习             
}


class PiPiGameWrapper:
    """游戏状态包装器"""
    def __init__(self, game=None):
        self.game = game if game else ChessGame()
        self.game.max_turns = CONFIG['max_turn']
        self.actions = CONFIG['actions']
        
    def get_feature(self):
        """6x6x6特征平面：
        0: 当前玩家棋子 1: 对手棋子 
        2: 中心区域 3: 当前禁走区 
        4: 对手禁走区 5: 回合信息"""
        feat = np.zeros((6,6,6), dtype=np.float32)
        cp = self.game.current_player
        op = 'blue' if cp == 'yellow' else 'yellow'
        
        # 棋子位置
        for x, y in self.game.pieces[cp]:
            feat[x,y,0] = 1
        for x, y in self.game.pieces[op]:
            feat[x,y,1] = 1
            
        # 中心区域
        for x, y in self.game.center:
            feat[x,y,2] = 1
            
        # 禁走区
        for x, y in self.game.blocked_pieces[cp]:
            feat[x,y,3] = 1
        for x, y in self.game.blocked_pieces[op]:
            feat[x,y,4] = 1
            
        # 回合信息（归一化到0-1）
        feat[:,:,5] = self.game.turn / (CONFIG['max_turn'] + 1e-5)

        for x, y in self.game.pieces[op]:
            if x in (0,5) or y in (0,5):
                feat[x,y,5] = 1  # 使用原第五通道存储该特征

        return feat
    
    def _get_valid_actions(self):
        """获取合法动作掩码"""
        valid = np.zeros(len(self.actions), dtype=np.float32)
        valid_actions = self.game.get_valid_actions()
        for i, (pos, act, direction) in enumerate(self.actions):
            if (act, pos, direction) in valid_actions: valid[i] = 1
        return valid
    
    def do_action(self, action_idx):
        """执行动作"""
        (pos, action_type, dir) = self.actions[action_idx]
        success = False
        if action_type == 'move':
            success = self.game.move_piece(pos, dir)
        else:
            success = self.game.shoot_piece(pos, dir)
        self.game.change_player()
        return success
        
    def get_result(self):
        """获取游戏结果"""
        winner, ended, _ = self.game.is_victory()
        if not ended:
            # 中间奖励
            current_player = self.game.current_player
            opponent_player = 'yellow' if current_player == 'blue' else 'blue'
            center_count = len(self.game.pieces[current_player] & self.game.center)
            own_count = len(self.game.pieces[current_player])
            enemy_count = len(self.game.pieces[opponent_player])
            return 0.1 * (center_count + own_count - enemy_count)  # 鼓励占中心，逼对手到边缘
        return 1 if winner == self.game.current_player else -1
    
    def copy(self):
        return PiPiGameWrapper(self.game.copy())

    def print_board(self):
        """ASCII方式打印棋盘"""
        print("\n" + "="*40)
        op = 'yellow' if self.game.current_player == 'blue' else 'blue'
        print(f"当前玩家: {op}")
        print(f"当前回合: {self.game.turn}/{self.game.max_turns}")
        
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


class AlphaZeroModel:
    """神经网络模型"""
    def __init__(self, model_path=None):
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._build_model()
        
    def _build_model(self):
        inputs = tf.keras.Input(shape=(6,6,6))
        
        # 增强特征提取层
        x = tf.keras.layers.Conv2D(256, 3, padding='same')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        # 增加残差层到12层
        for _ in range(12):
            shortcut = x
            x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.add([x, shortcut])
            x = tf.keras.layers.ReLU()(x)
        
        # 增强策略头
        policy = tf.keras.layers.Conv2D(4, 1)(x)
        policy = tf.keras.layers.Flatten()(policy)
        policy = tf.keras.layers.Dense(512, activation='relu')(policy)
        policy = tf.keras.layers.Dense(432, activation='softmax', name='policy')(policy)
        
        # 增强价值头
        value = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
        value = tf.keras.layers.Flatten()(value)
        value = tf.keras.layers.Dense(256, activation='relu')(value)
        value = tf.keras.layers.Dense(128, activation='relu')(value)
        value = tf.keras.layers.Dense(1, activation='tanh', name='value')(value)
        
        return tf.keras.Model(inputs, [policy, value])
    
    def train(self, data, step):
        """训练模型"""
        lr = CONFIG['learning_rate'] * (CONFIG['lr_decay'] ** (step//100))
        states, policies, values = zip(*data)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=['categorical_crossentropy', 'mse'])
        self.model.fit(
            np.array(states), [np.array(policies), np.array(values)],
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size']
        )
        history = self.model.fit(
            np.array(states), [np.array(policies), np.array(values)],
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            verbose=0
        )
        
        return history
    
    def predict(self, state):
        """预测策略和价值"""
        policy, value = self.model.predict(state[np.newaxis], verbose=0)
        return policy[0], value[0][0]


class TreeNode:
    """MCTS树节点"""
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}  # {action: TreeNode}
        self.N = 0          # 访问次数
        self.Q = 0          # 平均动作价值
        self.P = prior_prob # 先验概率
        self.virtual_loss = 0  # 虚拟损失


class MCTS:
    """蒙特卡洛树搜索完整实现"""
    def __init__(self, model):
        self.model = model
        self.root = TreeNode()
        self.c_puct = CONFIG['c_puct']
        self.min_valid_probs = 1e-8  # 新增最小概率阈值
        self.history = deque(maxlen=10)
    
    def _select(self, game_state):
        """选择阶段：从根节点到叶节点"""
        node = self.root
        path = []  # 仅记录节点路径
        while node.children:
            # 修改为仅存储节点
            action, node = max(node.children.items(),
                             key=lambda item: self._ucb(item[1]))
            success = game_state.do_action(action)
            path.append(node)  # 关键修改：仅存储节点对象
        return node, game_state, path
    
    def _ucb(self, node):
        """计算UCB值"""
        return node.Q + self.c_puct * node.P * np.sqrt(node.parent.N) / (1 + node.N)
    
    def _expand(self, node, game_state):
        """优化后的扩展方法，仅处理合法动作"""
        state_tensor = game_state.get_feature()
        policy, value = self.model.predict(state_tensor)
        valid_actions = game_state._get_valid_actions()
        
        # 直接获取合法动作索引
        valid_indices = np.where(valid_actions > 0.5)[0]
        if len(valid_indices) == 0:
            return 0  # 无合法动作时直接返回
        
        # 过滤并归一化合法动作概率
        valid_probs = policy[valid_indices]
        sum_probs = np.sum(valid_probs)
        if sum_probs < self.min_valid_probs:
            valid_probs = np.ones_like(valid_probs) / len(valid_indices)
        else:
            valid_probs /= sum_probs
        
        # 仅创建合法动作的子节点
        for idx, action in enumerate(valid_indices):
            node.children[action] = TreeNode(
                parent=node, 
                prior_prob=valid_probs[idx]
            )
        return value
    
    def _backpropagate(self, path, value):
        """增强的回传方法"""
        current_value = value
        for node in reversed(path):
            node.N += 1
            node.Q += (current_value - node.Q) / node.N
            current_value = -current_value  # 切换视角
    
    def search(self, game_state, n_playout=CONFIG['n_playout']):
        """执行多次模拟搜索"""
        self._prune_tree(max_depth=20) 
        for playout in range(n_playout):
            state_copy = game_state.copy()
            node, leaf_state, path = self._select(state_copy)
            
            (_, is_v, _) = leaf_state.game.is_victory()
            if not is_v:
                # 扩展并获取网络评估值
                value = self._expand(node, leaf_state)
            else:
                # 终局直接获取结果
                value = leaf_state.get_result()
            self._backpropagate(path, value)
    
    def _prune_tree(self, max_depth=5):
        """修剪树深度，释放内存"""
        def prune(node, depth):
            if depth >= max_depth:
                node.children.clear()
            else:
                for child in node.children.values():
                    prune(child, depth+1)
        prune(self.root, 0)
    
    def get_action_probs(self, game_state, temp=1):
        """获取动作概率分布"""
        self.search(game_state)
        visits = np.zeros(len(game_state.actions), dtype=np.int32)
        
        for action, node in self.root.children.items():
            visits[action] = node.N
        
        if temp == 0:  # 贪婪选择
            probs = np.zeros_like(visits, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
        else:  # 带温度参数的softmax
            visits = visits ** (1.0 / temp)
            probs = visits / np.sum(visits)
        
        return probs
    
    def update_with_move(self, action):
        """推进树根节点"""
        self.root = self.root.children[action]
        self.root.parent = None


class Trainer:
    """训练流程控制器"""
    def __init__(self, init_model_path=None):
        self.buffer = deque(maxlen=CONFIG['buffer_size'])
        self.loss_history = []  # 记录损失值
        self.policy_entropy_history = []  # 记录策略熵
        self.value_loss_history = []  # 记录价值损失
        if init_model_path:
            self.model = AlphaZeroModel(model_path=init_model_path)
        else:
            self.model = AlphaZeroModel()
        
    def self_play(self):
        
        game = PiPiGameWrapper()
        mcts = MCTS(self.model)
        game.game.pieces['yellow'] = set(random.sample(list(game.game.pieces['yellow']), CONFIG['y_sample']))
        game.game.pieces['blue'] = set(random.sample(list(game.game.pieces['blue']), CONFIG['b_sample']))
        
        while True:
            # MCTS获取动作概率
            temp = 1 if game.game.turn < CONFIG['temp_threshold'] else 0
            action_probs = mcts.get_action_probs(game, temp)
            action = np.random.choice(len(action_probs), p=action_probs)
            state = game.get_feature()
            game.do_action(action)
            game.print_board()
            value = game.get_result() 
            self.buffer.append((state, action_probs, value))
            print(f'本轮行动: {game.actions[action]}')
            print(f'value: {value}')
            mcts.update_with_move(action)
    
            winner, is_v, way = game.game.is_victory()

            if is_v: break
        
    def train_loop(self):
        """训练主循环"""
        for step in range(CONFIG['train_steps']):
            CONFIG['n_playout'] = max(10, CONFIG['n_playout'] - step)
            CONFIG['max_turn'] = max(100, CONFIG['max_turn'] - step*10)
            CONFIG['temp_threshold'] = max(10, CONFIG['temp_threshold'] - step*2)
            sample = [(2,2),(2,3),(3,2),(2,4),(4,2),(3,3),(4,3),(3,4),(4,4)]
            (CONFIG['y_sample'], CONFIG['b_sample']) = sample[min(math.floor(step/10), len(sample)-1)]
            # threads = []
            # for _ in range(10):
            #     t = Thread(target=self.self_play)
            #     threads.append(t)
            #     t.start()
            # for t in threads: t.join()
            self.self_play()
            if len(self.buffer) >= CONFIG['batch_size']:
                batch = random.sample(self.buffer, CONFIG['batch_size'])
                history = self.model.train(batch, step)
                self.loss_history.extend(history.history['loss'])
                self.policy_entropy_history.extend(history.history['policy_loss'])
                self.value_loss_history.extend(history.history['value_loss'])
                CONFIG['batch_size'] = max(2048, 2*CONFIG['batch_size'])
            
            print('训练完成, step=', end='')
            print(step)
            self.model.model.save(f'models\\ai_{step}.h5')
            self.plot_training_curves(step)

    def plot_training_curves(self, step):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 6))

        # 绘制总损失曲线
        plt.subplot(3, 1, 1)
        plt.plot(self.loss_history, label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制策略熵曲线
        plt.subplot(3, 1, 2)
        plt.plot(self.policy_entropy_history, label='Policy Entropy', color='orange')
        plt.title('Policy Entropy')
        plt.xlabel('Step')
        plt.ylabel('Entropy')
        plt.legend()

        # 绘制价值损失曲线
        plt.subplot(3, 1, 3)
        plt.plot(self.value_loss_history, label='Value Loss', color='green')
        plt.title('Value Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'training_plots\\training_curves_step_{step}.png')
        plt.close()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_loop()