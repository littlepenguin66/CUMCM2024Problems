import numpy as np
import tensorflow as tf
from collections import deque
import random
from deap import base, creator, tools, algorithms

# --------------- 参数设置 --------------- #

# 设置基本参数
p1_cost, p2_cost, assembly_cost = 4, 18, 6
p1_defect_rate, p2_defect_rate, product_defect_rate = 0.1, 0.1, 0.1
market_price, replace_loss, disassemble_cost = 56, 6, 5
p1_test_cost, p2_test_cost, product_test_cost = 2, 3, 3

# --------------- 多目标优化与模糊综合评判 --------------- #

def total_cost(x):
    return x[0] * p1_cost + x[1] * p2_cost + x[2] * assembly_cost

def quality_loss(x):
    return (1 - x[0]) * p1_defect_rate + (1 - x[1]) * p2_defect_rate + (1 - x[2]) * product_defect_rate

def production_efficiency(x):
    return 1 / ((1 - p1_defect_rate) * (1 - p2_defect_rate) * (1 - product_defect_rate))

def fuzzy_evaluation():
    A = np.array([0.3, 0.4, 0.3])  # 权重向量
    R = np.array([[0.7, 0.5, 0.8],  # 成本维度
                  [0.6, 0.9, 0.7],  # 质量维度
                  [0.8, 0.4, 0.6]]) # 效率维度
    B = np.dot(A, R)
    return B

# --------------- 定义决策策略函数 --------------- #

def decision_strategy(detect_p1, detect_p2, detect_product, disassemble):
    """
    根据输入策略计算总成本和利润。
    """
    total_cost = 0
    
    # 零件检测成本
    if detect_p1:
        total_cost += p1_test_cost
    if detect_p2:
        total_cost += p2_test_cost
    
    # 成品装配与检测成本
    assembly_success_rate = (1 - p1_defect_rate) * (1 - p2_defect_rate)
    
    if detect_product:
        total_cost += product_test_cost * assembly_success_rate
    else:
        total_cost += assembly_cost

    # 拆解成本
    if disassemble:
        total_cost += disassemble_cost * (1 - assembly_success_rate)

    # 调换损失
    replacement_loss = replace_loss * product_defect_rate
    total_cost += replacement_loss

    # 收益
    total_revenue = market_price * assembly_success_rate

    # 最终利润
    profit = total_revenue - total_cost
    return profit

# --------------- 遗传算法 --------------- #

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def eval_production(individual):
    # individual: [是否检测零件1, 是否检测零件2, 是否检测成品, 是否拆解]
    return decision_strategy(detect_p1=individual[0], detect_p2=individual[1], detect_product=individual[2], disassemble=individual[3]),

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_production)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)  # 增加变异概率
toolbox.register("select", tools.selTournament, tournsize=5)  # 增加锦标赛的规模

def run_genetic_algorithm():
    population = toolbox.population(n=500)  # 增加种群规模
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=50, verbose=True)  # 增加代数

# --------------- 深度强化学习 (DQN) 使用 TensorFlow --------------- #

class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(action_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)

# DQN 参数
state_size = 4  # 状态维度
action_size = 2  # 动作数量
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64  # 增加批量大小
memory = deque(maxlen=2000)

# Q 网络和目标网络
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 初始化网络权重
dummy_input = np.random.rand(1, state_size).astype(np.float32)
q_network(dummy_input)  # 初始化 q_network 权重
target_network(dummy_input)  # 初始化 target_network 权重

def store_memory(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = q_network(np.array([state], dtype=np.float32))
    return np.argmax(q_values[0])

def train():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    
    states = np.array([x[0] for x in minibatch], dtype=np.float32)
    actions = [x[1] for x in minibatch]
    rewards = [x[2] for x in minibatch]
    next_states = np.array([x[3] for x in minibatch], dtype=np.float32)
    dones = [x[4] for x in minibatch]
    
    q_values_next = target_network(next_states)
    targets = rewards + gamma * np.amax(q_values_next, axis=1) * np.logical_not(dones)
    
    with tf.GradientTape() as tape:
        q_values = q_network(states)
        q_values = tf.gather_nd(q_values, np.array(list(enumerate(actions))))
        loss = tf.keras.losses.MSE(targets, q_values)
    
    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

def update_target_network():
    target_network.set_weights(q_network.get_weights())

def dqn_train(episodes):
    global epsilon
    for e in range(episodes):
        state = np.random.rand(state_size)  # 初始化状态
        done = False
        total_reward = 0
        
        while not done:
            action = choose_action(state, epsilon)
            next_state = np.random.rand(state_size)  # 随机生成下一状态
            reward = np.random.rand()  # 随机生成奖励
            done = np.random.choice([True, False])  # 随机结束条件
            
            store_memory(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode {e + 1}: Total Reward: {total_reward}")
        
        train()
        update_target_network()
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

# --------------- 综合调用 --------------- #

if __name__ == "__main__":
    print("开始运行遗传算法优化...")
    run_genetic_algorithm()
    
    print("\n模糊综合评判结果:")
    fuzzy_result = fuzzy_evaluation()
    print(fuzzy_result)
    
    print("\n开始训练深度强化学习模型 (DQN)...")
    dqn_train(episodes=1000)