import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import random

# 1. 贝叶斯实验设计
def bayesian_experimental_design(p_y_given_x_xi, p_y_given_x):
    epsilon = 1e-10  # 小正数避免零值
    kl_divergence = lambda y: (p_y_given_x_xi(y) + epsilon) * np.log((p_y_given_x_xi(y) + epsilon) / (p_y_given_x(y) + epsilon))
    y_values = np.linspace(0, 1, 100)
    information_gain = np.sum([kl_divergence(y) for y in y_values])
    print(f"贝叶斯实验设计 - 信息增益: {information_gain}")
    return information_gain

# 2. 多臂赌博机模型
class MultiArmedBandit:
    def __init__(self, n_arms, epsilon=0.3, decay=0.98):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.epsilon = epsilon
        self.decay = decay  # 衰减因子用于探索率逐步降低

    def select_arm(self):
        if np.random.rand() < self.epsilon:  # ε-贪婪策略
            return np.random.randint(self.n_arms)
        sampled_theta = np.random.beta(self.alpha, self.beta)
        chosen_arm = np.argmax(sampled_theta)
        return chosen_arm

    def update(self, chosen_arm, reward):
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1 - reward
        self.epsilon *= self.decay  # 逐步衰减探索率

    def print_estimates(self):
        print(f"多臂赌博机模型 - α: {self.alpha}, β: {self.beta}")

# 3. 序贯分析
def sequential_analysis(data, f1, f0):
    epsilon = 1e-10
    likelihood_ratio = np.prod([f1(x) / (f0(x) + epsilon) for x in data])
    print(f"序贯分析 - 似然比: {likelihood_ratio}")
    return likelihood_ratio

# 4. 经验贝叶斯方法
def empirical_bayes(data):
    def neg_log_likelihood(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf
        return -np.sum(stats.beta.logpdf(data, alpha, beta))

    result = minimize(neg_log_likelihood, [2, 2], bounds=[(0.1, None), (0.1, None)])  # 调整初始值
    alpha_hat, beta_hat = result.x
    print(f"经验贝叶斯方法 - 估计的 α: {alpha_hat}, β: {beta_hat}")
    return alpha_hat, beta_hat

# 5. 强化学习与上下文赌博机
class ContextualBandit:
    def __init__(self, n_arms, n_features, epsilon=0.3, decay=0.98):
        self.n_arms = n_arms
        self.n_features = n_features
        self.epsilon = epsilon  # ε-贪婪策略中的探索概率
        self.decay = decay
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, x):
        if np.random.rand() < self.epsilon:  # ε-贪婪策略
            return np.random.randint(self.n_arms)
        best_value = -np.inf
        best_arm = 0
        for arm in range(self.n_arms):
            theta_hat = np.linalg.inv(self.A[arm]).dot(self.b[arm])
            upper_confidence_bound = x.T.dot(theta_hat) + 0.1 * np.sqrt(x.T.dot(np.linalg.inv(self.A[arm])).dot(x))
            if upper_confidence_bound > best_value:
                best_value = upper_confidence_bound
                best_arm = arm
        return best_arm

    def update(self, chosen_arm, x, reward):
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x
        self.epsilon *= self.decay  # 逐步衰减探索率

    def print_policies(self):
        print("强化学习与上下文赌博机策略:")
        for arm in range(self.n_arms):
            theta_hat = np.linalg.inv(self.A[arm]).dot(self.b[arm])
            print(f"臂 {arm} 的估计 θ: {theta_hat}")

# 主函数：整合所有步骤
def main():
    # 模拟抽样检测结果和生产过程
    sample_data = np.random.binomial(1, 0.1, 100)  # 假设次品率为10%

    # 贝叶斯实验设计
    p_y_given_x_xi = lambda y: stats.norm.pdf(y, loc=0.1, scale=0.02)
    p_y_given_x = lambda y: stats.norm.pdf(y, loc=0.1, scale=0.05)
    bayesian_experimental_design(p_y_given_x_xi, p_y_given_x)

    # 多臂赌博机模型
    bandit = MultiArmedBandit(n_arms=5, epsilon=0.3, decay=0.98)
    for _ in range(200):  # 增加实验次数
        chosen_arm = bandit.select_arm()
        reward = np.random.binomial(1, 0.3)  # 增加奖励概率
        bandit.update(chosen_arm, reward)
    bandit.print_estimates()

    # 序贯分析
    sequential_analysis(sample_data, f1=lambda x: stats.norm.pdf(x, loc=0.1, scale=0.02),
                        f0=lambda x: stats.norm.pdf(x, loc=0.1, scale=0.05))

    # 经验贝叶斯方法
    empirical_bayes(sample_data)

    # 强化学习与上下文赌博机
    contextual_bandit = ContextualBandit(n_arms=3, n_features=5, epsilon=0.3, decay=0.98)  # 增加探索概率
    for _ in range(200):  # 增加实验次数
        context = np.random.rand(5)  # 增加上下文信息的范围
        chosen_arm = contextual_bandit.select_arm(context)
        reward = np.random.binomial(1, 0.3)  # 增加奖励概率
        contextual_bandit.update(chosen_arm, context, reward)
    contextual_bandit.print_policies()

if __name__ == "__main__":
    main()