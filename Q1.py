import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

class AdvancedAdaptiveSamplingPlan:
    def __init__(self, p0, alpha, beta, max_samples=100, cs=1):
        self.p0 = p0  # 标称次品率
        self.alpha = alpha  # I 型错误率
        self.beta = beta  # II 型错误率
        self.max_samples = max_samples  # 最大样本数量
        self.cs = cs  # 单位抽样成本
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)  # 随机森林模型

    def compute_sample_size(self, prev_n, prev_p):
        # 样本量动态调整，设置下限为最小样本数量 10，确保不会太小
        return max(int(prev_n * (prev_p / self.p0)), 10)

    def bayesian_update(self, successes, trials, prior_alpha=1, prior_beta=1):
        # 通过成功样本数和总样本数进行贝叶斯更新
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + trials - successes
        estimated_rate = posterior_alpha / (posterior_alpha + posterior_beta)
        return estimated_rate

    def sequential_probability_ratio_test(self, observed, p1, confidence_level):
        observed = np.array(observed)
        # 计算似然比，使用 NumPy 的矢量化运算提高效率
        likelihood_ratio = np.prod(np.power(p1 / self.p0, observed) * np.power((1 - p1) / (1 - self.p0), 1 - observed))
        A = (self.beta / (1 - confidence_level))  # 基于信度调整 A
        B = ((1 - self.beta) / confidence_level)  # 基于信度调整 B

        if likelihood_ratio <= A:
            return "拒收"  # 拒绝 H0，接受 H1（拒收批次）
        elif likelihood_ratio >= B:
            return "接收"  # 接受 H0（接收批次）
        else:
            return "继续抽样"

    def train_model(self, X, y):
        # 计算类别权重，使模型对不平衡数据更加鲁棒
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(enumerate(class_weights))

        # 训练和测试集划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 设置类权重，减少训练时间
        self.model.set_params(class_weight=class_weight_dict)
        self.model.fit(X_train, y_train)

        # 测试集准确率
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"模型训练完成，测试集准确率: {accuracy:.2f}")

    def predict_defect_rate(self, features):
        try:
            # 使用模型预测次品率
            prediction = self.model.predict_proba([features])[0][1]
        except Exception as e:
            print(f"预测次品率时出错: {e}")
            prediction = self.p0  # 使用标称次品率作为默认值
        print(f"预测次品率: {prediction:.2f}")
        return prediction

    def compute_cost(self, n_samples, decision, defect_rate, loss_per_defect, dismantle_cost):
        # 计算采样成本和潜在的损失
        sampling_cost = self.cs * n_samples
        if decision == "拒收":
            expected_loss = defect_rate * loss_per_defect
        else:
            expected_loss = 0  # 接收的情况下，没有损失
        total_cost = sampling_cost + expected_loss + dismantle_cost
        return total_cost

    def run_adaptive_sampling(self, true_defect_rate, confidence_level, features, loss_per_defect=40, dismantle_cost=5):
        np.random.seed(42)
        samples = []
        predicted_defect_rate = self.predict_defect_rate(features)
        sample_size = self.max_samples  # 初始样本大小
        prev_p = predicted_defect_rate
        bayesian_rate = prev_p  # 初始化贝叶斯估计
        for k in range(1, 6):  # 5 个阶段
            sample_size = self.compute_sample_size(sample_size, prev_p)
            sample_size = min(sample_size, self.max_samples)  # 确保不超过最大样本量

            # 早期停止条件：贝叶斯估计已经非常接近拒收或接收阈值
            if bayesian_rate < 0.01 or bayesian_rate > 0.99:
                break  # 提前终止抽样

            for i in range(sample_size):
                sample = np.random.binomial(1, true_defect_rate)
                samples.append(sample)
                bayesian_rate = self.bayesian_update(sum(samples), len(samples))
                decision = self.sequential_probability_ratio_test(samples, bayesian_rate, confidence_level)
                print(f"第 {k} 阶段，第 {i+1} 次抽样：实际次品率 = {true_defect_rate:.2f}，贝叶斯估计次品率 = {bayesian_rate:.2f}")
                if decision != "继续抽样":
                    total_cost = self.compute_cost(len(samples), decision, true_defect_rate, loss_per_defect, dismantle_cost)
                    print(f"在第 {k} 阶段，第 {i+1} 次抽样后做出决策：{decision}，总成本 = {total_cost:.2f}")
                    return decision, len(samples), total_cost
            prev_p = bayesian_rate  # 更新估计的次品率

        print("达到最大抽样次数，仍未做出决策。")
        total_cost = self.compute_cost(len(samples), "继续抽样", true_defect_rate, loss_per_defect, dismantle_cost)
        return "继续抽样", self.max_samples, total_cost

# 数据生成
np.random.seed(42)
n_samples = 500  # 数据规模
n_features = 5  # 特征数量

# 生成特征数据，特征值在 [0, 1] 范围内
features = np.random.rand(n_samples, n_features)

# 生成标签，基于次品率
def generate_labels(n_samples, defect_rate):
    return np.random.binomial(1, defect_rate, n_samples)

# 假设初始次品率为 10%
labels = generate_labels(n_samples, defect_rate=0.1)

# 初始化抽样计划
sampling_plan = AdvancedAdaptiveSamplingPlan(p0=0.1, alpha=0.01, beta=0.05, max_samples=100, cs=1)
sampling_plan.train_model(features, labels)

# 测试数据，假设特征有合理值
test_features = np.random.rand(n_features)

# 95% 信度情形
decision_95, samples_95, cost_95 = sampling_plan.run_adaptive_sampling(true_defect_rate=0.12, confidence_level=0.95, features=test_features)
print(f"95% 信度下结果: {decision_95}，抽样次数: {samples_95}，总成本: {cost_95:.2f}")

# 90% 信度情形
decision_90, samples_90, cost_90 = sampling_plan.run_adaptive_sampling(true_defect_rate=0.08, confidence_level=0.90, features=test_features)
print(f"90% 信度下结果: {decision_90}，抽样次数: {samples_90}，总成本: {cost_90:.2f}")
