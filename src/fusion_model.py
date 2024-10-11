# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: 数据预处理
# 加载数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
data = pd.read_csv(url, names=column_names)

# 特征工程与预处理
# 将分类特征Sex编码为数值类型
data["Sex"] = data["Sex"].map({"M": 0, "F": 1, "I": 2})

# 将目标变量Rings转化为年龄
data["Age"] = data["Rings"] + 1.5  # 年龄 = 环数 + 1.5
data = data.drop("Rings", axis=1)

# 分离特征和目标变量
X = data.drop("Age", axis=1)
y = data["Age"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: 基模型训练
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义基模型
base_models = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.2)),
    ("mlp", MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
]

# Step 3: 模型融合
# 堆叠模型（Stacking Regressor）
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),  # 使用线性回归作为元模型
    cv=5  # 交叉验证折数
)

# 训练堆叠模型
stacking_model.fit(X_train, y_train)

# Step 4: 预测与评估
# 对测试集进行预测
y_pred = stacking_model.predict(X_test)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Stacking Model MSE: {rmse:.4f}")
print(f"Stacking Model R^2: {r2:.4f}")

# # Step 5: 结果解释与可视化
# import shap

# # 使用SHAP对随机森林基模型进行解释
# explainer = shap.Explainer(base_models[0][1].predict, X_train)  # 选择随机森林模型进行解释
# shap_values = explainer(X_test)
# shap.summary_plot(shap_values, X_test, feature_names=data.columns[:-1])  # 绘制SHAP值的汇总图

