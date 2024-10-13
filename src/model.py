import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline



#1 Develop a linear regression model using all features for ring-age using 60 percent of data picked randomly for training and remaining for testing. Visualise your model prediction using appropriate plots. Report the correct metrics for the given model (i.e RMSE and R-squared score and classification score and AUC score and ROC plot. 

#2 Compare  linear/logistic regression model with all features, i) without normalising input data (taken from Step 1), ii) with normalising input data.

#3 Develop a linear/logistic regression model with two selected input features from the data processing step

#4 Compare the best approach from the above investigations using a neural network trained with SGD. You need to run some trial experiments to determine optimal hyperparameters, i.e number of hidden neurons and layers and learning rate etc. You can discuss your results and major observations about trial experiments (15 Marks) Note trial and error runs do not need multiple experiments (ie. 30 exp)

#5 Discuss the neural network with the linear model results. Discuss how you can improve model further

# 1. load data
def load_data():
    """load data"""
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_test = pd.read_csv('data/y_test.csv')
    return X_train, X_test, y_train, y_test

# 2. model training and evaluation

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    """train and evaluate model"""
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)


    threshold = 7
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_test_binary = (y_test >= threshold).astype(int)


    auc = roc_auc_score(y_test_binary, y_pred)
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    
    return rmse, r2, auc, accuracy

# 2. model training and evaluation for classification only
def train_and_evaluate_accuracy(X_train, X_test, y_train, y_test, model):
    """Train and evaluate the model for accuracy only"""
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)

    # Convert predictions to binary classification based on a threshold
    threshold = 0.5
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_test_binary = (y_test >= threshold).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    
    return accuracy

# 3. visualization
def visualize_predictions(y_test, y_pred, y_test_binary, y_pred_binary, auc, accuracy, title_suffix=''):
    """visualize predictions and plot ROC curve"""
    plt.figure(figsize=(15, 5))

    # Actual vs Predicted scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Rings')
    plt.ylabel(f'Predicted Rings {title_suffix}')
    plt.title(f'Actual vs Predicted Rings {title_suffix}')

    # Residuals histogram
    plt.subplot(1, 3, 2)
    sns.histplot(y_test['Rings'] - y_pred.flatten(), bins=30, kde=True)
    plt.xlabel(f'Residuals {title_suffix}')
    plt.title(f'Residuals Distribution {title_suffix}')

    # ROC curve plot
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {auc:.4f}, Accuracy = {accuracy:.4f})')

    plt.tight_layout()
    plt.show()

def plot_experiment_results(rmse_list, r2_list, auc_list, accuracy_list):
    """Visualize the results from multiple experiments"""
    plt.figure(figsize=(12, 8))

    # RMSE 和 R² 的折线图
    plt.subplot(2, 2, 1)
    plt.plot(rmse_list, label='RMSE', marker='o')
    plt.xlabel('Experiment')
    plt.ylabel('RMSE')
    plt.title('RMSE over Multiple Experiments')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(r2_list, label='R-squared', color='orange', marker='o')
    plt.xlabel('Experiment')
    plt.ylabel('R-squared')
    plt.title('R-squared over Multiple Experiments')
    plt.grid(True)

    # AUC 和 Accuracy 的箱线图
    plt.subplot(2, 2, 3)
    sns.boxplot(data=[auc_list, accuracy_list], palette="Set2")
    plt.xticks([0, 1], ['AUC', 'Accuracy'])
    plt.title('AUC and Accuracy Distribution over Multiple Experiments')

    plt.tight_layout()
    plt.show()

# 4. 运行多次实验的函数
def run_experiments(n_experiments, model_func, X, y, selected_features=None,case='regression'):
    """运行 n 次实验并报告结果"""
    if case == 'regression':
        rmse_list = []
        r2_list = []
        auc_list = []
        accuracy_list = []

        for experiment_number in range(n_experiments):
            # 每次重新分割数据集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=experiment_number)

            # 如果是选择了特定的特征
            if selected_features:
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]

            # 调用模型函数
            rmse, r2, auc, accuracy = model_func(X_train, X_test, y_train, y_test)

            # 保存每次实验的结果
            rmse_list.append(rmse)
            r2_list.append(r2)
            auc_list.append(auc)
            accuracy_list.append(accuracy)

            print(f'Experiment {experiment_number + 1} - RMSE: {rmse:.4f}, R-squared: {r2:.4f}, AUC: {auc:.4f}, Accuracy: {accuracy:.4f}')

        # 计算和显示结果的均值与标准差
        print("\nFinal Results after running multiple experiments:")
        print(f'Average RMSE: {np.mean(rmse_list):.4f}, RMSE Std Dev: {np.std(rmse_list):.4f}')
        print(f'Average R-squared: {np.mean(r2_list):.4f}, R-squared Std Dev: {np.std(r2_list):.4f}')
        print(f'Average AUC: {np.mean(auc_list):.4f}, AUC Std Dev: {np.std(auc_list):.4f}')
        print(f'Average Accuracy: {np.mean(accuracy_list):.4f}, Accuracy Std Dev: {np.std(accuracy_list):.4f}')

        # 对实验结果进行可视化
        plot_experiment_results(rmse_list, r2_list, auc_list, accuracy_list)
    else:
        accuracy_list = []

        for experiment_number in range(n_experiments):
            # 每次随机分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=experiment_number)

            # 调用模型函数
            accuracy = model_func(X_train, X_test, y_train, y_test)

            # 保存accuracy
            accuracy_list.append(accuracy)

            print(f'Experiment {experiment_number + 1} - Accuracy: {accuracy:.4f}')

        # 显示平均accuracy
        print("\nFinal Results after running multiple experiments:")
        print(f'Average Accuracy: {np.mean(accuracy_list):.4f}, Accuracy Std Dev: {np.std(accuracy_list):.4f}')
        

# 5. 线性回归模型实验函数
def linear_regression_experiment_multiple(X, y, selected_features=None, n_experiments=30):
    """线性回归模型实验运行多次"""
    def model_func(X_train, X_test, y_train, y_test):
        lin_reg = LinearRegression()
        rmse, r2, auc, accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test, lin_reg)
        return rmse, r2, auc, accuracy

    run_experiments(n_experiments, model_func, X, y, selected_features)

# 5.1 logestic regression model
def logistic_regression_experiment_multiple(X, y, n_experiments=10):
    """Logistic Regression experiment calculating accuracy"""
    def model_func(X_train, X_test, y_train, y_test):
        # Convert to binary classification
        threshold = 20
        y_train_binary = (y_train >= threshold).astype(int)
        y_test_binary = (y_test >= threshold).astype(int)

        # 使用 Logistic 回归
        log_reg = LogisticRegression(C=100,class_weight='balanced')
        accuracy = train_and_evaluate_accuracy(X_train, X_test, y_train_binary, y_test_binary, log_reg)

        return accuracy

    run_experiments(n_experiments, model_func, X, y,case='classification')


# 6. 神经网络模型实验函数
def neural_network_experiment_multiple(X, y, hidden_layer_sizes=(50,), learning_rate_init=0.01, n_experiments=30):
    """神经网络实验运行多次"""
    def model_func(X_train, X_test, y_train, y_test):
        nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver='sgd', learning_rate_init=learning_rate_init, max_iter=1000, random_state=42)
        
        # 训练神经网络并预测
        nn.fit(X_train, y_train.values.ravel())
        y_pred = nn.predict(X_test)

        # 回归和分类计算
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # 将回归结果转为二分类任务
        threshold = 7
        y_pred_binary = (y_pred >= threshold).astype(int)
        y_test_binary = (y_test >= threshold).astype(int)
        
        auc = roc_auc_score(y_test_binary, y_pred)
        accuracy = accuracy_score(y_test_binary, y_pred_binary)

        # 返回四个值：rmse, r2, auc, accuracy
        return rmse, r2, auc, accuracy

    run_experiments(n_experiments, model_func, X, y)

# 7. 堆叠回归模型实验函数
def stacking_regression_experiment_multiple(X, y, n_experiments=30):
    """堆叠模型实验运行多次"""
    def model_func(X_train, X_test, y_train, y_test):
        # 定义基模型
        base_models = [
            ("lr", LinearRegression()),  
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.1)),
            ("mlp", MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.001,max_iter=2000, random_state=42))
        ]

        # 堆叠模型（Stacking Regressor）
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, random_state=42),
            cv=5
        )
        
        rmse, r2, auc, accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test, stacking_model)
        return rmse, r2, auc, accuracy

    run_experiments(n_experiments, model_func, X, y)

# 8. 主函数
if __name__ == '__main__':
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    # linear regression
    print("Running Linear Regression Experiments")
    linear_regression_experiment_multiple(X, y, n_experiments=10)

    # # 选择与目标最相关的特征进行线性回归实验运行30次
    # selected_features = ['Length', 'Diameter']  # 根据相关系数选择的特征
    # linear_regression_experiment_multiple(X, y, selected_features=selected_features, n_experiments=10)

    # lossitic regression
    print("\nRunning Logistic Regression Experiments")
    logistic_regression_experiment_multiple(X, y, n_experiments=10)
    

    # scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # NN
    print("\nRunning Neural Network Experiments")
    neural_network_experiment_multiple(X_scaled, y, hidden_layer_sizes=(100,50), learning_rate_init=0.001, n_experiments=10)


    # # MMFF
    # print("\nRunning Stacking Regression Experiments")
    # stacking_regression_experiment_multiple(X_scaled, y, n_experiments=10)

