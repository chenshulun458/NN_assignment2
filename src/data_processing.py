# src/data_processing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_and_clean_data():
    # 加载数据
    abalone = fetch_ucirepo(id=1)

    #Clean the data (convert M and F to 0 and 1) DATA PROCESSING POINT 1
    abalone['data']['features'].loc[:, 'Sex'] = abalone['data']['features']['Sex'].map({'M': -1, 'F': 1, 'I': 0})

    # 合并 features 和 targets
    merged_data = pd.concat([abalone['data']['features'], abalone['data']['targets']], axis=1)
    return merged_data


def visualize_data(merged_data):
    # Develop a correlation map using a heatmap and discuss major observations  DATA PROCESSING POINT 2
    corr = merged_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

    return corr

def plot_top_two_correlated_features(corr, abalone_data):
    # Develop a correlation map using a heatmap and discuss major observations  DATA PROCESSING POINT 3
    top_two_corr = corr['Rings'].drop('Rings').abs().nlargest(2).index
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(abalone_data['data']['features'][top_two_corr[0]], abalone_data['data']['targets']['Rings'])
    plt.xlabel(top_two_corr[0])
    plt.ylabel('Rings')
    plt.title(f'Scatter plot of {top_two_corr[0]} vs Rings')

    plt.subplot(1, 2, 2)
    plt.scatter(abalone_data['data']['features'][top_two_corr[1]], abalone_data['data']['targets']['Rings'])
    plt.xlabel(top_two_corr[1])
    plt.ylabel('Rings')
    plt.title(f'Scatter plot of {top_two_corr[1]} vs Rings')
    plt.show()

def plot_top_two_features_and_target_histograms(corr, abalone_data):
    # Create histograms of the two most correlated features, and the ring-age DATA PROCESSING POINT 4
    top_two_corr = corr['Rings'].drop('Rings').abs().nlargest(2).index

    # 创建直方图
    plt.figure(figsize=(15, 5))

    # 绘制第一个最相关特征的直方图
    plt.subplot(1, 3, 1)
    abalone_data['data']['features'][top_two_corr[0]].hist(bins=30)
    plt.xlabel(top_two_corr[0])
    plt.title(f'Histogram of {top_two_corr[0]}')

    # 绘制第二个最相关特征的直方图
    plt.subplot(1, 3, 2)
    abalone_data['data']['features'][top_two_corr[1]].hist(bins=30)
    plt.xlabel(top_two_corr[1])
    plt.title(f'Histogram of {top_two_corr[1]}')

    # 绘制目标特征 'Rings' 的直方图
    plt.subplot(1, 3, 3)
    abalone_data['data']['targets']['Rings'].hist(bins=30)
    plt.xlabel('Rings')
    plt.title('Histogram of Rings')

    # 展示图形
    plt.show()

# Create a 60/40 train/test split - which takes a random seed based on the experiment number to create a new dataset for every experiment DATA PROCESSING POINT 5
def split_data(merged_data, experiment_number=1):
    X = merged_data.drop(columns='Rings')
    y = merged_data[['Rings']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=experiment_number)

    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

if __name__ == "__main__":
    abalone = fetch_ucirepo(id=1)
    merged_data = load_and_clean_data()
    visualize_data(merged_data)
    plot_top_two_correlated_features(visualize_data(merged_data),abalone)
    plot_top_two_features_and_target_histograms(visualize_data(merged_data), abalone)
    X_train, X_test, y_train, y_test = split_data(merged_data)
    save_data(X_train, X_test, y_train, y_test)
