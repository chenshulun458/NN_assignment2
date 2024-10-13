import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison_separately(rmse_dict, r2_dict, auc_dict, accuracy_dict):
    """Visualize the comparison results from multiple models"""

    models = list(rmse_dict.keys())
    
    # Plot RMSE comparison
    plt.figure(figsize=(8, 6))
    for model in models:
        if model != 'Logistic Regression':
            plt.plot(range(len(rmse_dict[model])), rmse_dict[model], label=model)
    plt.xlabel('Experiment')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/rmse_comparison.png")
    plt.show()

    # Plot R-squared comparison
    plt.figure(figsize=(8, 6))
    for model in models:
        if model != 'Logistic Regression':
            plt.plot(range(len(r2_dict[model])), r2_dict[model], label=model)
    plt.xlabel('Experiment')
    plt.ylabel('R-squared')
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/r2_comparison.png")
    plt.show()

    # Boxplot for AUC
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[auc_dict[model] for model in models], palette="Set2")
    plt.xticks(range(len(models)), models)
    plt.ylabel('AUC')
    plt.savefig("experiments/auc_comparison.png")
    plt.show()

    # Boxplot for Accuracy
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[accuracy_dict[model] for model in models], palette="Set2")
    plt.xticks(range(len(models)), models)
    plt.ylabel('Accuracy')
    plt.savefig("experiments/accuracy_comparison.png")
    plt.show()


rmse_dict = {
    'Linear Regression': [2.2162, 2.2040, 2.2589, 2.2209, 2.2246, 2.2191, 2.2222, 2.2988, 2.2994, 2.2393], 
    'Logistic Regression': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Neural Network': [2.1511, 2.0435, 2.1234, 2.0594, 2.0930, 2.0883, 2.0788, 2.2665, 2.2180, 2.1882],
}

r2_dict = {
    'Linear Regression': [0.4944, 0.5093, 0.5399, 0.5335, 0.5183, 0.4954, 0.5132, 0.4860, 0.5002, 0.5265], 
    'Logistic Regression': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Neural Network': [0.5237, 0.5782, 0.5934, 0.5989, 0.5737, 0.5531, 0.5740, 0.5004, 0.5350, 0.5478],
}

auc_dict = {
    'Linear Regression': [0.9371, 0.9515, 0.9538, 0.9499, 0.9429, 0.9555, 0.9469, 0.9508, 0.9482, 0.9473], 
    'Logistic Regression': [0.8845, 0.8821, 0.8683, 0.8875, 0.9078, 0.8869, 0.8833, 0.8731, 0.8845, 0.8791],
    'Neural Network': [0.9459, 0.9600, 0.9537, 0.9512, 0.9467, 0.9572, 0.9485, 0.9557, 0.9532, 0.9470],
}

accuracy_dict = {
    'Linear Regression': [0.9318, 0.9246, 0.9348, 0.9282, 0.9288, 0.9348, 0.9390, 0.9288, 0.9330, 0.9306], 
    'Logistic Regression': [0.8845, 0.8821, 0.8683, 0.8875, 0.9078, 0.8869, 0.8833, 0.8731, 0.8845, 0.8791],
    'Neural Network': [0.9360, 0.9354, 0.9366, 0.9306, 0.9246, 0.9348, 0.9306, 0.9348, 0.9354, 0.9282],
}

def plot_comparison_combined(rmse_dict, r2_dict, auc_dict, accuracy_dict):
    """Visualize the comparison results from multiple models in a 2x2 grid"""

    models = list(rmse_dict.keys())
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

    # Plot RMSE comparison
    for model in models:
        if model != 'Logistic Regression':
            axs[0, 0].plot(range(len(rmse_dict[model])), rmse_dict[model], label=model)
    axs[0, 0].set_title('RMSE Comparison', fontsize=10)
    axs[0, 0].set_xlabel('Experiment', fontsize=8)
    axs[0, 0].set_ylabel('RMSE', fontsize=8)
    axs[0, 0].legend(fontsize=6)
    axs[0, 0].grid(True)

    # Plot R-squared comparison
    for model in models:
        if model != 'Logistic Regression':
            axs[0, 1].plot(range(len(r2_dict[model])), r2_dict[model], label=model)
    axs[0, 1].set_title('R-squared Comparison', fontsize=10)
    axs[0, 1].set_xlabel('Experiment', fontsize=8)
    axs[0, 1].set_ylabel('R-squared', fontsize=8)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=6)
    axs[0, 1].tick_params(axis='both', which='minor', labelsize=6)
    axs[0, 1].legend(fontsize=6)
    axs[0, 1].grid(True)

    # Boxplot for AUC
    sns.boxplot(data=[auc_dict[model] for model in models], palette="Set2", ax=axs[1, 0])
    axs[1, 0].set_title('AUC Comparison', fontsize=10)
    axs[1, 0].set_xticks(range(len(models)))
    axs[1, 0].set_xticklabels(models, fontsize=6)
    axs[1, 0].set_ylabel('AUC', fontsize=8)

    # Boxplot for Accuracy
    sns.boxplot(data=[accuracy_dict[model] for model in models], palette="Set2", ax=axs[1, 1])
    axs[1, 1].set_title('Accuracy Comparison', fontsize=10)
    axs[1, 1].set_xticks(range(len(models)))
    axs[1, 1].set_xticklabels(models, fontsize=6)
    axs[1, 1].set_ylabel('Accuracy', fontsize=8)

    plt.tight_layout()
    plt.savefig("experiments/comparison_combined1.png")
    plt.show()

plot_comparison_combined(rmse_dict, r2_dict, auc_dict, accuracy_dict)
