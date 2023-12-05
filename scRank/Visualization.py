import torch
from TrainPre import transform_test_exp

# Data Preparation

def create_tensor(data_matrix):
    tensor = torch.from_numpy(np.array(data_matrix))
    return torch.tensor(tensor, dtype=torch.float32)


# Model Prediction

def model_predict(model, data_tensor, mode):
    _, prob_scores, _ = model(data_tensor)

    if mode == "Cox":
        pred_prob = prob_scores.detach().numpy().reshape(-1, 1)

    elif mode == "Bionomial":
        pred_label = torch.max(prob_scores, dim=1).indices.detach().numpy().reshape(-1, 1)
        pred_prob = prob_scores[:, 1].detach().numpy().reshape(-1, 1)

    elif mode == "Regression":
        pred_prob = prob_scores.detach().numpy().reshape(-1, 1)
        pred_label = pred_prob


    return pred_label, pred_prob

# Probability Score Distribution Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_prob_distribution(pred_prob_bulk, pred_prob_sc, output_path):
    sns.distplot(pred_prob_bulk, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='Bulk')
    sns.distplot(pred_prob_sc, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='Single Cell')

    plt.title('Density Plot')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend(title='Sample Type', loc='upper left')
    plt.savefig(output_path)
    plt.show()
    plt.close()

# Evaluation on Other Data
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

def evaluate_on_test_data(model, test_set, data_path, save_path, bulk_gene_pairs_mat):
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)

    save_path_ = os.path.join(save_path, "bulk_test")

    if not (os.path.exists(save_path_)):
        os.makedirs(save_path_)

    for data_id in test_set:
        test_bulk_clinical = pd.read_table(os.path.join(data_path, data_id + "_meta.csv"), sep=",", index_col=0)
        test_bulk_clinical.columns = ["Group", "OS_status", "OS_time"]
        test_bulk_clinical['Group'] = test_bulk_clinical['Group'].apply(lambda x: 0 if x in ['PR', 'CR', 'CRPR'] else 1)

        test_bulk_exp = pd.read_csv(os.path.join(data_path, data_id + "_exp.csv"), index_col=0)
        test_bulk_exp_gene_pairs_mat = transform_test_exp(bulk_gene_pairs_mat, test_bulk_exp)
        test_exp_tensor_bulk = create_tensor(test_bulk_exp_gene_pairs_mat)

        test_pred_label, _ = model_predict(model, test_exp_tensor_bulk, mode = "Bionomial")
        test_bulk_clinical["scRank_Label"] = test_pred_label.flatten()
        test_bulk_clinical.to_csv(os.path.join(save_path_, data_id + "_predict_score.csv"))

        true_labels_bulk = test_bulk_clinical['Group']
        predicted_labels_bulk = test_bulk_clinical["scRank_Label"]

        cm = confusion_matrix(true_labels_bulk, predicted_labels_bulk)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig(os.path.join(save_path_, f'Pred on bulk: {data_id}.png'))
        plt.show()
        plt.close()

# Functions for RNA-seq to scRNA with Regression mode
from scipy.stats import mannwhitneyu

def create_boxplot(data, title, ax, group_column='True Label', score_column='Predicted Score'):
    sns.boxplot(x=group_column, y=score_column, data=data, ax=ax)
    ax.set_title(title)

    # Statistical test
    group0 = data[data[group_column] == 0][score_column]
    group1 = data[data[group_column] == 1][score_column]
    stat, p_value = mannwhitneyu(group0, group1)
    ax.text(0.5, 0.95, f'p = {p_value:.2e}', ha='center', va='center', transform=ax.transAxes)

def create_density_plot(data, label, ax, title):
    sns.kdeplot(data, shade=True, linewidth=3, label=label, ax=ax)
    ax.set_title(title)
    ax.legend()

def create_hist_plot(data, ax, title):
    sns.histplot(data, bins=20, kde=True, ax=ax)
    ax.set_title(title)

def create_comparison_density_plot(data1, label1, data2, label2, ax, title):
    sns.kdeplot(data1, shade=True, linewidth=3, label=label1, ax=ax)
    sns.kdeplot(data2, shade=True, linewidth=3, label=label2, ax=ax)
    ax.set_title(title)
    ax.legend()

