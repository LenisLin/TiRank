import torch
import os
import pickle
import math

import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gseapy as gp

from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from .Dataloader import transform_test_exp

# Data Preparation

def create_tensor(data_matrix):
    tensor = torch.from_numpy(np.array(data_matrix))
    return torch.tensor(tensor, dtype=torch.float32)

# Plot loss function
def plot_loss(train_loss_dict, alphas, savePath="./loss_on_epoch.png"):
    """
    Plots the change in different types of loss values across epochs.

    Args:
    train_loss_dict (dict): Dictionary containing the loss values for each epoch. 
                            The keys should be 'Epoch_x' and the values should be 
                            dictionaries of different loss types.
    """
    # Check if the dictionary is empty
    if not train_loss_dict:
        print("The loss dictionary is empty.")
        return

    # Determine the loss types from the first epoch
    loss_types = list(train_loss_dict[next(iter(train_loss_dict))].keys())
    
    # Extracting the number of epochs
    epochs = range(1, len(train_loss_dict) + 1)
    
    # Reformatting the data for plotting
    loss_data = {loss_type: [epoch_data[loss_type] for epoch_data in train_loss_dict.values()] for loss_type in loss_types}
    
    # Plotting
    plt.figure(figsize=(10, 6))

    for loss_type, losses in loss_data.items():
        plt.plot(epochs, losses, label=loss_type)

    plt.title('Loss Value Change Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(savePath, bbox_inches='tight', pad_inches=1) 
    plt.show()
    plt.clf()
    plt.close()

    return None


# Model Prediction

def model_predict(model, data_tensor, mode):
    _, prob_scores, _ = model(data_tensor)

    if mode == "Cox":
        pred_prob = prob_scores.detach().numpy().reshape(-1, 1)

    elif mode == "Classification":
        pred_label = torch.max(prob_scores, dim=1).indices.detach().numpy().reshape(-1, 1)
        pred_prob = prob_scores[:, 1].detach().numpy().reshape(-1, 1)

    elif mode == "Regression":
        pred_prob = prob_scores.detach().numpy().reshape(-1, 1)
        pred_label = pred_prob


    return pred_label, pred_prob

# Probability Score Distribution Visualization
def plot_score_distribution(savePath):
    savePath_3 = os.path.join(savePath,"3_Analysis")

    ## Load data
    f = open(os.path.join(savePath_3, 'saveDF_bulk.pkl'), 'rb')
    bulk_PredDF = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_3, 'saveDF_sc.pkl'), 'rb')
    sc_PredDF = pickle.load(f)
    f.close()

    pred_prob_sc = sc_PredDF["Pred_score"]  # scRNA
    pred_prob_bulk = bulk_PredDF["Pred_score"]  # Bulk RNA

    ## Plot
    sns.distplot(pred_prob_bulk, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='Bulk')
    sns.distplot(pred_prob_sc, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='Single Cell/Spot')

    plt.title('Density Plot')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend(title='Sample Type', loc='upper left')
    plt.savefig(os.path.join(savePath_3, 'TiRank Pred Score Distribution.png'), bbox_inches='tight', pad_inches=1)
    plt.show()
    plt.close()

    return None

# DEG analysis
def DEG_analysis(savePath, fc_threshold = 2, Pvalue_threshold = 0.05, do_p_adjust = True):
    savePath_3 = os.path.join(savePath,"3_Analysis")

    ## Load final single-cell data
    adata = sc.read_h5ad(os.path.join(savePath_3,"final_anndata.h5ad"))

    ## DEG
    sc.tl.rank_genes_groups(adata, 'Rank_Label', groups=['Rank+'], reference='Rank-', method='wilcoxon')

    ## Extract dataframe
    df_DEG = pd.concat([
        pd.DataFrame(adata.uns['rank_genes_groups']['names']),
        pd.DataFrame(adata.uns['rank_genes_groups']['scores']),
        pd.DataFrame(adata.uns['rank_genes_groups']['pvals']),
        pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj']),
        pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'])], 
        axis=1)
    
    df_DEG.columns = ['GeneSymbol', 'Scores', 'Pvalue', 'Pvalue_adj', 'LogFoldChange']
    df_DEG.index = df_DEG["GeneSymbol"]
    
    df_DEG.to_csv(os.path.join(savePath_3,"All DEGs dataframe.csv"))

    df_DEG = df_DEG[np.abs(df_DEG['LogFoldChange']) >= math.log2(fc_threshold)]
    
    if do_p_adjust:
        df_DEG = df_DEG[np.abs(df_DEG['Pvalue_adj']) <= Pvalue_threshold]
    else:
        df_DEG = df_DEG[np.abs(df_DEG['Pvalue']) <= Pvalue_threshold]
    
    df_DEG = df_DEG.sort_values(by='LogFoldChange', ascending=False)
    df_DEG.to_csv(os.path.join(savePath_3,"Differentially expressed genes data frame.csv"))

    return None

# volcano plot display the differential expressed genes
def DEG_volcano(savePath, fc_threshold=2, Pvalue_threshold=0.05, do_p_adjust=True, top_n = 5):
    # Path for saving analysis results
    savePath_3 = os.path.join(savePath, "3_Analysis")

    # Load data from the specified file
    result = pd.read_csv(os.path.join(savePath_3, "All DEGs dataframe.csv"), index_col=1)
    result['group'] = 'black'  # Default color for all points

    log2FC = math.log2(fc_threshold)
    result['-lg10Qvalue'] = -(np.log10(result['Pvalue_adj']))
    result['-lg10Pvalue'] = -(np.log10(result['Pvalue']))

    # Coloring points based on thresholds and adjusted P-values
    if do_p_adjust:
        # Marking significant upregulated genes in red
        result.loc[(result['LogFoldChange'] >= log2FC) & (result["Pvalue_adj"] <= Pvalue_threshold), 'group'] = 'tab:red'
        # Marking significant downregulated genes in blue
        result.loc[(result['LogFoldChange'] <= (-log2FC)) & (result["Pvalue_adj"] <= Pvalue_threshold), 'group'] = 'tab:blue'
        # Marking non-significant genes in grey
        result.loc[result["Pvalue_adj"] > Pvalue_threshold, 'group'] = 'dimgrey'
    else:
        result.loc[(result['LogFoldChange'] >= log2FC) & (result["Pvalue"] <= Pvalue_threshold), 'group'] = 'tab:red'
        result.loc[(result['LogFoldChange'] <= (-log2FC)) & (result["Pvalue"] <= Pvalue_threshold), 'group'] = 'tab:blue'
        result.loc[result["Pvalue"] > Pvalue_threshold, 'group'] = 'dimgrey'

    # Define axis display range
    xmin, xmax, ymin, ymax = -8, 8, -10, 100

    # Create scatter plot
    fig = plt.figure(figsize=plt.figaspect(7/6))  # Set figure aspect ratio (height/width)
    ax = fig.add_subplot()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), title='')
    ax.scatter(result['LogFoldChange'], result['-lg10Qvalue'], s=2, c=result['group'])
    
    # Annotate points
    # top N up-regulated genes
    top_up = result[(result['LogFoldChange'] >= log2FC) & (result['Pvalue_adj'] <= Pvalue_threshold)].nlargest(top_n, 'LogFoldChange')
    for index, row in top_up.iterrows():
        ax.annotate(row.name, (row['LogFoldChange'], row['-lg10Qvalue']), textcoords="offset points", xytext=(0,10), ha='center')

    # top N down-regulated genes
    top_down = result[(result['LogFoldChange'] <= -log2FC) & (result['Pvalue_adj'] <= Pvalue_threshold)].nsmallest(top_n, 'LogFoldChange')
    for index, row in top_down.iterrows():
        ax.annotate(row.name, (row['LogFoldChange'], row['-lg10Qvalue']), textcoords="offset points", xytext=(0,10), ha='center')

    
    ax.set_ylabel('-Log10(Q value)', fontweight='bold')
    ax.set_xlabel('Log2 (fold change)', fontweight='bold')
    ax.spines['right'].set_visible(False)  # Remove right border
    ax.spines['top'].set_visible(False)    # Remove top border
    # Draw horizontal and vertical lines
    ax.vlines(-log2FC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1) # Vertical line for negative log2FC
    ax.vlines(log2FC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1) # Vertical line for positive log2FC
    ax.hlines(-math.log10(Pvalue_threshold), xmin, xmax, color='dimgrey', linestyle='dashed', linewidth=1) # Horizontal line for Pvalue threshold

    # Set x and y axis ticks
    ax.set_xticks(range(-8, 8, 2))  # x-axis ticks with start point and step
    ax.set_yticks(range(-10, 100, 20))  # y-axis ticks with start point and step

    # Save the figure
    fig.savefig(os.path.join(savePath_3, 'DEG_volcano_plot.png'), dpi=300)
    plt.show()

    return None

# Pathway enrichment analysis
def Pathway_Enrichment(savePath,database = "KEGG_2016"):
    savePath_3 = os.path.join(savePath, "3_Analysis")
    result = pd.read_csv(os.path.join(savePath_3, "Differentially expressed genes data frame.csv"), index_col=1)

    # up and down genes
    upgenes = result[result["LogFoldChange"]>0]['GeneSymbol'].tolist()
    downgenes = result[result["LogFoldChange"]<0]['GeneSymbol'].tolist()
    
    upenr = gp.enrichr(
        gene_list=upgenes,
                 gene_sets=database,
                 organism='Human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=os.path.join(savePath_3,'enrichr','up'),
                 no_plot=True,
                 cutoff=0.5 # test dataset, use lower value from range(0,1)
                )

    downenr = gp.enrichr(
        gene_list=downgenes,
                 gene_sets=database,
                 organism='Human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=os.path.join(savePath_3,'enrichr','down'),
                 no_plot=True,
                 cutoff=0.5 # test dataset, use lower value from range(0,1)
                )
    
    database_name = '_'.join(database)

    if np.min(downenr.results["Adjusted P-value"]) > 0.05:
        print("Up regulated genes do not enrich in any pathway of "+database_name+"!")
    else:
        gp.plot.dotplot(upenr.results, title="Up regulated genes enrich in "+database_name)
        plt.savefig(os.path.join(savePath_3,'enrichr','up',"Up regulated genes enrich in "+database_name+".png"), bbox_inches='tight', pad_inches=1)
        plt.close()

    if np.min(downenr.results["Adjusted P-value"]) > 0.05:
        print("Down regulated genes do not enrich in any pathway of "+database_name+"!")
    else:
        gp.plot.dotplot(downenr.results, title="Down regulated genes enrich in "+database_name)
        plt.savefig(os.path.join(savePath_3,'enrichr','down',"Down regulated genes enrich in "+database_name+".png"), bbox_inches='tight', pad_inches=1)
        plt.close()

    upenr.results.to_csv(os.path.join(savePath_3,'enrichr','up',"Pathway enrichment in "+database_name+" data frame.csv"))
    downenr.results.to_csv(os.path.join(savePath_3,'enrichr','down',"Pathway enrichment in "+database_name+" data frame.csv"))

    return None

# Upload new meta files to analysis
def upload_metafile(savePath, uploadPath):

    # Path for saving analysis results
    savePath_3 = os.path.join(savePath, "3_Analysis")
    savePath_selecting = os.path.join(savePath_3, "selecting")

    if uploadPath.lower().endswith('.csv'):
        uploadFile = pd.read_csv(uploadPath, index_col=0)
    elif uploadPath.lower().endswith('.txt'):
        uploadFile = pd.read_table(uploadPath, index_col=0,sep='\t')
    else:
        uploadFile = pd.read_excel(uploadPath, index_col=0) 

    adata = sc.read_h5ad(os.path.join(savePath_3,"final_anndata.h5ad"))

    common_elements = adata.obs.index.intersection(uploadFile.index)
    if(len(common_elements)==0):
        print("The rownames of upload meta data was not match with output file!")
    
    uploadFile = uploadFile.loc[common_elements,:]
    for select_col in uploadFile.columns:
        adata.obs[select_col] = uploadFile[select_col]
    
    adata.write_h5ad(os.path.join(savePath_selecting,"final_anndata_with_anno.h5ad"))
    return None

# Visualize sub-regions
## Just for Spatial transcriptomic data
def visua_sub(savePath, classCol, interes_class):
    # Path for saving analysis results
    savePath_3 = os.path.join(savePath, "3_Analysis")
    savePath_selecting = os.path.join(savePath_3, "selecting")

    adata = sc.read_h5ad(os.path.join(savePath_selecting,"final_anndata_with_anno.h5ad"))
    adata_sub = adata[adata.obs[classCol] == interes_class,:]
    if adata_sub.obs.shape[0] == 0:
        raise ValueError(f"The class '{interes_class}' was not in '{classCol}' of data!")

    adata.write_h5ad(os.path.join(savePath_selecting,"subset_final_anndata.h5ad"))

def visua_abundance(savePath, interes_type):
    # Path for saving analysis results
    savePath_3 = os.path.join(savePath, "3_Analysis")
    savePath_selecting = os.path.join(savePath_3, "selecting")
    adata = sc.read_h5ad(os.path.join(savePath_selecting,"final_anndata_with_anno.h5ad"))

    colors = ["lightblue", "lightyellow", "red"]
    
    spatial_location = pd.DataFrame(adata.obsm['spatial'])
    spatial_location.columns = ["x","y"]
    spatial_location.index = adata.obs.index

    # Selecting columns for visualization
    proportion = pd.DataFrame(adata.obs[interes_type]) 
    
    # Normalizing the data
    scaler = MinMaxScaler()
    proportion_scaled = pd.DataFrame(scaler.fit_transform(proportion))
    proportion_scaled.index = proportion.index
    proportion_scaled.columns = proportion.columns

    # Adding spatial location
    proportion_scaled['x'] = spatial_location['x']
    proportion_scaled['y'] = spatial_location['y']
    
    # Melting the data for seaborn
    mData = proportion_scaled.reset_index().melt(id_vars=['index', 'x', 'y'], var_name='Cell_Type', value_name='value')
    
    # Plotting
    g = sns.FacetGrid(mData, col="Cell_Type", hue="value", palette=colors)
    g.map(plt.scatter, 'x', 'y', s=0.2)
    g.set(xticks=[], yticks=[])
    g.fig.subplots_adjust(wspace=0, hspace=0)

    # Show plot
    plt.savefig(os.path.join(savePath_selecting,f"{interes_type} abudance on slice.png"))
    plt.close()

    return None

# Evaluation on Other Data

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

        test_pred_label, _ = model_predict(model, test_exp_tensor_bulk, mode = "Classification")
        test_bulk_clinical["TiRank_Label"] = test_pred_label.flatten()
        test_bulk_clinical.to_csv(os.path.join(save_path_, data_id + "_predict_score.csv"))

        true_labels_bulk = test_bulk_clinical['Group']
        predicted_labels_bulk = test_bulk_clinical["TiRank_Label"]

        cm = confusion_matrix(true_labels_bulk, predicted_labels_bulk)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig(os.path.join(save_path_, f'Pred on bulk: {data_id}.png'), bbox_inches='tight', pad_inches=1)
        plt.show()
        plt.close()

# Functions for RNA-seq to scRNA with Regression mode

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

