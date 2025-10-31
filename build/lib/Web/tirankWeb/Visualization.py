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
from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix
from .dataloader import transform_test_exp


def create_tensor(data_matrix):
    tensor = torch.from_numpy(np.array(data_matrix))
    return torch.tensor(tensor, dtype=torch.float32)


# Plot loss function
def plot_loss(train_loss_dict, alphas, savePath="./img/loss_on_epoch.png"):
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
    loss_data = {loss_type: [epoch_data[loss_type] for epoch_data in train_loss_dict.values()] for loss_type in
                 loss_types}

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


def plot_score_distribution_(save_path):
    # Load data
    f = open(os.path.join(save_path, 'saveDF_bulk.pkl'), 'rb')
    bulk_PredDF = pickle.load(f)
    f.close()

    f = open(os.path.join(save_path, 'saveDF_sc.pkl'), 'rb')
    sc_PredDF = pickle.load(f)
    f.close()

    pred_prob_sc = sc_PredDF["Pred_score"]  # scRNA
    pred_prob_bulk = bulk_PredDF["Pred_score"]  # Bulk RNA

    # Plot
    sns.distplot(pred_prob_bulk, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='Bulk')
    sns.distplot(pred_prob_sc, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='Single Cell')

    plt.title('Density Plot')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend(title='Sample Type', loc='upper left')
    plt.savefig(os.path.join('./img/', 'TiRank Pred Score Distribution.png'), bbox_inches='tight', pad_inches=1)
    plt.savefig(os.path.join('./assets/', 'TiRank Pred Score Distribution.png'), bbox_inches='tight', pad_inches=1)
    plt.close()

    return None


def deg_analysis_(save_path, fc_threshold=2, Pvalue_threshold=0.05, do_p_adjust=True):
    # Load final single-cell data
    adata = sc.read_h5ad(os.path.join(save_path, "final_anndata.h5ad"))

    # DEG
    sc.tl.rank_genes_groups(adata, 'Rank_Label', groups=['Rank+'], reference='Rank-', method='wilcoxon')

    # Extract dataframe
    df_DEG = pd.concat([
        pd.DataFrame(adata.uns['rank_genes_groups']['names']),
        pd.DataFrame(adata.uns['rank_genes_groups']['scores']),
        pd.DataFrame(adata.uns['rank_genes_groups']['pvals']),
        pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj']),
        pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'])],
        axis=1)

    df_DEG.columns = ['GeneSymbol', 'Scores', 'Pvalue', 'Pvalue_adj', 'LogFoldChange']
    df_DEG.index = df_DEG["GeneSymbol"]

    df_DEG.to_csv(os.path.join(save_path, "All DEGs dataframe.csv"))

    df_DEG = df_DEG[np.abs(df_DEG['LogFoldChange']) >= math.log2(fc_threshold)]

    if do_p_adjust:
        df_DEG = df_DEG[np.abs(df_DEG['Pvalue_adj']) <= Pvalue_threshold]
    else:
        df_DEG = df_DEG[np.abs(df_DEG['Pvalue']) <= Pvalue_threshold]

    df_DEG = df_DEG.sort_values(by='LogFoldChange', ascending=False)
    df_DEG.to_csv(os.path.join(save_path, "Differentially expressed genes data frame.csv"))

    return None


# volcano plot display the differential expressed genes
def deg_volcano_(save_path, fc_threshold=2, Pvalue_threshold=0.05, do_p_adjust=True, top_n=5):
    # Load data from the specified file
    result = pd.read_csv(os.path.join(save_path, "All DEGs dataframe.csv"), index_col=1)
    result['group'] = 'black'  # Default color for all points

    log2FC = math.log2(fc_threshold)
    result['-lg10Qvalue'] = -(np.log10(result['Pvalue_adj']))
    result['-lg10Pvalue'] = -(np.log10(result['Pvalue']))

    # Coloring points based on thresholds and adjusted P-values
    if do_p_adjust:
        # Marking significant upregulated genes in red
        result.loc[
            (result['LogFoldChange'] >= log2FC) & (result["Pvalue_adj"] <= Pvalue_threshold), 'group'] = 'tab:red'
        # Marking significant downregulated genes in blue
        result.loc[
            (result['LogFoldChange'] <= (-log2FC)) & (result["Pvalue_adj"] <= Pvalue_threshold), 'group'] = 'tab:blue'
        # Marking non-significant genes in grey
        result.loc[result["Pvalue_adj"] > Pvalue_threshold, 'group'] = 'dimgrey'
    else:
        result.loc[(result['LogFoldChange'] >= log2FC) & (result["Pvalue"] <= Pvalue_threshold), 'group'] = 'tab:red'
        result.loc[
            (result['LogFoldChange'] <= (-log2FC)) & (result["Pvalue"] <= Pvalue_threshold), 'group'] = 'tab:blue'
        result.loc[result["Pvalue"] > Pvalue_threshold, 'group'] = 'dimgrey'

    # Define axis display range
    xmin, xmax, ymin, ymax = -8, 8, -10, 100

    # Create scatter plot
    fig = plt.figure(figsize=plt.figaspect(7 / 6))  # Set figure aspect ratio (height/width)
    ax = fig.add_subplot()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), title='')
    ax.scatter(result['LogFoldChange'], result['-lg10Qvalue'], s=2, c=result['group'])

    # Annotate points
    # top N up-regulated genes
    top_up = result[(result['LogFoldChange'] >= log2FC) & (result['Pvalue_adj'] <= Pvalue_threshold)].nlargest(top_n,
                                                                                                               'LogFoldChange')
    for index, row in top_up.iterrows():
        ax.annotate(row.name, (row['LogFoldChange'], row['-lg10Qvalue']), textcoords="offset points", xytext=(0, 10),
                    ha='center')

    # top N down-regulated genes
    top_down = result[(result['LogFoldChange'] <= -log2FC) & (result['Pvalue_adj'] <= Pvalue_threshold)].nsmallest(
        top_n, 'LogFoldChange')
    for index, row in top_down.iterrows():
        ax.annotate(row.name, (row['LogFoldChange'], row['-lg10Qvalue']), textcoords="offset points", xytext=(0, 10),
                    ha='center')

    ax.set_ylabel('-Log10(Q value)', fontweight='bold')
    ax.set_xlabel('Log2 (fold change)', fontweight='bold')
    ax.spines['right'].set_visible(False)  # Remove right border
    ax.spines['top'].set_visible(False)  # Remove top border
    # Draw horizontal and vertical lines
    ax.vlines(-log2FC, ymin, ymax, color='dimgrey', linestyle='dashed',
              linewidth=1)  # Vertical line for negative log2FC
    ax.vlines(log2FC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1)  # Vertical line for positive log2FC
    ax.hlines(-math.log10(Pvalue_threshold), xmin, xmax, color='dimgrey', linestyle='dashed',
              linewidth=1)  # Horizontal line for Pvalue threshold

    # Set x and y axis ticks
    ax.set_xticks(range(-8, 8, 2))  # x-axis ticks with start point and step
    ax.set_yticks(range(-10, 100, 20))  # y-axis ticks with start point and step

    # Save the figure
    fig.savefig(os.path.join('./img/', 'DEG_volcano_plot.png'), dpi=300)
    fig.savefig(os.path.join('./assets/', 'DEG_volcano_plot.png'), dpi=300)

    return None


# Pathway enrichment analysis
def pathway_enrichment(save_path, database="KEGG_2016"):
    result = pd.read_csv(os.path.join(save_path, "Differentially expressed genes data frame.csv"), index_col=1)

    # up and down genes
    upgenes = result[result["LogFoldChange"] > 0]['GeneSymbol'].tolist()
    downgenes = result[result["LogFoldChange"] < 0]['GeneSymbol'].tolist()

    upenr = gp.enrichr(
        gene_list=upgenes,
        gene_sets=database,
        organism='Human',  # don't forget to set organism to the one you desired! e.g. Yeast
        outdir=os.path.join(save_path, 'enrichr', 'up'),
        no_plot=True,
        cutoff=0.5  # test dataset, use lower value from range(0,1)
    )

    downenr = gp.enrichr(
        gene_list=downgenes,
        gene_sets=database,
        organism='Human',  # don't forget to set organism to the one you desired! e.g. Yeast
        outdir=os.path.join(save_path, 'enrichr', 'down'),
        no_plot=True,
        cutoff=0.5  # test dataset, use lower value from range(0,1)
    )

    database_name = '_'.join(database)

    if np.min(upenr.results["Adjusted P-value"]) > 0.05:
        print("Up regulated genes do not enrich in any pathway of " + database_name + "!")
    else:
        gp.plot.dotplot(upenr.results, title="Up regulated genes enrich in " + database_name)
        plt.savefig(os.path.join('./img/', "Up regulated genes enrich in " + database_name + ".png"),
                    bbox_inches='tight', pad_inches=1)
        plt.savefig(
            os.path.join('./assets/', "Up regulated genes enrich in " + database_name + ".png"),
            bbox_inches='tight', pad_inches=1)
        plt.close()

    if np.min(downenr.results["Adjusted P-value"]) > 0.05:
        print("Down regulated genes do not enrich in any pathway of " + database_name + "!")
    else:
        gp.plot.dotplot(downenr.results, title="Down regulated genes enrich in " + database_name)
        plt.savefig(
            os.path.join('./img/', "Down regulated genes enrich in " + database_name + ".png"),
            bbox_inches='tight', pad_inches=1)
        plt.savefig(
            os.path.join('./assets/', "Down regulated genes enrich in " + database_name + ".png"),
            bbox_inches='tight', pad_inches=1)
        plt.close()

    upenr.results.to_csv(
        os.path.join(save_path, 'enrichr', 'up', "Pathway enrichment in " + database_name + " data frame.csv"))
    downenr.results.to_csv(
        os.path.join(save_path, 'enrichr', 'down', "Pathway enrichment in " + database_name + " data frame.csv"))

    return None


def plot_score_umap_(save_path, infer_mode):
    sc_PredDF = pd.read_csv(
        os.path.join(save_path, "spot_predict_score.csv"), index_col=0
    )

    label_color_map = {
        "Rank+": "#DE6E66",
        "Rank-": "#5096DE",
        "Background": "lightgrey",
    }

    if infer_mode == "SC":
        f = open(os.path.join(save_path, "scAnndata.pkl"), "rb")
        scAnndata = pickle.load(f)
        f.close()

        scAnndata.obs["TiRank_Score"] = sc_PredDF["Rank_Score"]
        scAnndata.obs["TiRank_Label"] = sc_PredDF["Rank_Label"]

        sc.pl.umap(scAnndata, color="TiRank_Score", title="", show=False)
        plt.savefig(
            os.path.join('./assets/', "UMAP of TiRank Pred Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.show()
        plt.close()

        sc.pl.umap(
            scAnndata,
            color="TiRank_Label",
            title="",
            show=False,
            palette=label_color_map,
        )
        plt.savefig(
            os.path.join('./assets/', "UMAP of TiRank Label Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.close()

    elif infer_mode == "ST":
        f = open(os.path.join(save_path, "scAnndata.pkl"), "rb")
        scAnndata = pickle.load(f)
        f.close()

        scAnndata.obs["TiRank_Score"] = sc_PredDF["Rank_Score"]
        scAnndata.obs["TiRank_Label"] = sc_PredDF["Rank_Label"]

        sc.pl.umap(scAnndata, color="TiRank_Score", title="", show=False)
        plt.savefig(
            os.path.join('./assets/', "UMAP of TiRank Pred Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.close()

        sc.pl.spatial(
            scAnndata, color="TiRank_Score", title="", show=False, alpha_img=0.6
        )
        plt.savefig(
            os.path.join('./assets/', "Spatial of TiRank Pred Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.close()

        sc.pl.umap(
            scAnndata,
            color="TiRank_Label",
            title="",
            show=False,
            palette=label_color_map,
        )
        plt.savefig(
            os.path.join('./assets/', "UMAP of TiRank Label Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.close()

        sc.pl.spatial(
            scAnndata,
            color="TiRank_Label",
            title="",
            show=False,
            alpha_img=0.6,
            palette=label_color_map,
        )
        plt.savefig(
            os.path.join('./assets/', "Spatial of TiRank Label Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.show()
        plt.close()

    else:
        raise ValueError("Invalid infer_mode selected")

    return None


# Probability Score Distribution Visualization on UMAP
def plot_label_distribution_among_conditions_(save_path, group):
    sc_PredDF = pd.read_csv(
        os.path.join(save_path, "spot_predict_score.csv"), index_col=0
    )

    if group not in sc_PredDF.columns:
        raise ValueError("Invalid grouping condition selected")

    # Creating a frequency table
    freq_table = pd.crosstab(index=sc_PredDF[group], columns=sc_PredDF["Rank_Label"])
    df = freq_table.stack().reset_index(name="Freq")
    df = df[df[group] != ""]

    # Calculating cluster totals and proportions
    cluster_totals = df.groupby(group)["Freq"].sum().reset_index(name="TotalFreq")
    df = pd.merge(df, cluster_totals, on=group, how="left")
    df["Proportion"] = df["Freq"] / df["TotalFreq"]

    # For now, skipping direct entropy calculation for brevity

    # Order and adjust DataFrame for plotting
    df[group] = pd.Categorical(df[group], categories=pd.unique(df[group]), ordered=True)
    df = df.sort_values(by=[group, "Rank_Label"])

    # Plotting
    sns.set_style("white")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x=group,
        y="Proportion",
        hue="Rank_Label",
        palette={"Rank-": "#4cb1c4", "Rank+": "#b5182b", "Background": "grey"},
    )
    plt.legend(title="Rank Label")
    plt.xlabel(f"{group}")
    plt.ylabel("Proportion")
    plt.title(f"Proportion of Rank Labels by {group}")

    # Construct the filename using the 'group' variable. This ensures the file name reflects the content of the plot.
    filename = f"Distribution of TiRank label in {group}.png"
    # Save the figure, using os.path.join to construct the file path correctly.
    plt.savefig(
        os.path.join('./assets/', filename),
        bbox_inches="tight",
        pad_inches=1,
    )
    plt.show()
    plt.close()

    return None


def plot_genepair(df, data_type, save_path=None):
    """
    Plots a heatmap with hierarchical clustering applied to rows and columns.

    Parameters:
    - df : pandas.DataFrame
        DataFrame with binary values (e.g., 1 and -1).
    - method : str, optional
        The linkage algorithm to use for clustering (e.g., 'average', 'single', 'complete').
    - metric : str, optional
        The distance metric to use (e.g., 'euclidean', 'cityblock').
    - cmap : str, optional
        The colormap used to plot the heatmap. Default is 'coolwarm'.
    - figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (10, 8).

    Returns:
    - None
    """

    ## difine the figure
    figsize = (15, 12)
    cmap = "coolwarm"

    ## define cluster
    method = "average"
    metric = "euclidean"

    nrow, ncol = df.shape

    if nrow > ncol:
        n_size = ncol
        sampled_df = df.sample(n=n_size, random_state=42)
    else:
        sampled_df = df

    # Generate the linkage matrices
    row_clusters = linkage(sampled_df, method=method, metric=metric)
    col_clusters = linkage(sampled_df.T, method=method, metric=metric)

    # Create the row and column dendrogram orders
    row_dendr = dendrogram(row_clusters, no_plot=True)
    col_dendr = dendrogram(col_clusters, no_plot=True)

    # Reorder the dataframe according to the dendrograms
    df_clustered = sampled_df.iloc[row_dendr["leaves"], col_dendr["leaves"]]

    # Plotting
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_clustered, cmap=cmap, annot=False
    )  # Turn off tick labels if not meaningful
    plt.title("Clustered Heatmap of Gene Pairs")
    plt.savefig(
        os.path.join(save_path, data_type + " gene pair heatmap.png"),
        bbox_inches="tight",
        pad_inches=0.1
    )
    plt.close()

    return None
