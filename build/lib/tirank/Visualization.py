import torch
import os
import pickle
import math
import json

import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gseapy as gp

from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import confusion_matrix
from .Dataloader import transform_test_exp

# Data Preparation


def create_tensor(data_matrix):
    """Converts a numpy array or list-like object to a float32 PyTorch tensor.

    Args:
        data_matrix (np.ndarray or list): The input data matrix.

    Returns:
        torch.Tensor: The data converted to a float32 PyTorch tensor.
    """
    tensor = torch.from_numpy(np.array(data_matrix))
    return torch.tensor(tensor, dtype=torch.float32)


# Plot loss function
def plot_loss(train_loss_dict, savePath="./loss_on_epoch.png"):
    """Plots and saves the training loss curves over epochs.

    This function takes a dictionary of loss values recorded at each epoch,
    plots the trends for each loss type on a single graph, and saves
    the plot to a file.

    Args:
        train_loss_dict (dict): A dictionary where keys are epoch identifiers
            (e.g., 'Epoch_1') and values are dictionaries mapping loss
            names (e.g., 'total_loss') to their numerical values.
        savePath (str, optional): The file path to save the resulting
            loss plot. Defaults to "./loss_on_epoch.png".

    Returns:
        None
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
    loss_data = {
        loss_type: [epoch_data[loss_type] for epoch_data in train_loss_dict.values()]
        for loss_type in loss_types
    }

    # Plotting
    plt.figure(figsize=(10, 6))

    for loss_type, losses in loss_data.items():
        plt.plot(epochs, losses, label=loss_type)

    plt.title("Loss Value Change Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(savePath, bbox_inches="tight", pad_inches=1)
    plt.show()
    plt.clf()
    plt.close()

    return None


# Model Prediction


def model_predict(model, data_tensor, mode):
    """Generates predictions from a trained model based on the specified mode.

    Args:
        model (torch.nn.Module): The trained PyTorch model to use for prediction.
        data_tensor (torch.Tensor): The input data as a PyTorch tensor.
        mode (str): The operational mode, determining how to interpret the
            model's output. Expected values are "Cox", "Classification",
            or "Regression".

    Returns:
        tuple: A tuple containing:
            - pred_label (np.ndarray): Predicted labels. For "Classification",
              these are the class indices. For "Regression" and "Cox",
              this is the same as `pred_prob`.
            - pred_prob (np.ndarray): Predicted probability scores. For
              "Classification", this is the probability of class 1.
    """
    _, prob_scores, _ = model(data_tensor)

    if mode == "Cox":
        pred_prob = prob_scores.detach().numpy().reshape(-1, 1)

    elif mode == "Classification":
        pred_label = (
            torch.max(prob_scores, dim=1).indices.detach().numpy().reshape(-1, 1)
        )
        pred_prob = prob_scores[:, 1].detach().numpy().reshape(-1, 1)

    elif mode == "Regression":
        pred_prob = prob_scores.detach().numpy().reshape(-1, 1)
        pred_label = pred_prob

    return pred_label, pred_prob


# Probability Score Distribution Visualization
def plot_score_distribution(savePath):
    """Plots the density distribution of prediction scores for bulk and single-cell data.

    This function loads prediction dataframes for bulk and single-cell
    experiments from pickled files, plots their "Pred_score" distributions
    on a single density plot, and saves the figure.

    Args:
        savePath (str): The root directory containing the '3_Analysis'
            subfolder, which must hold 'saveDF_bulk.pkl' and
            'saveDF_sc.pkl'.

    Returns:
        None
    """
    savePath_3 = os.path.join(savePath, "3_Analysis")

    ## Load data
    f = open(os.path.join(savePath_3, "saveDF_bulk.pkl"), "rb")
    bulk_PredDF = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_3, "saveDF_sc.pkl"), "rb")
    sc_PredDF = pickle.load(f)
    f.close()

    pred_prob_sc = sc_PredDF["Pred_score"]  # scRNA
    pred_prob_bulk = bulk_PredDF["Pred_score"]  # Bulk RNA

    ## Plot
    sns.distplot(
        pred_prob_bulk,
        hist=False,
        kde=True,
        kde_kws={"shade": True, "linewidth": 3},
        label="Bulk",
    )
    sns.distplot(
        pred_prob_sc,
        hist=False,
        kde=True,
        kde_kws={"shade": True, "linewidth": 3},
        label="Single Cell",
    )

    plt.title("Density Plot")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend(title="Sample Type", loc="upper left")
    plt.savefig(
        os.path.join(savePath_3, "TiRank Pred Score Distribution.png"),
        bbox_inches="tight",
        pad_inches=1,
    )
    plt.show()
    plt.close()

    return None


# Probability Score Distribution Visualization on UMAP
def plot_score_umap(savePath, infer_mode):
    """Visualizes TiRank prediction scores and labels on UMAP and spatial plots.

    This function loads an AnnData object and corresponding prediction scores.
    It then generates and saves visualization plots based on the inference mode.
    - For "SC" (single-cell) mode, it saves UMAP plots colored by score and label.
    - For "ST" (spatial) mode, it saves UMAP and spatial plots colored by
      score and label.

    Args:
        savePath (str): The base directory containing '2_preprocessing' and
            '3_Analysis' subdirectories.
        infer_mode (str): The type of data being plotted, either "SC" or "ST".

    Returns:
        None

    Raises:
        ValueError: If `infer_mode` is not "SC" or "ST".
    """
    ## DataPath
    savePath_2 = os.path.join(savePath, "2_preprocessing")
    savePath_3 = os.path.join(savePath, "3_Analysis")

    ## Load Predict Data
    sc_PredDF = pd.read_csv(
        os.path.join(savePath_3, "spot_predict_score.csv"), index_col=0
    )

    label_color_map = {
        "Rank+": "#DE6E66",
        "Rank-": "#5096DE",
        "Background": "lightgrey",
    }

    if infer_mode == "SC":
        f = open(os.path.join(savePath_2, "scAnndata.pkl"), "rb")
        scAnndata = pickle.load(f)
        f.close()

        scAnndata.obs["TiRank_Score"] = sc_PredDF["Rank_Score"]
        scAnndata.obs["TiRank_Label"] = sc_PredDF["Rank_Label"]

        sc.pl.umap(scAnndata, color="TiRank_Score", title="", show=False)
        plt.savefig(
            os.path.join(savePath_3, "UMAP of TiRank Pred Score.png"),
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
            os.path.join(savePath_3, "UMAP of TiRank Label Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.show()
        plt.close()

    elif infer_mode == "ST":
        f = open(os.path.join(savePath_2, "scAnndata.pkl"), "rb")
        scAnndata = pickle.load(f)
        f.close()

        scAnndata.obs["TiRank_Score"] = sc_PredDF["Rank_Score"]
        scAnndata.obs["TiRank_Label"] = sc_PredDF["Rank_Label"]

        sc.pl.umap(scAnndata, color="TiRank_Score", title="", show=False)
        plt.savefig(
            os.path.join(savePath_3, "UMAP of TiRank Pred Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.show()
        plt.close()

        sc.pl.spatial(
            scAnndata, color="TiRank_Score", title="", show=False, alpha_img=0.6
        )
        plt.savefig(
            os.path.join(savePath_3, "Spatial of TiRank Pred Score.png"),
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
            os.path.join(savePath_3, "UMAP of TiRank Label Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.show()
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
            os.path.join(savePath_3, "Spatial of TiRank Label Score.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.show()
        plt.close()

    else:
        raise ValueError("Invalid infer_mode selected")

    return None


# Probability Score Distribution Visualization on UMAP
def plot_label_distribution_among_conditions(savePath, group):
    """Plots the proportional distribution of TiRank labels within different groups.

    This function loads prediction scores and calculates the frequency and
    proportion of each 'Rank_Label' ('Rank+', 'Rank-', 'Background')
    within the categories of a specified 'group' column (e.g., cell type,
    cluster). It then saves a bar plot of these proportions.

    Args:
        savePath (str): The base directory containing the '3_Analysis'
            subfolder.
        group (str): The column name in 'spot_predict_score.csv' to use
            for grouping the data.

    Returns:
        None

    Raises:
        ValueError: If the specified `group` column is not found in the
            loaded DataFrame.
    """
    ## DataPath
    savePath_2 = os.path.join(savePath, "2_preprocessing")
    savePath_3 = os.path.join(savePath, "3_Analysis")

    ## Load Predict Data
    sc_PredDF = pd.read_csv(
        os.path.join(savePath_3, "spot_predict_score.csv"), index_col=0
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
    plt.xlabel("{group}")
    plt.ylabel("Proportion")
    plt.title(f"Proportion of Rank Labels by {group}")

    # Construct the filename using the 'group' variable. This ensures the file name reflects the content of the plot.
    filename = f"Distribution of TiRank label in {group}.png"
    # Save the figure, using os.path.join to construct the file path correctly.
    plt.savefig(
        os.path.join(savePath_3, filename),
        bbox_inches="tight",
        pad_inches=1,
    )
    plt.show()
    plt.close()

    # Plot the spatial map

    return None

# Spatial Hub map (For ST only)
def plot_STmap(savePath,group):
    """Generates a composite spatial map for ST data showing cluster hubs.

    This function is for Spatial Transcriptomics (ST) data. It loads
    prediction scores, cluster-to-rank mappings from a JSON file, and the
    AnnData object. It creates a new 'new_Rank_Label' based on the hub
    classification ('Rank+', 'Rank-', 'Background') of each spot's `group`.
    It then saves a figure with three subplots:
    1. Spatial plot colored by the original `group`.
    2. The H&E image alone.
    3. Spatial plot colored by 'new_Rank_Label' overlaid on the H&E image.

    Args:
        savePath (str): The base directory containing '2_preprocessing' and
            '3_Analysis'.
        group (str): The column name used for grouping (e.g., 'cluster')
            which corresponds to the JSON file
            (f"{group}_category_dict.json").

    Returns:
        None

    Raises:
        ValueError: If the specified `group` column is not found in the
            loaded DataFrame.
    """
    ## DataPath
    savePath_2 = os.path.join(savePath, "2_preprocessing")
    savePath_3 = os.path.join(savePath, "3_Analysis")

    ## Load Predict Data
    sc_PredDF = pd.read_csv(
        os.path.join(savePath_3, "spot_predict_score.csv"), index_col=0
    )

    if group not in sc_PredDF.columns:
        raise ValueError("Invalid grouping condition selected")

    ## Load p-cluster results
    with open(os.path.join(savePath_3,f"{group}_category_dict.json"), 'r') as file:
        categories_ = json.load(file)

    ## Assign new label
    new_RankLabel = []
    cluster_label = sc_PredDF[group].tolist()
    for i in range(len(cluster_label)):
        if cluster_label[i] in categories_["Rank+"]:
            new_RankLabel.append("Rank+")
        elif cluster_label[i] in categories_["Rank-"]:
            new_RankLabel.append("Rank-")
        else:
            new_RankLabel.append("Background")
    sc_PredDF["new_Rank_Label"] = new_RankLabel
    sc_PredDF["new_Rank_Label"] = sc_PredDF["new_Rank_Label"].astype('category')

    ## Load scAnndata
    f = open(os.path.join(savePath_2, "scAnndata.pkl"), "rb")
    scAnndata = pickle.load(f)
    f.close()
    scAnndata.obs = sc_PredDF

    ## Color bar
    label_color_map = {
            "Rank+": "#DE6E66",
            "Rank-": "#5096DE",
            "Background": "lightgrey",
    }
    
    # Plot and save
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize as needed

    ## Plot 1: Category Labels Without the HE Image
    sc.pl.spatial(
    scAnndata,
    color=group,       # Your categorical column
    img_key=None,       # No background image
    alpha_img=0.0,      # No background image opacity
    #spot_size=5,       # Adjust spot size
    show=False,         # Do not display immediately
    #frameon=False,
    ax=axs[0]           # Plot on the first subplot
    )
    ## Plot 2: Only the HE Image
    sc.pl.spatial(
        scAnndata,
        img_key='hires',   # Your image key (e.g., 'hires' or 'lowres')
        color=None,         # No data overlay
        alpha_img=1.0,      # Full opacity
        spot_size=0,        # No spots plotted
        show=False,         # Do not display immediately
        ax=axs[1]           # Plot on the second subplot
    )

    ## Plot 3: HE Image with Category Labels
    sc.pl.spatial(
        scAnndata,
        color="new_Rank_Label",       # Your categorical column
        img_key='hires',   # Your image key
        alpha_img=0.25,      # Full opacity for background image
        #spot_size=5,       # Adjust spot size
        palette=label_color_map,
        show=False,         # Do not display immediately
        ax=axs[2]           # Plot on the third subplot
    )
    plt.tight_layout()
    plt.savefig(
            os.path.join(savePath_3, "Spatial of TiRank Hubs.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
    plt.show()
    plt.close()

    return None


# DEG analysis
def DEG_analysis(savePath, fc_threshold=2, Pvalue_threshold=0.05, do_p_adjust=True):
    """Performs and saves Differential Gene Expression (DEG) analysis.

    This function loads a finalized AnnData object, computes DEGs between
    'Rank+' and 'Rank-' groups using 'wilcoxon', saves all results,
    and then saves a filtered list of DEGs based on log-fold-change and
    p-value thresholds.

    Args:
        savePath (str): The base directory containing the '3_Analysis'
            subfolder, where 'final_anndata.h5ad' is located and
            results will be saved.
        fc_threshold (float, optional): The fold-change threshold for
            filtering. Defaults to 2.
        Pvalue_threshold (float, optional): The p-value threshold for
            filtering. Defaults to 0.05.
        do_p_adjust (bool, optional): If True, use adjusted p-values for
            filtering. If False, use raw p-values. Defaults to True.

    Returns:
        None
    """
    savePath_3 = os.path.join(savePath, "3_Analysis")

    ## Load final single-cell data
    adata = sc.read_h5ad(os.path.join(savePath_3, "final_anndata.h5ad"))

    ## DEG
    sc.tl.rank_genes_groups(
        adata, "Rank_Label", groups=["Rank+"], reference="Rank-", method="wilcoxon"
    )

    ## Extract dataframe
    df_DEG = pd.concat(
        [
            pd.DataFrame(adata.uns["rank_genes_groups"]["names"]),
            pd.DataFrame(adata.uns["rank_genes_groups"]["scores"]),
            pd.DataFrame(adata.uns["rank_genes_groups"]["pvals"]),
            pd.DataFrame(adata.uns["rank_genes_groups"]["pvals_adj"]),
            pd.DataFrame(adata.uns["rank_genes_groups"]["logfoldchanges"]),
        ],
        axis=1,
    )

    df_DEG.columns = ["GeneSymbol", "Scores", "Pvalue", "Pvalue_adj", "LogFoldChange"]
    df_DEG.index = df_DEG["GeneSymbol"]

    df_DEG.to_csv(os.path.join(savePath_3, "All DEGs dataframe.csv"))

    df_DEG = df_DEG[np.abs(df_DEG["LogFoldChange"]) >= math.log2(fc_threshold)]

    if do_p_adjust:
        df_DEG = df_DEG[np.abs(df_DEG["Pvalue_adj"]) <= Pvalue_threshold]
    else:
        df_DEG = df_DEG[np.abs(df_DEG["Pvalue"]) <= Pvalue_threshold]

    df_DEG = df_DEG.sort_values(by="LogFoldChange", ascending=False)
    df_DEG.to_csv(
        os.path.join(savePath_3, "Differentially expressed genes data frame.csv")
    )

    return None


# volcano plot display the differential expressed genes
def DEG_volcano(
    savePath, fc_threshold=2, Pvalue_threshold=0.05, do_p_adjust=True, top_n=5
):
    """Generates and saves a volcano plot for DEG results.

    This function loads the 'All DEGs dataframe.csv' file, creates a
    volcano plot (Log2(FoldChange) vs -Log10(P-value)), colors genes
    based on significance thresholds, and annotates the top N most
    significant up- and down-regulated genes.

    Args:
        savePath (str): The base directory containing the '3_Analysis'
            subfolder.
        fc_threshold (float, optional): Fold-change threshold for coloring
            and vertical lines. Defaults to 2.
        Pvalue_threshold (float, optional): P-value threshold for coloring
            and the horizontal line. Defaults to 0.05.
        do_p_adjust (bool, optional): If True, use adjusted p-values for
            the Y-axis and filtering. If False, use raw p-values.
            Defaults to True.
        top_n (int, optional): The number of top up- and down-regulated
            genes to annotate. Defaults to 5.

    Returns:
        None
    """
    # Path for saving analysis results
    savePath_3 = os.path.join(savePath, "3_Analysis")

    # Load data from the specified file
    result = pd.read_csv(
        os.path.join(savePath_3, "All DEGs dataframe.csv"), index_col=1
    )
    result["group"] = "black"  # Default color for all points

    log2FC = math.log2(fc_threshold)
    result["-lg10Qvalue"] = -(np.log10(result["Pvalue_adj"]))
    result["-lg10Pvalue"] = -(np.log10(result["Pvalue"]))

    # Coloring points based on thresholds and P-values
    if do_p_adjust:
        # Marking significant upregulated genes in red
        result.loc[
            (result["LogFoldChange"] >= log2FC)
            & (result["Pvalue_adj"] <= Pvalue_threshold),
            "group",
        ] = "tab:red"
        # Marking significant downregulated genes in blue
        result.loc[
            (result["LogFoldChange"] <= (-log2FC))
            & (result["Pvalue_adj"] <= Pvalue_threshold),
            "group",
        ] = "tab:blue"
        # Marking non-significant genes in grey
        result.loc[result["Pvalue_adj"] > Pvalue_threshold, "group"] = "lightgrey"
    else:
        result.loc[
            (result["LogFoldChange"] >= log2FC)
            & (result["Pvalue"] <= Pvalue_threshold),
            "group",
        ] = "tab:red"
        result.loc[
            (result["LogFoldChange"] <= (-log2FC))
            & (result["Pvalue"] <= Pvalue_threshold),
            "group",
        ] = "tab:blue"
        result.loc[result["Pvalue"] > Pvalue_threshold, "group"] = "lightgrey"

    result.loc[
        (result["LogFoldChange"] < log2FC) & (result["LogFoldChange"] > -(log2FC)),
        "group",
    ] = "lightgrey"

    # Define axis display range
    xmin, xmax, ymin, ymax = -8, 8, -10, 100

    # Create scatter plot
    fig = plt.figure(
        figsize=plt.figaspect(7 / 6)
    )  # Set figure aspect ratio (height/width)
    ax = fig.add_subplot()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), title="")
    ax.scatter(result["LogFoldChange"], result["-lg10Qvalue"], s=2, c=result["group"])

    # Annotate points
    # top N up-regulated genes
    top_up = result[
        (result["LogFoldChange"] >= log2FC) & (result["Pvalue_adj"] <= Pvalue_threshold)
    ].nlargest(top_n, "LogFoldChange")
    for index, row in top_up.iterrows():
        ax.annotate(
            row.name,
            (row["LogFoldChange"], row["-lg10Qvalue"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # top N down-regulated genes
    top_down = result[
        (result["LogFoldChange"] <= -log2FC)
        & (result["Pvalue_adj"] <= Pvalue_threshold)
    ].nsmallest(top_n, "LogFoldChange")
    for _, row in top_down.iterrows():
        ax.annotate(
            row.name,
            (row["LogFoldChange"], row["-lg10Qvalue"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax.set_ylabel("-Log10(Q value)", fontweight="bold")
    ax.set_xlabel("Log2 (fold change)", fontweight="bold")
    ax.spines["right"].set_visible(False)  # Remove right border
    ax.spines["top"].set_visible(False)  # Remove top border
    # Draw horizontal and vertical lines
    ax.vlines(
        -log2FC, ymin, ymax, color="dimgrey", linestyle="dashed", linewidth=1
    )  # Vertical line for negative log2FC
    ax.vlines(
        log2FC, ymin, ymax, color="dimgrey", linestyle="dashed", linewidth=1
    )  # Vertical line for positive log2FC
    ax.hlines(
        -math.log10(Pvalue_threshold),
        xmin,
        xmax,
        color="dimgrey",
        linestyle="dashed",
        linewidth=1,
    )  # Horizontal line for Pvalue threshold

    # Set x and y axis ticks
    ax.set_xticks(range(-8, 8, 2))  # x-axis ticks with start point and step
    ax.set_yticks(range(-10, 100, 20))  # y-axis ticks with start point and step

    # Save the figure
    fig.savefig(os.path.join(savePath_3, "DEG_volcano_plot.png"), dpi=300)
    plt.show()

    return None


# Pathway enrichment analysis
def Pathway_Enrichment(savePath, database="KEGG_2016"):
    """Performs and plots pathway enrichment analysis on DEGs.

    This function loads the filtered 'Differentially expressed genes data frame.csv',
    separates genes into up-regulated and down-regulated lists, and
    runs 'gseapy.enrichr' on the up, down, and all DEG lists using the
    specified database. It saves the enrichment tables and dot plots.

    Args:
        savePath (str): The base directory containing the '3_Analysis'
            subfolder.
        database (str or list, optional): The gene set library or libraries
            to use for enrichment (e.g., "KEGG_2016", ["GO_Biological_Process_2021"]).
            Defaults to "KEGG_2016".

    Returns:
        None
    """
    savePath_3 = os.path.join(savePath, "3_Analysis")
    result = pd.read_csv(
        os.path.join(savePath_3, "Differentially expressed genes data frame.csv"),
        index_col=1,
    )

    # up and down genes
    upgenes = result[result["LogFoldChange"] > 0]["GeneSymbol"].tolist()
    downgenes = result[result["LogFoldChange"] < 0]["GeneSymbol"].tolist()

    allgenes = upgenes.copy()
    allgenes.extend(downgenes)

    upenr = gp.enrichr(
        gene_list=upgenes,
        gene_sets=database,
        organism="Human",  # don't forget to set organism to the one you desired! e.g. Yeast
        outdir=os.path.join(savePath_3, "enrichr", "up"),
        no_plot=True,
        cutoff=0.5,  # test dataset, use lower value from range(0,1)
    )

    downenr = gp.enrichr(
        gene_list=downgenes,
        gene_sets=database,
        organism="Human",  # don't forget to set organism to the one you desired! e.g. Yeast
        outdir=os.path.join(savePath_3, "enrichr", "down"),
        no_plot=True,
        cutoff=0.5,  # test dataset, use lower value from range(0,1)
    )

    allenr = gp.enrichr(
        gene_list=allgenes,
        gene_sets=database,
        organism="Human",  # don't forget to set organism to the one you desired! e.g. Yeast
        outdir=os.path.join(savePath_3, "enrichr", "all"),
        no_plot=True,
        cutoff=0.5,  # test dataset, use lower value from range(0,1)
    )

    database_name = "_".join(database)

    ## up regulated
    if np.min(upenr.results["P-value"]) > 0.05:
        print(
            "Up regulated genes do not enrich in any pathway of " + database_name + "!"
        )
    else:
        gp.plot.dotplot(
            upenr.results,
            column="P-value",
            title="Up regulated genes enrich in " + database_name,
        )
        plt.savefig(
            os.path.join(
                savePath_3,
                "enrichr",
                "up",
                "Up regulated genes enrich in " + database_name + ".png",
            ),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.close()

    ## down regulated
    if np.min(downenr.results["P-value"]) > 0.05:
        print(
            "Down regulated genes do not enrich in any pathway of "
            + database_name
            + "!"
        )
    else:
        gp.plot.dotplot(
            downenr.results,
            column="P-value",
            title="Down regulated genes enrich in " + database_name,
        )
        plt.savefig(
            os.path.join(
                savePath_3,
                "enrichr",
                "down",
                "Down regulated genes enrich in " + database_name + ".png",
            ),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.close()

    ## all genes
    if np.min(allenr.results["P-value"]) > 0.05:
        print("All differential do not enrich in any pathway of " + database_name + "!")
    else:
        gp.plot.dotplot(
            allenr.results,
            column="P-value",
            title="All differential genes enrich in " + database_name,
        )
        plt.savefig(
            os.path.join(
                savePath_3,
                "enrichr",
                "all",
                "All differential enrich in " + database_name + ".png",
            ),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.close()

    upenr.results.to_csv(
        os.path.join(
            savePath_3,
            "enrichr",
            "up",
            "Pathway enrichment in " + database_name + " data frame.csv",
        )
    )
    downenr.results.to_csv(
        os.path.join(
            savePath_3,
            "enrichr",
            "down",
            "Pathway enrichment in " + database_name + " data frame.csv",
        )
    )
    allenr.results.to_csv(
        os.path.join(
            savePath_3,
            "enrichr",
            "all",
            "Pathway enrichment in " + database_name + " data frame.csv",
        )
    )

    return None


# Evaluation on Other Data


def evaluate_on_test_data(model, test_set, data_path, save_path, bulk_gene_pairs_mat):
    """Evaluates the model on external bulk RNA-seq test datasets.

    This function iterates through a list of test dataset IDs. For each
    dataset, it loads the expression and clinical metadata, transforms the
    expression data into the gene-pair format, predicts labels using the
    model, and saves the predictions along with a confusion matrix plot.

    Args:
        model (torch.nn.Module): The trained classification model.
        test_set (list of str): A list of dataset identifiers (e.g., ['GSE_ID1'])
            to be loaded from `data_path`.
        data_path (str): The directory containing the test data files, which
            should be named like '{data_id}_meta.csv' and '{data_id}_exp.csv'.
        save_path (str): The root directory where results will be saved. A
            'bulk_test' subdirectory will be created here.
        bulk_gene_pairs_mat (pd.DataFrame): The gene-pair matrix used as a
            template to transform the test expression data.

    Returns:
        None
    """
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)

    save_path_ = os.path.join(save_path, "bulk_test")

    if not (os.path.exists(save_path_)):
        os.makedirs(save_path_)

    for data_id in test_set:
        test_bulk_clinical = pd.read_table(
            os.path.join(data_path, data_id + "_meta.csv"), sep=",", index_col=0
        )
        test_bulk_clinical.columns = ["Group", "OS_status", "OS_time"]
        test_bulk_clinical["Group"] = test_bulk_clinical["Group"].apply(
            lambda x: 0 if x in ["PR", "CR", "CRPR"] else 1
        )

        test_bulk_exp = pd.read_csv(
            os.path.join(data_path, data_id + "_exp.csv"), index_col=0
        )
        test_bulk_exp_gene_pairs_mat = transform_test_exp(
            bulk_gene_pairs_mat, test_bulk_exp
        )
        test_exp_tensor_bulk = create_tensor(test_bulk_exp_gene_pairs_mat)

        test_pred_label, _ = model_predict(
            model, test_exp_tensor_bulk, mode="Classification"
        )
        test_bulk_clinical["TiRank_Label"] = test_pred_label.flatten()
        test_bulk_clinical.to_csv(
            os.path.join(save_path_, data_id + "_predict_score.csv")
        )

        true_labels_bulk = test_bulk_clinical["Group"]
        predicted_labels_bulk = test_bulk_clinical["TiRank_Label"]

        cm = confusion_matrix(true_labels_bulk, predicted_labels_bulk)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.savefig(
            os.path.join(savePath_, f"Pred on bulk: {data_id}.png"),
            bbox_inches="tight",
            pad_inches=1,
        )
        plt.show()
        plt.close()


# Functions for RNA-seq to scRNA with Regression mode


def create_boxplot(
    data, title, ax, group_column="True Label", score_column="Predicted Score"
):
    """Creates a boxplot on a given axis with a Mann-Whitney U test.

    Args:
        data (pd.DataFrame): DataFrame containing the plot data.
        title (str): Title for the subplot.
        ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
        group_column (str, optional): The column for the x-axis groups
            (must contain two groups, 0 and 1). Defaults to "True Label".
        score_column (str, optional): The column for the y-axis numerical
            values. Defaults to "Predicted Score".

    Returns:
        None
    """
    sns.boxplot(x=group_column, y=score_column, data=data, ax=ax)
    ax.set_title(title)

    # Statistical test
    group0 = data[data[group_column] == 0][score_column]
    group1 = data[data[group_column] == 1][score_column]
    stat, p_value = mannwhitneyu(group0, group1)
    ax.text(
        0.5,
        0.95,
        f"p = {p_value:.2e}",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


def create_density_plot(data, label, ax, title):
    """Creates a single density (KDE) plot on a given axis.

    Args:
        data (pd.Series or np.ndarray): The data to plot.
        label (str): The label for the data series in the legend.
        ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
        title (str): Title for the subplot.

    Returns:
        None
    """
    sns.kdeplot(data, shade=True, linewidth=3, label=label, ax=ax)
    ax.set_title(title)
    ax.legend()


def create_hist_plot(data, ax, title):
    """Creates a histogram with a KDE overlay on a given axis.

    Args:
        data (pd.Series or np.ndarray): The data to plot.
        ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
        title (str): Title for the subplot.

    Returns:
        None
    """
    sns.histplot(data, bins=20, kde=True, ax=ax)
    ax.set_title(title)


def create_comparison_density_plot(data1, label1, data2, label2, ax, title):
    """Creates a density plot comparing two distributions on a given axis.

    Args:
        data1 (pd.Series or np.ndarray): The first data series.
        label1 (str): Label for the first data series.
        data2 (pd.Series or np.ndarray): The second data series.
        label2 (str): Label for the second data series.
        ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
        title (str): Title for the subplot.

    Returns:
        None
    """
    sns.kdeplot(data1, shade=True, linewidth=3, label=label1, ax=ax)
    sns.kdeplot(data2, shade=True, linewidth=3, label=label2, ax=ax)
    ax.set_title(title)
    ax.legend()

def plot_genepair(df, data_type, savePath=None):
    """Plots and saves a clustered heatmap of a gene-pair matrix.

    If the input DataFrame has more rows than columns, it is sampled to be
    square. Hierarchical clustering ('average' linkage, 'euclidean' metric)
    is then applied to both rows and columns, and the resulting reordered
    DataFrame is plotted as a heatmap.

    Args:
        df (pd.DataFrame): The gene-pair DataFrame (e.g., samples vs. gene-pairs).
        data_type (str): A string identifier (e.g., "bulk", "sc") used to
            name the output file.
        savePath (str, optional): The root directory containing the
            '2_preprocessing' subfolder where the plot will be saved.
            Defaults to None.

    Returns:
        None
    """
    savePath_2 = os.path.join(savePath, "2_preprocessing")

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
        os.path.join(savePath_2, data_type + " gene pair heatmap.png"),
        bbox_inches="tight",
        pad_inches=0.1
    )
    plt.close()

    return None