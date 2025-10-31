# Preprocessing function for scRNA-seq data
import scanpy as sc
import pandas as pd
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.stats import zscore

# unbalanced
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

"""
Preprocessing utilities for scRNA-seq and ST data using Scanpy.

This module provides a pipeline of functions to load, filter, normalize,
log-transform, and cluster single-cell and spatial data. It also includes
functions for handling class imbalance in bulk training data (sampling) and
for computing similarity/distance matrices.
"""

def merge_datasets(bulkClinical_1, bulkClinical_2, bulkExp_1, bulkExp_2):
    """
    Merges two bulk expression and clinical datasets, finding intersecting genes.

    Args:
        bulkClinical_1 (pd.DataFrame): Clinical data for the first cohort.
        bulkClinical_2 (pd.DataFrame): Clinical data for the second cohort.
        bulkExp_1 (pd.DataFrame): Expression data for the first cohort (genes x samples).
        bulkExp_2 (pd.DataFrame): Expression data for the second cohort (genes x samples).

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The merged expression DataFrame.
            - pd.DataFrame: The merged clinical DataFrame.
        Or returns 0 if no intersecting genes are found.
    """

    genes1 = {x for x in bulkExp_1.index.values}
    genes2 = {x for x in bulkExp_2.index.values}
    intersectGenes = genes1.intersection(genes2)

    if len(intersectGenes) == 0:
        print(f"The length of interaction genes between these two bulk RNA-seq datasets was zero!")
        return 0

    intersectGenes_list = [x for x in intersectGenes]

    bulkExp_1 = bulkExp_1.loc[intersectGenes_list, :]
    bulkExp_2 = bulkExp_2.loc[intersectGenes_list, :]

    bulkClinical = np.vstack((bulkClinical_1, bulkClinical_2))
    bulkExp = np.hstack((bulkExp_1, bulkExp_2))

    pid1 = [x for x in bulkExp_1.columns]
    pid2 = [y for y in bulkExp_2.columns]

    pid1.extend(pid2)

    pd.DataFrame(bulkExp)
    bulkClinical = pd.DataFrame(
        bulkClinical, columns=bulkClinical_1.columns, index=pid1)
    bulkExp = pd.DataFrame(bulkExp, columns=pid1, index=bulkExp_1.index)

    return bulkExp, bulkClinical


def normalize_data(exp):
    """
    Normalize gene expression data using z-score normalization (row-wise).

    Args:
        exp (pd.DataFrame): A pandas DataFrame with genes as rows and samples as columns.

    Returns:
        pd.DataFrame: A z-score normalized DataFrame.
    """
    # Apply z-score normalization
    normalized_exp = exp.apply(zscore, axis=1)
    normalized_exp = normalized_exp.dropna()

    return normalized_exp


def is_imbalanced(bulkClinical, threshold):
    """
    Checks if the primary clinical variable is imbalanced.

    Args:
        bulkClinical (pd.DataFrame): DataFrame with clinical data. Assumes
            the variable of interest is in the first column.
        threshold (float): The minimum proportion for a class to be
            considered 'balanced'.

    Returns:
        bool: True if the minority class is below the threshold, False otherwise.
    """
    counts = bulkClinical.iloc[:, 0].value_counts(normalize=True)
    return counts.min() < threshold


def perform_sampling_on_RNAseq(savePath, mode="SMOTE", threshold=0.5):
    """
    Performs sampling (over- or under-sampling) on the bulk training data.

    This function is used to correct for class imbalance in 'Classification' mode.
    It loads the training data, applies the specified sampling method, and
    overwrites the training files with the resampled data.

    Args:
        savePath (str): The main project directory path.
        mode (str, optional): The sampling method to use. One of 'SMOTE',
            'downsample' (RandomUnderSampler), 'upsample' (RandomOverSampler),
            or 'tomeklinks' (TomekLinks). Defaults to "SMOTE".
        threshold (float, optional): The imbalance threshold. Sampling is only
            performed if the minority class proportion is below this value.
            Defaults to 0.5.
    
    Returns:
        None
    """
    savePath_2 = os.path.join(savePath,"2_preprocessing")
    savePath_splitData = os.path.join(savePath_2,"split_data")

    ## load
    f = open(os.path.join(savePath_splitData, 'bulkExp_train.pkl'), 'rb')
    bulkExp = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_splitData, 'bulkClinical_train.pkl'), 'rb')
    bulkClinical = pickle.load(f)
    f.close()
    
    # Ensure classes are imbalanced before any action
    if not is_imbalanced(bulkClinical, threshold):
        print("Classes are balanced!")
        return bulkExp, bulkClinical

    X = bulkExp.T.values
    y = bulkClinical.values.ravel()

    if mode == "downsample":
        sampler = RandomUnderSampler(random_state=42)
    elif mode == "upsample":
        sampler = RandomOverSampler(random_state=42)
    elif mode == "SMOTE":
        sampler = SMOTE(random_state=42)
    elif mode == "tomeklinks":
        sampler = TomekLinks()
    else:
        raise ValueError("Invalid mode selected")

    X_res, y_res = sampler.fit_resample(X, y)

    # Convert back to DataFrame, making sure the samples are consistent with their labels.
    samples_order = [f"sample_{i}" for i in range(X_res.shape[0])]
    bulkExp_resampled = pd.DataFrame(
        X_res.T, columns=samples_order, index=bulkExp.index)
    bulkClinical_resampled = pd.DataFrame(
        y_res, index=samples_order, columns=bulkClinical.columns)
    
    ## save
    with open(os.path.join(savePath_splitData, 'bulkExp_train.pkl'), 'wb') as f:
        pickle.dump(pd.DataFrame(bulkExp_resampled), f) ## training bulk clinical info matrix
    f.close()
    with open(os.path.join(savePath_splitData, 'bulkClinical_train.pkl'), 'wb') as f:
        pickle.dump(pd.DataFrame(bulkClinical_resampled), f) ## training bulk clinical info matrix
    f.close()

    return None

# Perform standard workflow on ST or SC

# Filtering cells or spots
def FilteringAnndata(adata, max_count=35000, min_count=5000, MT_propor=10, min_cell=10, imgPath="./"):
    """
    Filters an AnnData object based on QC metrics.

    Filters cells/spots based on total counts and mitochondrial percentage.
    Filters genes based on minimum cell count. Also saves a QC violin plot.

    Args:
        adata (sc.AnnData): The AnnData object to filter.
        max_count (int, optional): Maximum total counts per cell/spot.
            Defaults to 35000.
        min_count (int, optional): Minimum total counts per cell/spot.
            Defaults to 5000.
        MT_propor (int, optional): Maximum percentage of mitochondrial gene
            counts. Defaults to 10.
        min_cell (int, optional): Minimum number of cells/spots a gene must
            be expressed in. Defaults to 10.
        imgPath (str, optional): Path to save the QC violin plot.
            Defaults to "./".

    Returns:
        sc.AnnData: The filtered AnnData object.
    """
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Plot
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],jitter=0.4, multi_panel=True,show=False)
    plt.savefig(os.path.join(imgPath,"qc_violins.png"))
    plt.close()

    # Filtering
    sc.pp.filter_cells(adata, min_counts=min_count)
    sc.pp.filter_cells(adata, max_counts=max_count)
    adata = adata[adata.obs["pct_counts_mt"] < MT_propor]
    sc.pp.filter_genes(adata, min_cells=min_cell)

    return adata


# Normalization
def Normalization(adata):
    """
    Performs total count normalization (target_sum=1e4) on an AnnData object.

    Args:
        adata (sc.AnnData): The AnnData object.

    Returns:
        sc.AnnData: The normalized AnnData object.
    """
    sc.pp.normalize_total(adata, target_sum=1e4, inplace = True)
    return adata

# log-transformation
def Logtransformation(adata):
    """
    Performs log1p transformation on an AnnData object.

    Args:
        adata (sc.AnnData): The AnnData object.

    Returns:
        sc.AnnData: The log-transformed AnnData object.
    """
    sc.pp.log1p(adata)
    return adata

def Clustering(ann_data,infer_mode, savePath):
    """
    Performs standard clustering (HVGs, PCA, neighbors, UMAP, Leiden).

    If neighbors are already computed, it just re-runs Leiden. Otherwise,
    it runs the full pipeline. Saves a UMAP or spatial plot.

    Args:
        ann_data (sc.AnnData): The AnnData object.
        infer_mode (str): The inference data type ('SC' or 'ST') for plotting.
        savePath (str): The main project directory path to save plots.

    Returns:
        sc.AnnData: The clustered AnnData object.
    """
    savePath_2 = os.path.join(savePath,"2_preprocessing")
    if ('connectivities' in ann_data.obsp) and ('leiden' in ann_data.uns):
        sc.tl.leiden(ann_data, key_added="leiden_clusters")

    else:
        # Identify highly variable genes
        sc.pp.highly_variable_genes(ann_data, flavor="seurat", n_top_genes=2000)
        # Perform PCA dimension reduction
        sc.tl.pca(ann_data)
        # Computing the neighborhood graph
        sc.pp.neighbors(ann_data, use_rep='X_pca')
        # UMAP and leiden
        sc.tl.umap(ann_data)
        sc.tl.leiden(ann_data, key_added="leiden_clusters")

    if infer_mode == "SC":
        sc.pl.umap(ann_data, color=['leiden_clusters'],show = False)
        plt.savefig(os.path.join(savePath_2,"leiden cluster.png"))

    if infer_mode == "ST":
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # Create a 1x2 grid for the plots

        ann_data.obsm["spatial"] = np.array(ann_data.obsm["spatial"],dtype = float)
        
        sc.pl.spatial(ann_data, img_key="hires", color=["leiden_clusters"],show = False,ax=axs[0])
        sc.pl.umap(ann_data, color=["leiden_clusters"],show = False, ax=axs[1])
        plt.tight_layout()  # Ensure proper spacing between the two plots
        plt.savefig(os.path.join(savePath_2,"leiden cluster.png"))
        plt.close()
        
    return ann_data

# This function computes the cell similarity network for single-cell or spatial transcriptomics data.
def compute_similarity(savePath, ann_data, calculate_distance=False):
    """
    Extracts and saves the cell/spot similarity matrix (connectivities).

    Optionally, it can also calculate a spatial distance-based adjacency
    matrix (6 nearest neighbors) for ST data.

    Args:
        savePath (str): The main project directory path.
        ann_data (sc.AnnData): A clustered AnnData object (must have
            `ann_data.obsp['connectivities']`).
        calculate_distance (bool, optional): Whether to compute the spatial
            distance matrix (ST only). Defaults to False.
    
    Returns:
        None
    """
    savePath_2 = os.path.join(savePath,"2_preprocessing")

    # data_path refers to the output directory from the Space Ranger.
    # perform_normalization indicates whether the input data needs to be normalized.

    # Obtain the cell-cell similarity matrix
    cell_cell_similarity = ann_data.obsp['connectivities']
    dense_similarity_matrix = cell_cell_similarity.toarray()
    similarity_df = pd.DataFrame(
        dense_similarity_matrix, columns=ann_data.obs_names, index=ann_data.obs_names)

    if calculate_distance:
        # Obtain the spatial positions and calculate the Euclidean distances
        spatial_positions = ann_data.obsm['spatial']
        euclidean_distances = cdist(
            spatial_positions, spatial_positions, metric='euclidean')

    # Create an adjacency matrix initialized with zeros
        adjacency_matrix = np.zeros_like(euclidean_distances, dtype=int)

        # For each spot, mark the six closest spots as neighbors
        for i in range(adjacency_matrix.shape[0]):
            # Get the indices of the six smallest distances
            # Skip the 0th index because it's the distance to itself
            closest_indices = euclidean_distances[i].argsort()[1:7]
            adjacency_matrix[i, closest_indices] = 1

        distance_df = pd.DataFrame(
            adjacency_matrix, columns=ann_data.obs_names, index=ann_data.obs_names)

        with open(os.path.join(savePath_2, 'distance_df.pkl'), 'wb') as f:
            pickle.dump(distance_df, f)
        f.close()

    with open(os.path.join(savePath_2, 'similarity_df.pkl'), 'wb') as f:
        pickle.dump(similarity_df, f)
    f.close()

    return None


# This function calculates the cell subpopulation mean rank.


def calculate_populations_meanRank(input_data, category):
    """
    Calculates the mean feature values for each cell subpopulation (category).

    Args:
        input_data (pd.DataFrame): Input DataFrame (samples x features).
        category (pd.Series): A Series indicating the category (e.g., cluster)
            of each sample. Must share the same index as `input_data`.

    Returns:
        pd.DataFrame: A DataFrame where rows are categories and columns
            are the mean of features for that category.
    """

    # First, ensure the category Series has the same index as the input_data DataFrame
    category.index = input_data.index

    # Combine the category series with input dataframe
    input_data_combined = pd.concat(
        [input_data, category.rename('Category')], axis=1)

    # Now group by the 'Category' column and find the mean of each group
    meanrank_df = input_data_combined.groupby('Category').mean()

    print(f"Cell subpopulation mean feature calculation done!")

    return meanrank_df