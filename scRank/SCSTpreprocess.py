# Preprocessing function for scRNA-seq data
import scanpy as sc
import pandas as pd
import numpy as np

# import magic

from scipy.spatial.distance import cdist

# unbalanced
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks


def is_imbalanced(bulkClinical, threshold):
    counts = bulkClinical.iloc[:, 0].value_counts(normalize=True)
    return counts.min() < threshold

def perform_sampling_on_RNAseq(bulkExp, bulkClinical, mode="SMOTE", threshold=0.5):
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

    return bulkExp_resampled, bulkClinical_resampled

# Perform standard workflow on ST


def PreprocessingST(adata):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Filtering
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)
    adata = adata[adata.obs["pct_counts_mt"] < 10]
    sc.pp.filter_genes(adata, min_cells=10)

    # Normalize
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

    # Embedding
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="clusters")

    return adata

# This function performs the MAGIC (Markov Affinity-based Graph Imputation of Cells) process on scRNA-seq data.


def perform_magic_preprocessing(input_data, require_normalization=True):

    # The input is a pandas DataFrame where rows represent genes and columns represent cells.

    # If the data needs normalization (require_normalization=True), then normalization is performed.

    if require_normalization:
        ann_data = input_data
        sc.pp.normalize_total(ann_data, target_sum=1e4)
        input_data = pd.DataFrame(ann_data.X, columns=ann_data.var_names,
                                  index=ann_data.obs_names).T

    transposed_data = input_data.T

    magic_operator = magic.MAGIC()
    magic_processed_data = magic_operator.fit_transform(
        transposed_data, genes='all_genes')

    magic_processed_data = magic_processed_data.T
    return magic_processed_data

# This function calculates the cell similarity network.


def calculate_cells_similarity(input_data, require_normalization=True):

    # The input is a anndata object where rows represent genes and columns represent cells.

    # If the data needs normalization (require_normalization=True), then normalization is performed.

    ann_data = input_data
    print(f"Gettting the cell similarity network!")

    if require_normalization:
        sc.pp.normalize_total(ann_data, target_sum=1e4)
    sc.pp.log1p(ann_data)

    # Identify highly variable genes
    sc.pp.highly_variable_genes(ann_data)

    # Perform PCA dimension reduction
    sc.tl.pca(ann_data)

    # Computing the neighborhood graph
    sc.pp.neighbors(ann_data, use_rep='X_pca')

    # Obtain the similarity matrix
    sparse_matrix = ann_data.obsp['connectivities']
    dense_matrix = sparse_matrix.toarray()
    similarity_df = pd.DataFrame(
        dense_matrix, columns=ann_data.obs_names, index=ann_data.obs_names)

    print(f"Cell similarity network done!")

    return similarity_df

# This function calculates the cell subpopulation mean rank.


def calculate_populations_meanRank(input_data, category):
    """
    Calculates the mean of features for each cell subpopulation (category)

    Args:
        input_data (DataFrame): Input dataframe where rows are different samples and columns are features.
        category (Series): A series indicating the category of each sample.

    Returns:
        DataFrame: A dataframe where rows represent categories and columns represent the mean of features for that category.
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

# This function computes the cell similarity network for spatial transcriptomics data.
# The function also calculates the Euclidean distances between spatial positions of cells.


def compute_spots_similarity(input_data, perform_normalization=True):

    # data_path refers to the output directory from the Space Ranger.
    # perform_normalization indicates whether the input data needs to be normalized.

    # Load the spatial transcriptomics data
    ann_data = input_data

    # Normalize the data if required
    if perform_normalization:
        sc.pp.normalize_total(ann_data, target_sum=1e4)
    sc.pp.log1p(ann_data)

    # Identify highly variable genes
    sc.pp.highly_variable_genes(ann_data)

    # Perform PCA dimension reduction
    sc.pp.pca(ann_data)

    # Compute the neighborhood graph
    sc.pp.neighbors(ann_data)

    # Obtain the cell-cell similarity matrix
    cell_cell_similarity = ann_data.obsp['connectivities']
    dense_similarity_matrix = cell_cell_similarity.toarray()
    similarity_df = pd.DataFrame(
        dense_similarity_matrix, columns=ann_data.obs_names, index=ann_data.obs_names)

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

    return similarity_df, distance_df
