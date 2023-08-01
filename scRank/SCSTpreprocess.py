# Preprocessing function for scRNA-seq data
import scanpy as sc
import pandas as pd

# import magic

from scipy.spatial.distance import cdist

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

    # The input is a pandas DataFrame where rows represent genes and columns represent cells.

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
    distance_df = pd.DataFrame(
        euclidean_distances, columns=ann_data.obs_names, index=ann_data.obs_names)

    return similarity_df, distance_df
