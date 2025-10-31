# Preprocessing function for scRNA-seq data
import scanpy as sc
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def clustering_(ann_data, infer_mode, savePath):
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
        sc.pl.umap(ann_data, color=['leiden_clusters'], show=False)
        plt.savefig(os.path.join(savePath, "leiden cluster.png"))
        plt.savefig("./assets/leiden cluster.png")

    if infer_mode == "ST":
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # Create a 1x2 grid for the plots
        sc.pl.spatial(ann_data, img_key="hires", color=["leiden_clusters"], show=False, ax=axs[0])
        sc.pl.umap(ann_data, color=["leiden_clusters"], show=False, ax=axs[1])
        plt.tight_layout()  # Ensure proper spacing between the two plots
        plt.savefig(os.path.join(savePath, "leiden cluster.png"))
        plt.savefig("./assets/leiden cluster.png")
        plt.close()

    return ann_data


# This function computes the cell similarity network for single-cell or spatial transcriptomics data.
def compute_similarity_(savePath, ann_data, calculate_distance=False):
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

        with open(os.path.join(savePath, 'distance_df.pkl'), 'wb') as f:
            pickle.dump(distance_df, f)
        f.close()

    with open(os.path.join(savePath, 'similarity_df.pkl'), 'wb') as f:
        pickle.dump(similarity_df, f)
    f.close()

    return None


def FilteringAnndata_(adata, max_count=35000, min_count=5000, MT_propor=10, min_cell=10, imgPath="./"):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Plot
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True,
                 show=False)
    plt.savefig(os.path.join(imgPath, "qc_violins.png"))
    plt.savefig("./assets/qc_violins.png")
    plt.close()

    # Filtering
    sc.pp.filter_cells(adata, min_counts=min_count)
    sc.pp.filter_cells(adata, max_counts=max_count)
    adata = adata[adata.obs["pct_counts_mt"] < MT_propor]
    sc.pp.filter_genes(adata, min_cells=min_cell)

    return adata
