# DataLoader classes
import numpy as np
import pandas as pd
import random, pickle, os

import torch
from torch.utils.data import Dataset, DataLoader

"""
PyTorch Dataset and DataLoader definitions for TiRank.

This module defines the custom Dataset classes for bulk, single-cell (SC),
and spatial transcriptomics (ST) data, handling the different analysis modes
(Cox, Classification, Regression). It also includes functions for splitting
data and packing it into PyTorch DataLoader objects for training and inference.
"""


def generate_val(savePath, validation_proportion=0.15, mode =  None):
    """
    Splits the bulk expression and clinical data into training and validation sets.
    
    Loads the full bulk expression and clinical data, combines them, performs a
    random split, and saves the training and validation sets back to disk in
    the '2_preprocessing/split_data' directory.

    Args:
        savePath (str): The main project directory path.
        validation_proportion (float, optional): The fraction of data to use for
            the validation set. Defaults to 0.15.
        mode (str, optional): The analysis mode ('Cox', 'Classification', 'Regression').
            This determines how many columns to use for the clinical data.
    
    Returns:
        None
    """
    savePath_1 = os.path.join(savePath,"1_loaddata")
    savePath_2 = os.path.join(savePath,"2_preprocessing")

    f = open(os.path.join(savePath_1, 'bulk_exp.pkl'), 'rb')
    bulkExp = pickle.load(f)
    f.close()
    
    f = open(os.path.join(savePath_1, 'bulk_clinical.pkl'), 'rb')
    bulkClinical = pickle.load(f)
    f.close()

    # Load data
    bulkExp, bulkClinical
    # Transpose bulkExp so that samples are rows
    bulkExp_transposed = bulkExp.T

    # Concatenate bulkExp and bulkClinical
    combined = pd.concat([bulkExp_transposed, bulkClinical], axis=1)

    # Split the combined dataframe
    random.seed(619)
    num_val = int(combined.shape[0] * validation_proportion)
    validx = random.sample(range(combined.shape[0]),num_val)

    combined_val = combined.iloc[validx,]
    mask = ~combined.index.isin(combined_val.index)
    combined_train = combined[mask]

    if mode == "Classification":
    # if mode == "Bionomial":
        # Separate the training and validation sets back into bulkExp and bulkClinical
        bulkExp_train = combined_train.iloc[:, :-1].T
        bulkClinical_train = combined_train.iloc[:, -1]

        bulkExp_val = combined_val.iloc[:, :-1].T
        bulkClinical_val = combined_val.iloc[:, -1]
    
    elif mode == "Cox":
        # Separate the training and validation sets back into bulkExp and bulkClinical
        bulkExp_train = combined_train.iloc[:, :-2].T
        bulkClinical_train = combined_train.iloc[:, -2:]

        bulkExp_val = combined_val.iloc[:, :-2].T
        bulkClinical_val = combined_val.iloc[:, -2:]

    elif mode == "Regression":
        # Separate the training and validation sets back into bulkExp and bulkClinical
        bulkExp_train = combined_train.iloc[:, :-1].T
        bulkClinical_train = combined_train.iloc[:, -1]

        bulkExp_val = combined_val.iloc[:, :-1].T
        bulkClinical_val = combined_val.iloc[:, -1]      

    ## save
    savePath_splitData = os.path.join(savePath_2,"split_data")
    if not os.path.exists(savePath_splitData):
        os.makedirs(savePath_splitData,exist_ok=True)

    with open(os.path.join(savePath_splitData, 'bulkExp_train.pkl'), 'wb') as f:
        pickle.dump(pd.DataFrame(bulkExp_train), f) ## training bulk clinical info matrix
    f.close()

    with open(os.path.join(savePath_splitData, 'bulkExp_val.pkl'), 'wb') as f:
        pickle.dump(pd.DataFrame(bulkExp_val), f) ## validating bulk clinical info matrix
    f.close()

    with open(os.path.join(savePath_splitData, 'bulkClinical_train.pkl'), 'wb') as f:
        pickle.dump(pd.DataFrame(bulkClinical_train), f) ## training bulk clinical info matrix
    f.close()

    with open(os.path.join(savePath_splitData, 'bulkClinical_val.pkl'), 'wb') as f:
        pickle.dump(pd.DataFrame(bulkClinical_val), f) ## validating bulk clinical info matrix
    f.close()

    return None

# RNA-seq


class BulkDataset(Dataset):
    """
    PyTorch Dataset class for bulk RNA-seq (gene pair) data.

    Handles different analysis modes by returning the appropriate clinical
    labels (e.g., time and event for Cox, a single label for Classification).
    
    Args:
        df_Xa (pd.DataFrame): DataFrame of gene pair features (samples x gene pairs).
        df_cli (pd.DataFrame or pd.Series): DataFrame/Series with clinical information.
        mode (str, optional): Analysis mode. One of 'Cox', 'Classification',
            or 'Regression'. Defaults to 'Cox'.
    """
    def __init__(self, df_Xa, df_cli, mode='Cox'):
        self.mode = mode

        if mode == 'Cox':
            self.Xa = torch.tensor(df_Xa.values, dtype=torch.float32)

            # Handle 'Cox' type: df_cli is expected to be a DataFrame with columns ['t', 'e']
            self.t = torch.tensor(df_cli.iloc[:,0].values, dtype=torch.float32)
            self.e = torch.tensor(df_cli.iloc[:,1].values, dtype=torch.float32)

        # elif mode == 'Bionomial':
        elif mode == 'Classification':
            self.Xa = torch.tensor(df_Xa.values, dtype=torch.float32)

            # Handle 'Bionomial' type: df_cli is expected to be a Series/1D array with group labels
            self.label = torch.tensor(df_cli.iloc[:,0].values, dtype=torch.long)

        elif mode == 'Regression':
            self.Xa = torch.tensor(df_Xa.values, dtype=torch.float32)

            # Handle 'Regression' type: df_cli is expected to be a Series/1D array with continuous values
            self.label = torch.tensor(df_cli.iloc[:,0].values, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.Xa)

    def __getitem__(self, idx):
        """
        Fetches a sample and its corresponding label(s).

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            tuple: A tuple containing the feature tensor and label(s).
                (Xa, t, e) for 'Cox' mode.
                (Xa, label) for 'Classification' or 'Regression' mode.
        """
        if self.mode == 'Cox':
            return self.Xa[idx], self.t[idx], self.e[idx]
        else:
            return self.Xa[idx], self.label[idx]

# scRNA


class SCDataset(Dataset):
    """
    PyTorch Dataset class for single-cell RNA-seq (gene pair) data.
    
    Args:
        df_Xb (pd.DataFrame or np.ndarray): DataFrame of gene pair features
            (cells x gene pairs).
    """
    def __init__(self, df_Xb):
        if type(df_Xb) is np.ndarray:
            df_Xb = pd.DataFrame(df_Xb)
        self.Xb = torch.tensor(df_Xb.values, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of cells."""
        return len(self.Xb)

    def __getitem__(self, idx):
        """
        Fetches a single cell's feature vector and its index.

        Args:
            idx (int): The index of the cell to fetch.

        Returns:
            tuple: (Xb, idx)
        """

        return self.Xb[idx], idx

# ST


class STDataset(Dataset):
    """
    PyTorch Dataset class for Spatial Transcriptomics (gene pair) data.
    
    Args:
        df_Xc (pd.DataFrame or np.ndarray): DataFrame of gene pair features
            (spots x gene pairs).
    """
    def __init__(self, df_Xc):
        self.Xc = torch.tensor(df_Xc.values, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of spots."""
        return len(self.Xc)

    def __getitem__(self, idx):
        """
        Fetches a single spot's feature vector and its index.

        Args:
            idx (int): The index of the spot to fetch.

        Returns:
            tuple: (Xc, idx)
        """

        return self.Xc[idx], idx
    

# Pack data
def PackData(savePath, mode, infer_mode, batch_size = 1024):
    """
    Loads all preprocessed data and packages it into PyTorch DataLoaders.
    
    This function reads the training/validation gene pair matrices, clinical data,
    AnnData object, similarity matrix, and pathological labels from disk. It
    instantiates the Dataset classes (BulkDataset, STDataset, SCDataset) and
    wraps them in DataLoader objects. It also prepares the adjacency matrix (adj_A)
    and pathological labels (patholabels) for the model.
    
    All resulting DataLoader objects and supporting data are saved to the
    '3_Analysis/data2train' directory.

    Args:
        savePath (str): The main project directory path.
        mode (str): The analysis mode ('Cox', 'Classification', 'Regression').
        infer_mode (str): The inference data type ('ST' or 'SC').
        batch_size (int, optional): Batch size for the DataLoaders. Defaults to 1024.
    
    Returns:
        None
    """
    savePath_2 = os.path.join(savePath,"2_preprocessing")
    savePath_3 = os.path.join(savePath,"3_Analysis")
    savePath_splitData = os.path.join(savePath_2,"split_data")

    f = open(os.path.join(savePath_2, 'train_bulk_gene_pairs_mat.pkl'), 'rb')
    train_bulk_gene_pairs_mat = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_2, 'val_bulkExp_gene_pairs_mat.pkl'), 'rb')
    val_bulkExp_gene_pairs_mat = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_splitData, 'bulkClinical_train.pkl'), 'rb')
    bulkClinical_train = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_splitData, 'bulkClinical_val.pkl'), 'rb')
    bulkClinical_val = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_2, 'scAnndata.pkl'), 'rb')
    scAnndata = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_2, 'sc_gene_pairs_mat.pkl'), 'rb')
    sc_gene_pairs_mat = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_2, 'similarity_df.pkl'), 'rb')
    similarity_df = pickle.load(f)
    f.close() 

    train_dataset_Bulk = BulkDataset(train_bulk_gene_pairs_mat, bulkClinical_train, mode=mode)
    val_dataset_Bulk = BulkDataset(val_bulkExp_gene_pairs_mat, bulkClinical_val, mode=mode)
    train_loader_Bulk = DataLoader(train_dataset_Bulk, batch_size=batch_size, shuffle=False)
    val_loader_Bulk = DataLoader(val_dataset_Bulk, batch_size=batch_size, shuffle=False)
    
    if infer_mode == "ST":
    # if infer_mode == "Spot":
        adj_A = torch.from_numpy(similarity_df.values)
        adj_B = None
        patholabels = scAnndata.obs["patho_class"]

        train_dataset_SC = STDataset(sc_gene_pairs_mat)
        train_loader_SC = DataLoader(train_dataset_SC, batch_size=batch_size, shuffle=True)

    elif infer_mode == "SC":
    # elif infer_mode == "Cell":
        adj_A = torch.from_numpy(similarity_df.values)
        adj_B = None
        patholabels = None

        train_dataset_SC = STDataset(sc_gene_pairs_mat)
        train_loader_SC = DataLoader(train_dataset_SC, batch_size=batch_size, shuffle=True)
    
    else:
        raise TypeError("Unexpected infer mode !")
    
    savePath_data2train = os.path.join(savePath_3,"data2train")
    if not os.path.exists(savePath_data2train):
        os.makedirs(savePath_data2train,exist_ok=True)

    with open(os.path.join(savePath_data2train, 'train_loader_Bulk.pkl'), 'wb') as f:
        pickle.dump(train_loader_Bulk, f)
    f.close()
    with open(os.path.join(savePath_data2train, 'val_loader_Bulk.pkl'), 'wb') as f:
        pickle.dump(val_loader_Bulk, f)
    f.close()
    with open(os.path.join(savePath_data2train, 'train_loader_SC.pkl'), 'wb') as f:
        pickle.dump(train_loader_SC, f)
    f.close()

    with open(os.path.join(savePath_data2train, 'adj_A.pkl'), 'wb') as f:
        pickle.dump(adj_A, f)
    f.close()
    with open(os.path.join(savePath_data2train, 'adj_B.pkl'), 'wb') as f:
        pickle.dump(adj_B, f)
    f.close()
    with open(os.path.join(savePath_data2train, 'patholabels.pkl'), 'wb') as f:
        pickle.dump(patholabels, f)
    f.close()

    return None

# Extract GP on other datasets

def transform_test_exp(train_exp, test_exp):
    """
    Transforms a test expression matrix into a gene pair matrix using pairs from a training set.

    Given a gene pair matrix from training (columns are 'GeneA__GeneB') and a new
    expression matrix (genes as rows), this function computes the gene pair
    values for the new data, matching the pairs from training.

    Args:
        train_exp (pd.DataFrame): The gene pair matrix from the training set.
            Its columns define the gene pairs to be used.
        test_exp (pd.DataFrame): The raw gene expression matrix for the test set
            (genes as rows, samples as columns).

    Returns:
        pd.DataFrame: A new gene pair matrix (samples x gene pairs) for the
            test set, with the same columns as `train_exp`.
    """
    # Initialize a new DataFrame to store the transformed test data
    transformed_test_exp = pd.DataFrame(index=test_exp.columns)

    # Iterate over the columns in the train_exp DataFrame
    for column in train_exp.columns:
        # Parse the column name to get the two gene names
        geneA, geneB = column.split('__')

        # Check if both genes are present in the test_exp
        if geneA in test_exp.index and geneB in test_exp.index:
            # Perform the comparison for each sample in test_exp and assign the values to the new DataFrame
            transformed_test_exp[column] = (
                test_exp.loc[geneA] > test_exp.loc[geneB]).astype(int) * 2 - 1

            # Handle cases where geneA == geneB by assigning 0
            transformed_test_exp.loc[:, column][test_exp.loc[geneA]
                                                == test_exp.loc[geneB]] = 0
        else:
            # If one or both genes are not present, assign 0 for all samples
            transformed_test_exp[column] = 0

    # Transpose the DataFrame to match the structure of train_exp
    # transformed_test_exp = transformed_test_exp.transpose()

    return transformed_test_exp