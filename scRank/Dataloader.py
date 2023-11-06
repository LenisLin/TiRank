# DataLoader classes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


def generate_val(bulk_gene_pairs_mat, bulkClinical, mode, need_val = True, validation_proportion = 0.15):
    """
    Splits the bulk_gene_pairs_mat and bulkClinical datasets into training and validation sets.

    Parameters:
    - bulk_gene_pairs_mat: Your bulk gene pairs matrix.
    - bulkClinical: Your bulk clinical data.
    - mode: Mode to pass to the BulkDataset constructor.
    - validation_proportion: The proportion of the dataset to be used as validation set.

    Returns:
    - A tuple containing:
        - training set as a BulkDataset instance
        - validation set as a BulkDataset instance
    """
    
    if not need_val:
        train_dataset = BulkDataset(bulk_gene_pairs_mat, bulkClinical, mode=mode)

    # Split the gene pairs matrix and clinical data into training and validation sets
    bulk_gene_pairs_train, bulk_gene_pairs_val, bulkClinical_train, bulkClinical_val = train_test_split(
        bulk_gene_pairs_mat, 
        bulkClinical, 
        test_size=validation_proportion, 
        random_state=42  # Ensures reproducibility of the split
    )

    # Create the BulkDataset instances for training and validation
    train_dataset = BulkDataset(bulk_gene_pairs_train, bulkClinical_train, mode=mode)
    validation_dataset = BulkDataset(bulk_gene_pairs_val, bulkClinical_val, mode=mode)

    return train_dataset, validation_dataset

# RNA-seq


class BulkDataset(Dataset):
    def __init__(self, df_Xa, df_cli, mode='Cox'):
        self.mode = mode

        if mode == 'Cox':
            self.Xa = torch.tensor(df_Xa.values, dtype=torch.float32)

            # Handle 'Cox' type: df_cli is expected to be a DataFrame with columns ['t', 'e']
            self.t = torch.tensor(df_cli.iloc[:,0].values, dtype=torch.float32)
            self.e = torch.tensor(df_cli.iloc[:,1].values, dtype=torch.float32)

        elif mode == 'Bionomial':
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
        return len(self.Xa)

    def __getitem__(self, idx):
        if self.mode == 'Cox':
            return self.Xa[idx], self.t[idx], self.e[idx]
        else:
            return self.Xa[idx], self.label[idx]

# scRNA


class SCDataset(Dataset):
    def __init__(self, df_Xb):
        if type(df_Xb) is np.ndarray:
            df_Xb = pd.DataFrame(df_Xb)
        self.Xb = torch.tensor(df_Xb.values, dtype=torch.float32)

    def __len__(self):
        return len(self.Xb)

    def __getitem__(self, idx):

        return self.Xb[idx], idx

# ST


class STDataset(Dataset):
    def __init__(self, df_Xc):
        self.Xc = torch.tensor(df_Xc.values, dtype=torch.float32)

    def __len__(self):
        return len(self.Xc)

    def __getitem__(self, idx):

        return self.Xc[idx], idx
