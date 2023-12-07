# DataLoader classes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


def generate_val(bulkExp, bulkClinical, validation_proportion=0.15, mode =  None):
    # Transpose bulkExp so that samples are rows
    bulkExp_transposed = bulkExp.T

    # Concatenate bulkExp and bulkClinical
    combined = pd.concat([bulkExp_transposed, bulkClinical], axis=1)

    # Split the combined dataframe
    combined_train, combined_val = train_test_split(
        combined, 
        test_size=validation_proportion, 
        random_state=42
    )

    if mode == "Bionomial":
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

    return bulkExp_train, bulkExp_val, pd.DataFrame(bulkClinical_train), pd.DataFrame(bulkClinical_val)

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
