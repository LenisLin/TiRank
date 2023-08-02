# DataLoader classes
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

# RNA-seq


class BulkDataset(Dataset):
    def __init__(self, df_Xa, df_cli, mode='Cox', expand_times = 1):
        self.mode = mode
        self.Xa = torch.tensor(df_Xa.values, dtype=torch.float32).repeat(expand_times, 1)

        if mode == 'Cox':
            # Handle 'Cox' type: df_cli is expected to be a DataFrame with columns ['t', 'e']
            self.t = torch.tensor(df_cli.iloc[:,0].values, dtype=torch.float32).repeat(expand_times)
            self.e = torch.tensor(df_cli.iloc[:,1].values, dtype=torch.float32).repeat(expand_times)
        elif mode == 'Bionomial':
            # Handle 'Bionomial' type: df_cli is expected to be a Series/1D array with group labels
            self.label = torch.tensor(df_cli.iloc[:,0].values, dtype=torch.long).repeat(expand_times)
        elif mode == 'Regression':
            # Handle 'Regression' type: df_cli is expected to be a Series/1D array with continuous values
            self.label = torch.tensor(df_cli.iloc[:,0].values, dtype=torch.float32).repeat(expand_times)
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
