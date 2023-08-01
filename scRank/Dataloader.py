# DataLoader classes
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

# RNA-seq


class BulkDataset(Dataset):
    def __init__(self, df_Xa, df_t, df_e):
        self.Xa = torch.tensor(df_Xa.values, dtype=torch.float32)
        self.t = torch.tensor(df_t.values, dtype=torch.float32)
        self.e = torch.tensor(df_e.values, dtype=torch.float32)

    def __len__(self):
        return len(self.Xa)

    def __getitem__(self, idx):
        return self.Xa[idx], self.t[idx], self.e[idx]

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
