import os
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse
from scipy import io

## Set Path 
resultPath = "/home/lenislin/Experiment/projectsResult/TiRank/bulk2st/CRC/results"

## samples
allSamples = os.listdir(resultPath)

for sample_ in allSamples:
    TiRankResultPath = os.path.join(resultPath,sample_,"3_Analysis")
    ConvertPath = os.path.join(resultPath,sample_,"4_Convert2R")

    if not os.path.exists(ConvertPath):
        os.makedirs(ConvertPath)

    ## Load data
    adata = sc.read_h5ad(os.path.join(TiRankResultPath,"final_anndata.h5ad"))

    # Metadate
    meta_ = pd.read_csv(os.path.join(TiRankResultPath,"spot_predict_score.csv"),index_col=0)
    adata.obs = meta_

    # Save the expression profile (X) as a sparse matrix in .mtx format
    matrix = scipy.sparse.csr_matrix(np.array(adata.X.todense()))

    io.mmwrite(os.path.join(ConvertPath,"expression_profile.mtx"), matrix) ## save exp
    adata.obs.to_csv(os.path.join(ConvertPath,'metadata.csv')) ## save meta

    # Save row names (var) and column names (obs) if needed
    adata.var_names.to_series().to_csv(os.path.join(ConvertPath,'row_names.csv'), header=False)
    adata.obs_names.to_series().to_csv(os.path.join(ConvertPath,'column_names.csv'), header=False)