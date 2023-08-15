import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
from scipy import sparse

from anndata import AnnData
import os

import cell2location
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel

savePath = "/mnt/data/lyx/scRankv2/data/ST/deconv/"
ref_run_name = f'{savePath}reference_signatures/'
run_name = f'{savePath}visuim/'

# Cell2location deconvlution
# Load scRNA reference set
scPath = "/mnt/data/lyx/scRankv2/data/scRNAseq/CRC/"

scExp = pd.read_csv(os.path.join(
    scPath, 'GSE144735_processed_KUL3_CRC_10X_raw_UMI_count_matrix.txt'), sep="\t", index_col="Index")
scClinical = pd.read_csv(os.path.join(
    scPath, 'GSE144735_processed_KUL3_CRC_10X_annotation.txt'), sep='\t', index_col="Index")

scExp_ = scExp.T

scAnndata = AnnData(sparse.csr_matrix(scExp_.values),
                    obs=pd.DataFrame(scExp_.index),
                    var=pd.DataFrame(scExp_.columns))

scAnndata.var_names_make_unique()
scAnndata.var['SYMBOL'] = scAnndata.var_names
scAnndata.obs.index = scAnndata.obs.iloc[:,0]
scAnndata.obs = scAnndata.obs.join(
    scClinical, how='inner', lsuffix='', rsuffix='')

del scExp, scClinical, scExp_

# Preprocessing
sc.pp.filter_cells(scAnndata, min_genes=200)


# filter the object
selected = filter_genes(scAnndata, cell_count_cutoff=5,
                        cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
scAnndata = scAnndata[:, selected].copy()
scAnndata = scAnndata[~scAnndata.obs['Cell_type'].isna(), :]

# scAnndata.write_h5ad(filename=os.path.join(savePath,"GSE144735.h5ad"))
# scAnndata = sc.read_h5ad(os.path.join(savePath, "GSE144735.h5ad"))

# Estimation reference cell type signatures
cell2location.models.RegressionModel.setup_anndata(adata=scAnndata,
                                                   # 10X reaction / sample / batch
                                                   batch_key='Sample',
                                                   # cell type, covariate used for constructing signatures
                                                   labels_key='Cell_subtype',
                                                   # multiplicative technical effects (platform, 3' vs 5', donor effect)
                                                   # categorical_covariate_keys=['Method']
                                                   categorical_covariate_keys=None
                                                   )

mod = RegressionModel(scAnndata)

# view anndata_setup as a sanity check and train
mod.view_anndata_setup()

mod.train(max_epochs=200, use_gpu=True)

scAnndata = mod.export_posterior(
    scAnndata, sample_kwargs={'num_samples': 1000,
                              'batch_size': 2500, 'use_gpu': True}
)

# save
mod.save(f"{ref_run_name}", overwrite=True)

with open(os.path.join(savePath, 'scAnndata.pkl'), 'wb') as f:
    pickle.dump(scAnndata, f)
f.close()


f = open(os.path.join(savePath, 'scAnndata.pkl'), 'rb')
scAnndata = pickle.load(f)
f.close()


# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in scAnndata.varm.keys():
    inf_aver = scAnndata.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                          for i in scAnndata.uns['mod']['factor_names']]].copy()
else:
    inf_aver = scAnndata.var[[f'means_per_cluster_mu_fg_{i}'
                              for i in scAnndata.uns['mod']['factor_names']]].copy()
inf_aver.columns = scAnndata.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]


# spatial mapping
# Load

scAnndata = sc.read_h5ad(os.path.join(savePath, "GSE144735.h5ad"))
mod = cell2location.models.RegressionModel.load(f"{ref_run_name}", scAnndata)

# Preprocessing - Visum
stPath = "/mnt/data/lyx/scRankv2/data/ST/CRC/"
slices = os.listdir(stPath)
for slice_ in slices:
    stAnndata = sc.read_visium(os.path.join(stPath, slices[0]))

    # find mitochondria-encoded (MT) genes
    stAnndata.var['MT_gene'] = [gene.startswith(
        'MT-') for gene in stAnndata.var.index]

    # remove MT genes for spatial mapping (keeping their counts in the object)
    stAnndata.obsm['MT'] = stAnndata[:,
                                     stAnndata.var['MT_gene'].values].X.toarray()
    stAnndata = stAnndata[:, ~stAnndata.var['MT_gene'].values]

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(stAnndata.var_names, inf_aver.index)
    stAnndata = stAnndata[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(
        adata=stAnndata, batch_key="sample")

    # create and train the model
    mod = cell2location.models.Cell2location(
        stAnndata, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=30,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=20
    )
    mod.view_anndata_setup()

    mod.train(max_epochs=30000,
              # train using full data (batch_size=None)
              batch_size=None,
              # use all data points in training because
              # we need to estimate cell abundance at all locations
              train_size=1,
              use_gpu=True,
              )

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    stAnndata = mod.export_posterior(
        stAnndata, sample_kwargs={'num_samples': 1000,
                                  'batch_size': mod.adata.n_obs, 'use_gpu': True}
    )

    # Save model
    stAnndata.write(os.path.join(savePath, slices[0] + "_deconv.h5ad"))


#############################
# stPath = "/mnt/data/lyx/scRankv2/data/ST/CRC/"
# slices = os.listdir(stPath)

# stAnndata = sc.read_h5ad(os.path.join(
#     savePath, slices[0] + "_downstream.h5ad"))

# ## Plot Prediction and cluster label
# fig, axs = plt.subplots(2, 2, figsize=(8, 6))

# ## Rank_Score
# sc.pl.spatial(stAnndata, img_key="hires", color="Rank_Score",
#               alpha=0.5, size=1.5, ax=axs[0][0], show=False)
# axs[0][0].set_title("Rank_Score")

# ## Rank_Label
# sc.pl.spatial(stAnndata, img_key="hires", color="Rank_Label",
#               alpha=0.5, size=1.5, ax=axs[0][1], show=False)
# axs[0][1].set_title("Rank_Label")

# ## original H&E
# sc.pl.spatial(stAnndata, img_key="hires", color=None,
#               size=1.5, ax=axs[1][0], show=False)
# axs[1][0].set_title("H&E Image")

# ## clusters
# sc.pl.spatial(stAnndata, img_key="hires", color="clusters",
#               alpha=0.5, size=1.5, ax=axs[1][1], show=False)
# axs[1][1].set_title("Clusters")

# plt.tight_layout()

# ## Save the combined figure
# plt.savefig(os.path.join(savePath, f"Combined Images of {slices[0]}.pdf"))
# plt.close()
