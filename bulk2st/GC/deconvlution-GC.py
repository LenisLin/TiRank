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

savePath = "/home/lenislin/Experiment/data/scRankv2/data/ST/GC_24/deconv/"
if not os.path.exists(savePath):
    os.mkdir(savePath)
ref_run_name = f'{savePath}reference_signatures/'
run_name = f'{savePath}visuim/'

# Cell2location deconvlution
# # Load scRNA reference set
# scPath = "/home/lenislin/Experiment/data/scRankv2/data/scRNAseq/GC/"

# scExp = pd.read_csv(os.path.join(scPath,"GSE183904_subset_Rawexp.csv"), index_col=0)
# scExp_ = scExp.T
# meta = pd.read_csv(os.path.join(scPath,"GSE183904_subset_anno.csv"), index_col=0)

# obs_df = pd.DataFrame({"CellID":scExp_.index})
# var_df = pd.DataFrame({"SYMBOL":scExp_.columns})

# scAnndata = AnnData(sparse.csr_matrix(scExp_.values),
#                     obs=obs_df,
#                     var=var_df)

# scAnndata.var_names_make_unique()
# scAnndata.var_names = scAnndata.var['SYMBOL']
# scAnndata.obs.index = scAnndata.obs.iloc[:,0]
# scAnndata.obs = scAnndata.obs.join(meta, how='inner', lsuffix='', rsuffix='')

# del scExp, meta, scExp_

# # Preprocessing
# sc.pp.filter_cells(scAnndata, min_genes=200)

# # filter the object
# selected = filter_genes(scAnndata, cell_count_cutoff=5,
#                         cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
# scAnndata = scAnndata[:, selected].copy()
# scAnndata = scAnndata[~scAnndata.obs['annotation'].isna(), :]

# scAnndata.write_h5ad(filename=os.path.join(savePath,"GSE183904.h5ad"))
# scAnndata = sc.read_h5ad(os.path.join(savePath, "GSE183904.h5ad"))

# Estimation reference cell type signatures
# cell2location.models.RegressionModel.setup_anndata(adata=scAnndata,
#                                                    # 10X reaction / sample / batch
#                                                    batch_key='Sample',
#                                                    # cell type, covariate used for constructing signatures
#                                                    labels_key='annotation',
#                                                    # multiplicative technical effects (platform, 3' vs 5', donor effect)
#                                                    # categorical_covariate_keys=['Method']
#                                                    categorical_covariate_keys=None
#                                                    )

# mod = RegressionModel(scAnndata)

# # view anndata_setup as a sanity check and train
# mod.view_anndata_setup()

# mod.train(max_epochs=200, use_gpu=True)

# scAnndata = mod.export_posterior(
#     scAnndata, sample_kwargs={'num_samples': 1000,
#                               'batch_size': 2500, 'use_gpu': True}
# )

# # save
# mod.save(f"{ref_run_name}", overwrite=True)

# with open(os.path.join(ref_run_name, 'scAnndata.pkl'), 'wb') as f:
#     pickle.dump(scAnndata, f)
# f.close()


f = open(os.path.join(ref_run_name, 'scAnndata.pkl'), 'rb')
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
# inf_aver.index = scAnndata.var.index[inf_aver.index]

# spatial mapping
# Load

# scAnndata = sc.read_h5ad(os.path.join(ref_run_name, "GSE183904.h5ad"))
mod = cell2location.models.RegressionModel.load(f"{ref_run_name}", scAnndata)

# Preprocessing - Visum
stPath = "/home/lenislin/Experiment/data/scRankv2/data/ST/GC_24/"
slices = os.listdir(stPath)
for slice_ in slices:
    stAnndata = sc.read_visium(os.path.join(stPath, slice_))

    # find mitochondria-encoded (MT) genes
    stAnndata.var['MT_gene'] = [gene.startswith(
        'MT-') for gene in stAnndata.var.index]

    # remove MT genes for spatial mapping (keeping their counts in the object)
    stAnndata.obsm['MT'] = stAnndata[:,
                                     stAnndata.var['MT_gene'].values].X.toarray()
    stAnndata = stAnndata[:, ~stAnndata.var['MT_gene'].values]
    stAnndata.var_names_make_unique()

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(stAnndata.var_names, inf_aver.index)
    stAnndata = stAnndata[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(
        adata=stAnndata, batch_key=None)

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

    mod.train(max_epochs=10000,
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
    stAnndata.write(os.path.join(run_name, slice_ + "_deconv.h5ad"))

# Save the results into csv
deconv_results_path = os.path.join(savePath,"visuim")
files = os.listdir(deconv_results_path)
for file_ in files:
    filename = file_.split("_deconv")[0]
    savepath = os.path.join("/home/lenislin/Experiment/projects/scRankv2/bulk2st/tempfiles_GC",filename)
    stAnndata = sc.read_h5ad(os.path.join(deconv_results_path,file_))
    colname = stAnndata.obsm["q05_cell_abundance_w_sf"].columns
    colname = [x.split("sf_")[-1] for x in colname]

    stAnndata.obsm["q05_cell_abundance_w_sf"].columns = colname
    stAnndata.obsm["q05_cell_abundance_w_sf"].to_csv(os.path.join(savepath,"q05_cell_abundance_w_sf.csv"))