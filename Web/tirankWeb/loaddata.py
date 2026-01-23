# -*- coding: utf-8 -*-
# @Date : 1.27.2024
# @Author : LingLUO

import pandas as pd
import os
import pickle
import scanpy as sc
import numpy as np


def load_bulk_exp_(path_to_bulk_exp):
    bulk_exp = pd.DataFrame()
    if path_to_bulk_exp.lower().endswith('.csv'):
        bulk_exp = pd.read_csv(path_to_bulk_exp)
    elif path_to_bulk_exp.lower().endswith('.txt'):
        bulk_exp = pd.read_table(path_to_bulk_exp, sep='\t')
    return bulk_exp


def load_bulk_clinical_(path_to_bulk_cli):
    if path_to_bulk_cli.lower().endswith('.csv'):
        bulk_clinical = pd.read_csv(path_to_bulk_cli)
    elif path_to_bulk_cli.lower().endswith('.txt'):
        bulk_clinical = pd.read_table(path_to_bulk_cli, sep='\t')
    else:
        bulk_clinical = pd.read_excel(path_to_bulk_cli)
    return bulk_clinical


def transfer_exp_profile_(scAnndata):
    if type(scAnndata.X) == type(np.array("1")):
        df = pd.DataFrame(scAnndata.X.T)
    else:
        df = pd.DataFrame(scAnndata.X.toarray().T)
    df.index = scAnndata.var_names
    df.columns = scAnndata.obs.index

    return df


def load_st_data_(path_to_st_folder):
    scAnndata = sc.read_visium(path_to_st_folder)
    with open(os.path.join('./data/', 'anndata.pkl'), 'wb') as f:
        pickle.dump(scAnndata, f)
    f.close()
    st_exp_df = transfer_exp_profile_(scAnndata)
    st_exp_df = st_exp_df.reset_index()
    return st_exp_df


def load_sc_data_(path_to_sc_folder):
    if path_to_sc_folder.lower().endswith('.h5ad'):
        scAnndata = sc.read_h5ad(path_to_sc_folder)

        with open(os.path.join('./data/', 'anndata.pkl'), 'wb') as f:
            pickle.dump(scAnndata, f)

        st_exp_df = transfer_exp_profile_(scAnndata)
        st_exp_df = st_exp_df.reset_index()
        return st_exp_df
