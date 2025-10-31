import pandas as pd
import numpy as np
import scanpy as sc
import pickle
import os

"""
Data Loading Utilities for TiRank.

This module provides functions to load bulk expression data (CSV, TXT),
bulk clinical data (CSV, TXT, Excel), and single-cell/spatial
transcriptomics data (H5AD, 10x Visium folders). It also includes
helper functions to check data consistency and preview DataFrames.
"""

## load the expression profile from bulk
def load_bulk_exp(path_to_bulk_exp):
    """
    Loads a bulk expression file (CSV or TXT) into a DataFrame.

    Args:
        path_to_bulk_exp (str): The file path to the bulk expression data.
            Assumes genes are rows and samples are columns.

    Returns:
        pd.DataFrame: A pandas DataFrame of the expression data.
    """
    if path_to_bulk_exp.lower().endswith('.csv'):
        bulkExp = pd.read_csv(path_to_bulk_exp, index_col=0)
    elif path_to_bulk_exp.lower().endswith('.txt'):
        bulkExp = pd.read_table(path_to_bulk_exp, index_col=0,sep='\t')

    return bulkExp

## load the clinical information from bulk
def load_bulk_clinical(path_to_bulk_cli):
    """
    Loads a bulk clinical data file (CSV, TXT, or Excel) into a DataFrame.

    Args:
        path_to_bulk_cli (str): The file path to the bulk clinical data.
            Assumes samples are rows and clinical variables are columns.

    Returns:
        pd.DataFrame: A pandas DataFrame of the clinical data.
    """
    if path_to_bulk_cli.lower().endswith('.csv'):
        bulkClinical = pd.read_csv(path_to_bulk_cli, index_col=0)
    elif path_to_bulk_cli.lower().endswith('.txt'):
        bulkClinical = pd.read_table(path_to_bulk_cli, index_col=0,sep='\t')
    else:
        bulkClinical = pd.read_excel(path_to_bulk_cli, index_col=0) 

    return bulkClinical

## check the bulk data of bulk
def check_bulk(savePath,bulkExp,bulkClinical):
    """
    Checks and filters bulk data for common samples.

    Intersects the samples in the bulk expression columns and the bulk
    clinical index. Filters both DataFrames to keep only the common
    samples and saves the results as pickle files.

    Args:
        savePath (str): The main project directory path to save results.
        bulkExp (pd.DataFrame): Bulk expression DataFrame (genes x samples).
        bulkClinical (pd.DataFrame): Bulk clinical DataFrame (samples x variables).

    Returns:
        None
    """
    savePath_1 = os.path.join(savePath,"1_loaddata")    

    common_elements = bulkClinical.index.intersection(bulkExp.columns)
    if(len(common_elements)==0):
        print("The rownames of clinical information was not match with expression profile !")
    
    bulkClinical = bulkClinical.loc[common_elements,:]
    bulkExp = bulkExp.loc[:,common_elements]
    
    # save Data
    with open(os.path.join(savePath_1, 'bulk_exp.pkl'), 'wb') as f:
        pickle.dump(bulkExp, f)
    f.close()
    with open(os.path.join(savePath_1, 'bulk_clinical.pkl'), 'wb') as f:
        pickle.dump(bulkClinical, f)
    f.close()

    return None

## load single cell expression data
def load_sc_data(path_to_sc_h5ead, savePath):
    """
    Loads single-cell RNA-seq data from an .h5ad file.

    Args:
        path_to_sc_h5ead (str): The file path to the .h5ad file.
        savePath (str): The main project directory path to save the
            loaded AnnData object as 'anndata.pkl'.

    Returns:
        sc.AnnData: The loaded AnnData object.
    """
    savePath_1 = os.path.join(savePath,"1_loaddata")

    ## highly recommend the user to upload the h5ad file 
    ## offer a link or file to teach how to create .h5ad file
    if path_to_sc_h5ead.lower().endswith('.h5ad'):
        scAnndata = sc.read_h5ad(path_to_sc_h5ead)

        with open(os.path.join(savePath_1, 'anndata.pkl'), 'wb') as f:
            pickle.dump(scAnndata, f)
        f.close()

    return scAnndata


## load spatial transcriptomics expression data
def load_st_data(path_to_st_floder, savePath):
    """
    Loads spatial transcriptomics (ST) data from a 10x Visium folder.

    Uses scanpy.read_visium() to load the data.

    Args:
        path_to_st_floder (str): The file path to the directory containing
            the 10x Visium output (e.g., 'spatial', 'filtered_feature_bc_matrix.h5').
        savePath (str): The main project directory path to save the
            loaded AnnData object as 'anndata.pkl'.

    Returns:
        sc.AnnData: The loaded AnnData object.
    """
    savePath_1 = os.path.join(savePath,"1_loaddata")

    ## Now just only can load the files output from spaceranger
    scAnndata = sc.read_visium(path_to_st_floder)

    with open(os.path.join(savePath_1, 'anndata.pkl'), 'wb') as f:
        pickle.dump(scAnndata, f)
    f.close()

    return  scAnndata

## View column of clinical dataframe
def view_dataframe(df,nrow=10,ncol=8):
    """
    Prints a top-left subset of a DataFrame for quick viewing.

    Args:
        df (pd.DataFrame): The DataFrame to view.
        nrow (int, optional): The number of rows to show. Defaults to 10.
        ncol (int, optional): The number of columns to show. Defaults to 8.

    Returns:
        None
    """
    print(df.iloc[0:nrow,0:ncol])
    return None

## Transfer sc / st expression profile
def transfer_exp_profile(scAnndata):
    """
    Converts the .X matrix of an AnnData object to a pandas DataFrame.

    Handles both sparse and dense .X matrices. The resulting DataFrame
    is structured as genes (rows) x cells/spots (columns).

    Args:
        scAnndata (sc.AnnData): The AnnData object.

    Returns:
        pd.DataFrame: The expression matrix as a pandas DataFrame.
    """
    if type(scAnndata.X) == type(np.array("1")):
        df = pd.DataFrame(scAnndata.X.T)
    else:
        df = pd.DataFrame(scAnndata.X.toarray().T)
    df.index = scAnndata.var_names
    df.columns = scAnndata.obs.index    

    return df