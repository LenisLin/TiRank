import pandas as pd
import numpy as np
import scanpy as sc
import pickle
import os

## load the expression profile from bulk
def load_bulk_exp(path_to_bulk_exp):
    if path_to_bulk_exp.lower().endswith('.csv'):
        bulkExp = pd.read_csv(path_to_bulk_exp, index_col=0)
    elif path_to_bulk_exp.lower().endswith('.txt'):
        bulkExp = pd.read_table(path_to_bulk_exp, index_col=0,sep='\t')

    return bulkExp

## load the clinical information from bulk
def load_bulk_clinical(path_to_bulk_cli):
    if path_to_bulk_cli.lower().endswith('.csv'):
        bulkClinical = pd.read_csv(path_to_bulk_cli, index_col=0)
    elif path_to_bulk_cli.lower().endswith('.txt'):
        bulkClinical = pd.read_table(path_to_bulk_cli, index_col=0,sep='\t')
    else:
        bulkClinical = pd.read_excel(path_to_bulk_cli, index_col=0) 

    return bulkClinical

## check the bulk data of bulk
def check_bulk(savePath,bulkExp,bulkClinical):
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
    savePath_1 = os.path.join(savePath,"1_loaddata")

    ## Now just only can load the files output from spaceranger
    scAnndata = sc.read_visium(path_to_st_floder)

    with open(os.path.join(savePath_1, 'anndata.pkl'), 'wb') as f:
        pickle.dump(scAnndata, f)
    f.close()

    return  scAnndata

## View column of clinical dataframe
def view_dataframe(df,nrow=10,ncol=8):
    print(df.iloc[0:nrow,0:ncol])
    return None

## Transfer sc / st expression profile
def transfer_exp_profile(scAnndata):
    if type(scAnndata.X) == type(np.array("1")):
        df = pd.DataFrame(scAnndata.X.T)
    else:
        df = pd.DataFrame(scAnndata.X.toarray().T)
    df.index = scAnndata.var_names
    df.columns = scAnndata.obs.index    

    return df
