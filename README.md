```

savePath = "./web_test"
# 2.3 Preprocessing on sc/st data
infer_mode = "Cell"  ## optional parameter

scAnndata = FilteringAnndata(
    scAnndata,
    max_count=35000,
    min_count=5000,
    MT_propor=10,
    min_cell=10,
    imgPath=savePath_2,
)  ## optional parameters: max_count, min_count, MT_propor, min_cell
scAnndata = Normalization(scAnndata)
scAnndata = Logtransformation(scAnndata)
scAnndata = Clustering(scAnndata, infer_mode=infer_mode, savePath=savePath)
compute_similarity(savePath=savePath, ann_data=scAnndata)

with open(os.path.join(savePath_2, "scAnndata.pkl"), "wb") as f:
    pickle.dump(scAnndata, f)
f.close()

# 2.4 clinical column selection and bulk data split
mode = "Bionomial"

bulkClinical = view_clinical_variables(savePath)
choose_clinical_variable(
    savePath,
    bulkClinical=bulkClinical,
    mode=mode,
    var_1="Response"
)

# data split
generate_val(
    savePath=savePath, validation_proportion=0.15, mode=mode
)  ## optinal parameter: validation_proportion

# sampling
perform_sampling_on_RNAseq(savePath=savePath,mode="SMOTE", threshold=0.5)

# 2.5 Genepair Transformation
GPextractor = GenePairExtractor(
    savePath=savePath,
    analysis_mode=mode,
    top_var_genes=2000,
    top_gene_pairs=1000,
    p_value_threshold=0.05,
    padj_value_threshold=None,
    max_cutoff=0.8,
    min_cutoff=-0.8,
)  ## optinal parameter: top_var_genes, top_gene_pairs, padj_value_threshold, padj_value_threshold

GPextractor.load_data()
GPextractor.run_extraction()
GPextractor.save_data()

## 3. Analysis
# 3.1 TiRank
savePath = "./web_test"
savePath_1 = os.path.join(savePath, "1_loaddata")
savePath_2 = os.path.join(savePath, "2_preprocessing")
savePath_3 = os.path.join(savePath, "3_Analysis")

if not os.path.exists(savePath_3):
    os.makedirs(savePath_3, exist_ok=True)


# 3.1.1 Dataloader
mode = "Bionomial"
infer_mode = "Cell"
device = "cuda" if torch.cuda.is_available() else "cpu"

PackData(savePath, mode=mode, infer_mode=infer_mode, batch_size=1024)

# 3.1.2 Training
encoder_type = "MLP"  ## Optional parameter

# Model parameter
initial_model_para(
    savePath=savePath,
    nhead=2,
    nhid1=96,
    nhid2=8,
    n_output=32,
    nlayers=3,
    n_pred=2,
    dropout=0.5,
    mode=mode,
    encoder_type=encoder_type,
    infer_mode=infer_mode,
)

tune_hyperparameters(
    ## Parameters Path
    savePath=savePath,
    device=device,
    n_trials=10,
)  ## optional parameters: n_trials

# 3.1.3 Inference
Predict(savePath=savePath, mode=mode, do_reject=True, tolerance=0.05, reject_mode="GMM")
```
