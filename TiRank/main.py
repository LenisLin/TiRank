from .Dataloader import *
from .GPextractor import *
from .Imageprocessing import *
from .LoadData import *
from .Model import *
from .SCSTpreprocess import *
from .TrainPre import *
from .Visualization import *
import pkg_resources

def GenePairSelection(scst_exp_path, bulk_exp_path, bulk_cli_path, datatype, mode, savePath, lognormalize, model_path = None, validation_proportion=0.15, top_var_genes=2000, top_gene_pairs=1000, p_value_threshold=0.05, max_cutoff=0.8, min_cutoff=-0.8):
  
  # Create folder
  
  if not os.path.exists(savePath):
    os.makedirs(savePath)
    
  savePath_1 = os.path.join(savePath, "1_loaddata")
  savePath_2 = os.path.join(savePath, "2_preprocessing")
  savePath_3 = os.path.join(savePath, "3_Analysis")
  
  if not os.path.exists(savePath_1):
    os.makedirs(savePath_1)
  if not os.path.exists(savePath_2):
    os.makedirs(savePath_2)
  if not os.path.exists(savePath_3):
    os.makedirs(savePath_3)    
    
  # Load data
  
  ## sc/st exp
  if datatype == "SC":
      scAnndata = sc.read_csv(scst_exp_path).T
  elif datatype == "ST":
      scAnndata = sc.read_visium(scst_exp_path)
      
  ## bulk exp
  bulkExp = pd.read_csv(bulk_exp_path, index_col=0)
  
  with open(os.path.join(savePath_1, 'bulk_exp.pkl'), 'wb') as f:
    pickle.dump(bulkExp, f)
  f.close()

  ## bulk cli
  bulkClinical = pd.read_csv(bulk_cli_path,header=None,index_col=0)
  
  with open(os.path.join(savePath_1, 'bulk_clinical.pkl'), 'wb') as f:
    pickle.dump(bulkClinical, f)
  f.close()
  
  with open(os.path.join(savePath_2, 'bulk_clinical.pkl'), 'wb') as f:
    pickle.dump(bulkClinical, f)
  f.close()
  
  # Process
  
  ## sc/st exp
  if lognormalize == True:
    sc.pp.normalize_total(scAnndata, target_sum=1e4, inplace = True)
    sc.pp.log1p(scAnndata)
    
  scAnndata = Clustering(scAnndata, infer_mode=datatype, savePath=savePath)
  compute_similarity(savePath=savePath, ann_data=scAnndata)
  
  
  # ST patho calculate
  if datatype == "ST":
    scAnndata = GetPathoClass(scAnndata,pretrain_path = model_path,image_save_path = savePath_2)
  
  
  with open(os.path.join(savePath_2, "scAnndata.pkl"), "wb") as f:
    pickle.dump(scAnndata, f)
  f.close()

  # data split
  generate_val(
      savePath=savePath, validation_proportion=validation_proportion, mode=mode
  )  ## optinal parameter: validation_proportion
  
  # sampling
  
  #if mode == "Classification":
  #  perform_sampling_on_RNAseq(savePath=savePath,mode="SMOTE", threshold=0.5)

  # GPextractor
  GPextractor = GenePairExtractor(
    savePath=savePath,
    analysis_mode=mode,
    top_var_genes=top_var_genes,
    top_gene_pairs=top_gene_pairs,
    p_value_threshold=p_value_threshold,
    max_cutoff=max_cutoff,
    min_cutoff=min_cutoff,
  )
  
  GPextractor.load_data()
  GPextractor.run_extraction()
  GPextractor.save_data()
  
def TiRank(savePath, datatype, mode, nhead=2, nhid1=96, nhid2=8, n_output=32, nlayers=3, dropout=0.5, encoder_type="MLP" ,device="cuda", tolerance=0.05):
  savePath_1 = os.path.join(savePath, "1_loaddata")
  savePath_2 = os.path.join(savePath, "2_preprocessing")
  savePath_3 = os.path.join(savePath, "3_Analysis")
  
  
  if mode == "Classification":
    n_pred = 2
  elif mode=="Regression":
    n_pred = 1
  elif mode=="Cox":
    n_pred = 1
      
  PackData(savePath, mode=mode, infer_mode=datatype, batch_size=1024)
  
  # 3.1.2 Training
  encoder_type = "MLP"  ## Optional parameter
  
  # Model parameter
  initial_model_para(
      savePath=savePath,
      nhead=nhead,
      nhid1=nhid1,
      nhid2=nhid2,
      n_output=n_output,
      nlayers=nlayers,
      n_pred=n_pred,
      dropout=dropout,
      mode=mode,
      encoder_type=encoder_type,
      infer_mode=datatype,
  )
  
  tune_hyperparameters(
      ## Parameters Path
      savePath=savePath,
      device=device,
      n_trials=10,
  )  ## optional parameters: n_trials
  
  # 3.1.3 Inference
  Predict(savePath=savePath, mode=mode, do_reject=True, tolerance=tolerance, reject_mode="GMM")
  
  # 3.1.4 Visualization
  plot_score_distribution(savePath)  # Display the prob score distribution