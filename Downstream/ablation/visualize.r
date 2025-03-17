## To visualize the results

library(ggplot2)
library(ggpubr)
library(pROC)    # For AUC-ROC
library(mclust)  # For ARI

## functions
compute_metrics <- function(df) {
  
  # Compute coverage: proportion of samples with confident predictions
  coverage <- mean(df$Reject == 0)
  
  # Filter to confident predictions only (Reject == 0)
  df_confident <- df[df$Reject == 0, ]
  
  # Extract true and predicted labels
  true_labels <- df_confident$response
  pred_labels <- ifelse(df_confident$Rank_Label == "Rank-", 0, 1)
  
  # Create confusion matrix
  conf_mat <- table(factor(true_labels, levels = c(0, 1)),
                    factor(pred_labels, levels = c(0, 1)))
  
  # Extract TP, TN, FP, FN from confusion matrix
  TN <- conf_mat[1, 1]  # True Negatives
  FP <- conf_mat[1, 2]  # False Positives
  FN <- conf_mat[2, 1]  # False Negatives
  TP <- conf_mat[2, 2]  # True Positives
  
  # Compute classification metrics
  accuracy <- (TP + TN) / sum(conf_mat)
  
  # Metrics for class 1 (positive class)
  precision_1 <- if (TP + FP > 0) TP / (TP + FP) else 0
  recall_1 <- if (TP + FN > 0) TP / (TP + FN) else 0
  f1_1 <- if (precision_1 + recall_1 > 0) 2 * (precision_1 * recall_1) / (precision_1 + recall_1) else 0
  
  # Metrics for class 0 (negative class)
  precision_0 <- if (TN + FN > 0) TN / (TN + FN) else 0
  recall_0 <- if (TN + FP > 0) TN / (TN + FP) else 0
  f1_0 <- if (precision_0 + recall_0 > 0) 2 * (precision_0 * recall_0) / (precision_0 + recall_0) else 0
  
  # Macro-averaged F1 score
  macro_f1 <- (f1_0 + f1_1) / 2
  
  # Balanced accuracy
  balanced_accuracy <- (recall_0 + recall_1) / 2
  
  # Adjusted Rand Index
  ari <- adjustedRandIndex(true_labels, pred_labels)
  
  # AUC-ROC using Rank_Score across all samples
  roc_obj <- roc(df$response, df$Rank_Score, quiet = TRUE)
  auc_roc <- auc(roc_obj)
  
  # Compile all metrics into a named vector
  metrics <- c(
    coverage = coverage,
    accuracy = accuracy,
    precision_1 = precision_1,
    recall_1 = recall_1,
    f1_1 = f1_1,
    precision_0 = precision_0,
    recall_0 = recall_0,
    f1_0 = f1_0,
    macro_f1 = macro_f1,
    balanced_accuracy = balanced_accuracy,
    ari = ari,
    auc_roc = auc_roc
  )
  
  return(metrics)
}

## Set path
savePath_base <- "/mnt/NAS_21T/ProjectResult/TiRank/results/ablation_results"

if(!dir.exists(savePath_base)){
    dir.create(savePath_base,recursive = T)
}

resultPath <- "/mnt/NAS_21T/ProjectResult/TiRank/results/ablation"

## Get iterations
iterations <- list.files(resultPath)

## Initialize list to store all metrics
all_metrics <- list()

## Define experimental settings (models and ablation studies)
settings <- list(
  list(name = "MLP", file = "spot_predict_score_MLP.csv"),
  list(name = "DenseNet", file = "spot_predict_score_DenseNet.csv"),
  list(name = "Transformer", file = "spot_predict_score_Transformer.csv"),
  list(name = "abLoss1", file = "spot_predict_score_abLoss1.csv"),
  list(name = "abLoss2", file = "spot_predict_score_abLoss2.csv"),
  list(name = "abLoss12", file = "spot_predict_score_abLoss12.csv")
)

## Get iterations
iterations <- list.files(resultPath)

for (inte_ in iterations) {
  resultPath_1 <- file.path(resultPath, inte_)
  
  ## Get datasets
  Datasets <- list.files(resultPath_1)
  
  for (dataset_ in Datasets) {
    ## Get analysis path
    resultPath_2 <- file.path(resultPath_1, dataset_, "3_Analysis")
    
    ## Process each experimental setting
    for (setting in settings) {
      ## Read the file
      model_data <- read.csv(file.path(resultPath_2, setting$file), row.names = 1)
      
      ## Compute metrics
      metrics <- compute_metrics(model_data)
      
      ## Create a dataframe with iteration, dataset, model, and metrics
      metrics_df <- data.frame(
        iteration = inte_,
        dataset = dataset_,
        model = setting$name,
        t(metrics)
      )
      
      ## Append to the list
      all_metrics[[length(all_metrics) + 1]] <- metrics_df
    }
  }
}

## Combine all metrics into a single dataframe
all_metrics_df <- do.call(rbind, all_metrics)

## Reset row names for clarity
rownames(all_metrics_df) <- NULL