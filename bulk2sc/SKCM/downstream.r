library(Seurat)
library(GSVA)

library(ggplot2)
library(ggpubr)
library(ggtext)
library(ggrepel)
# library(ComplexHeatmap)
library(cowplot)

library(reshape2)

library(survival)
library(survminer)

source("../../scRank/Visualization.r")

seu <- readRDS("./tempfiles/GSE120575_downstream.rds")
meta <- read.csv("./tempfiles/sc_predict_score.csv", row.names = 1)
figurePath <- paste0("/home/lenislin/Experiment/projects/scRankv2/bulk2sc/SKCM/tempfiles/", "figures/")
if (!dir.exists(figurePath)) {
    dir.create(figurePath, recursive = T)
}

# 火山图函数

colnames(meta) <- c("Reject", "Pred_score")
meta$Predict <- ifelse(
    ((1 - meta$Reject) * meta$Pred_score) == 0, "Background",
    ifelse(((1 - meta$Reject) * meta$Pred_score) > 0.5, "Rank+", "Rank-")
)
table(meta$Predict)
seu <- AddMetaData(seu, meta)

p1 <- DimPlot(seu)
png(paste0(figurePath, "UMAP-Clutering.png"))
print(p1)
dev.off()

p2 <- DimPlot(seu, group.by = c("Predict"), cols = c("Background" = "gray91", "Rank-" = "#4cb1c4", "Rank+" = "#b5182b"))
png(paste0(figurePath, "UMAP-Rank.png"))
print(p2)
dev.off()

# 差异基因火山图
Idents(seu) <- seu$Predict
all_marker <- FindMarkers(seu, ident.1 = "Rank+", ident.2 = "Rank-")
# all_marker <- FindMarkers(seu, ident.1 = "Rank+", ident.2 = "Background")
# all_marker <- FindMarkers(seu, ident.1 = "Rank-", ident.2 = "Background")

MarkerList <- SelectMarkers_from_scRNA(all_marker,
    FC_threshold = 1.5, P_threshold = 0.05, p.adj = FALSE, top_vars = 15,
    savePath = paste0(figurePath, "DEGs Volcano.png")
)

NRmarkers <- list(MarkerList[[1]])
Rmarkers <- list(MarkerList[[2]])

### Test in bulk dataset
dataPath <- "/home/lenislin/Experiment/data/scRankv2/data/RNAseq_treatment/Melanoma/"
metaPath <- "/home/lenislin/Experiment/projects/scRankv2/bulk2sc/SKCM/tempfiles/figures/bulk_test/"
datasets <- sapply(list.files(dataPath), function(x) {
    return(strsplit(x, "_")[[1]][1])
})
datasets <- unique(unname(datasets))

for (data in datasets) {
    exp <- read.csv(paste0(dataPath, data, "_exp.csv"), row.names = 1)
    meta <- read.csv(paste0(metaPath, data, "_predict_score.csv"), row.names = 1)

    colnames(meta) <- c("Response", "OS_status", "OS_time", "scRank_Label")
    meta$Response <- ifelse(meta$Response == 0, "R", "NR")
    meta$scRank_Label <- ifelse(meta$scRank_Label == 0, "R", "NR")

    exp <- as.matrix(exp)

    # p <- Plot_feature_distribution(exp)
    # png(paste0(figurePath, data, " feature distribution.png"), width = 600, height = 450)
    # print(p)
    # dev.off()

    NRscore <- gsva(exp, gset.idx.list = NRmarkers, method = "ssgsea")
    Rscore <- gsva(exp, gset.idx.list = Rmarkers, method = "ssgsea")

    plotGSEAResults(metadata = meta, score = NRscore, outputPath = paste0(figurePath, data, " NR genes enrich boxplot.png"))
    plotGSEAResults(metadata = meta, score = Rscore, outputPath = paste0(figurePath, data, " Response genes enrich boxplot.png"))

    ## Survival
    timeColumn <- "OS_time"
    eventColumn <- "OS_status"

    ### NR enrichment score
    combinedData <- GroupingPatients(metadata = meta, features = NRscore, timeColumn = "OS_time", eventColumn = "OS_status")
    survivalFit <- survfit(
        Surv(time = combinedData[, timeColumn], event = combinedData[, eventColumn]) ~ Group,
        data = combinedData
    )
    plotSurvivalKM(survivalFit = survivalFit, outputPath = paste0(figurePath, data, " NR genes enrichment score with OS.png"))


    ### R enrichment score
    combinedData <- GroupingPatients(metadata = meta, features = Rscore, timeColumn = "OS_time", eventColumn = "OS_status")
    survivalFit <- survfit(
        Surv(time = combinedData[, timeColumn], event = combinedData[, eventColumn]) ~ Group,
        data = combinedData
    )
    plotSurvivalKM(survivalFit = survivalFit, outputPath = paste0(figurePath, data, " R genes enrichment score with OS.png"))

    ### Ground Truth response label
    combinedData <- GroupingPatients(metadata = meta, features = meta$Response, timeColumn = "OS_time", eventColumn = "OS_status")
    survivalFit <- survfit(
        Surv(time = combinedData[, timeColumn], event = combinedData[, eventColumn]) ~ Group,
        data = combinedData
    )
    plotSurvivalKM(survivalFit = survivalFit, outputPath = paste0(figurePath, data, " Response with OS.png"))

    ### Predict response label
    combinedData <- GroupingPatients(metadata = meta, features = meta$scRank_Label, timeColumn = "OS_time", eventColumn = "OS_status")
    survivalFit <- survfit(
        Surv(time = combinedData[, timeColumn], event = combinedData[, eventColumn]) ~ Group,
        data = combinedData
    )
    plotSurvivalKM(survivalFit = survivalFit, outputPath = paste0(figurePath, data, " Pred response with OS.png"))
}

## scRank预测亚群分布
analyzeScRankDistribution(seuratObject = seu, outputPath = paste0(figurePath, "scRank_cluster_distribution.png"))

## plot the markers of scRNA-seq data
markers <- c("MS4A1","NKG7","CD3E","CD8A","CD4","CD14","CD68")
p1 <- FeaturePlot(seu, features = markers)
png(paste0(paste0(figurePath, "markers feature plot of GSE120575.png")),width = 800,height = 800)
print(p1)
dev.off()

## CD8+ T subcluster analysis
subcluster <- c("0", "1", "3", "4", "7","8","9")

## DEG and enrichment analysis
cluster_name = "CD8T"
seu_ <- analysis_subcluster(seu,subclusters = subcluster, cluster_name = cluster_name, ident.1 = "Rank+",ident.2 = "Rank-",figurePath = figurePath)

## Boxplot visualize diff
# gene_list <- c("TCF7","IL7R","CCR7","PRF1","TOX","PDCD1","CTLA4")
gene_list <- c(
    "GZMK","ICOS","TWF2", ## Response associated (Rank-)
    "RILPL2","PDCD1","IL21R" ## Resistant associated (Rank+)
    )

p <- plotGeneExpressionSwarm(seu_,gene_list,class_category=c("Rank+","Rank-"))
png(paste0(paste0(figurePath, "marker violin of ",cluster_name,".png")),width = 800,height = 800)
print(p)
dev.off()

