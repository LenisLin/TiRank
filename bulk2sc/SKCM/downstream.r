library(Seurat)
library(GSVA)

library(ggplot2)
library(ggpubr)
library(ggtext)
library(ggrepel)
library(ComplexHeatmap)
library(cowplot)

library(reshape2)

library(survival)
library(survminer)

seu <- readRDS("./tempfiles/GSE120575_downstream.rds")
meta <- read.csv("./tempfiles/sc_predict_score.csv", row.names = 1)
figurePath <- paste0("/home/lyx/project/scRankv2/bulk2sc/SKCM/tempfiles/", "figures/")
if (!dir.exists(figurePath)) {
    dir.create(figurePath, recursive = T)
}

# 火山图函数

colnames(meta) <- c("Reject", "Pred_score", "Predict")
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

MarkerList <- SelectMarkers_from_scRNA(all_marker,
    FC_threshold = 1.5, P_threshold = 0.05, p.adj = FALSE, top_vars = 50,
    savePath = paste0(figurePath, "DEGs Volcano.png")
)

NRmarkers <- list(MarkerList[[1]])
Rmarkers <- list(MarkerList[[2]])

### Test in bulk dataset
dataPath <- "/mnt/data/lyx/scRankv2/data/RNAseq_treatment/Melanoma/"
datasets <- sapply(list.files(dataPath), function(x) {
    return(strsplit(x, "_")[[1]][1])
})
datasets <- unique(unname(datasets))

for (data in datasets) {
    exp <- read.csv(paste0(dataPath, data, "_exp.csv"), row.names = 1)
    meta <- read.csv(paste0("/home/lyx/project/scRankv2/bulk2sc/SKCM/tempfiles/bulk_pred_score/", data, "_predict_score.csv"), row.names = 1)

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

    plotGSEAResults(
        cli = meta, score = NRscore, 
        savePath = paste0(figurePath, data, " NR genes enrich boxplot.png")
    )

    plotGSEAResults(
        cli = meta,score = Rscore, 
        savePath = paste0(figurePath, data, " Response genes enrich boxplot.png")
    )

    ## Survival
    Survival_KM(
        meta = meta, feature_ = NRscore,
        time = "OS_time", event = "OS_status", savePath = paste0(figurePath, data, " NR genes enrichment score with OS.png")
    )
    Survival_KM(
        meta = meta, feature_ = Rscore,
        time = "OS_time", event = "OS_status", savePath = paste0(figurePath, data, " R genes enrichment score with OS.png")
    )

    Survival_KM(
        meta = meta, feature_ = meta$Response,
        time = "OS_time", event = "OS_status", savePath = paste0(figurePath, data, " Response with OS.png")
    )

    Survival_KM(
        meta = meta, feature_ = meta$scRank_Label,
        time = "OS_time", event = "OS_status", savePath = paste0(figurePath, data, " Pred response with OS.png")
    )
}

## scRank预测亚群分布
analyzeScRankSubgroupDistribution(seu, savePath = paste0(figurePath, "scRank_cluster_distribution.png"))

## 特定亚群分析（CD8+ T细胞为例）
subcluster <- c("0", "1", "3", "4", "7", "8", "11")

seu_CD8 <- subset(seu, seurat_clusters %in% subcluster)
Idents(seu_CD8) <- seu_CD8$Predict
marker <- FindMarkers(seu_CD8, ident.1 = "Rank+", ident.2 = "Rank-")
marker <- marker[marker$p_val < pvalue_threshold, ]
# 根据logFC排序
marker <- marker[order(marker$avg_log2FC, decreasing = TRUE), ]
colnames(marker) <- c("p_val", "log2FoldChange", "pct.1", "pct.2", "padj")
marker$gene <- rownames(marker)

p <- plot_Volcano_2(marker, label_geneset = c(head(rownames(marker), 5), tail(rownames(marker), 5)), logFC = 0.5)
png(paste0(figurePath, "subcluster_DEGs Volcano.png"), width = 600, height = 800)
print(p)
dev.off()
