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
plot_Volcano_2 <- function(result, logFC = 0.5, p_val = 0.05, label_geneset = NULL) {
    result$change <- "NONE"
    result$change[which(result$log2FoldChange >= logFC & result$p_val <= p_val)] <- "UP"
    result$change[which(result$log2FoldChange <= (-logFC) & result$p_val <= p_val)] <- "DOWN"
    xlim <- max(abs(result$log2FoldChange))
    if (is.null(label_geneset)) {
        p <- ggplot(result, aes(x = log2FoldChange, y = -log10(p_val))) +
            geom_point(data = result, aes(x = log2FoldChange, y = -log10(p_val), color = change)) +
            theme_bw() +
            geom_vline(xintercept = c(-logFC, logFC), lty = 2) +
            geom_hline(yintercept = c(-log10(p_val)), lty = 2) +
            scale_x_continuous(limits = c(-xlim, xlim)) +
            coord_fixed(ratio = (2 * xlim) / (max(-log10(result$p_val), na.rm = T))) +
            theme(
                panel.grid = element_blank(), legend.title = element_blank(),
                panel.grid.minor = element_blank(),
                axis.text = element_text(color = "black")
            ) +
            xlab("log2FoldChange") +
            ylab("-log10P-value") +
            scale_color_manual(values = c("NONE" = "grey", "UP" = "red", "DOWN" = "blue")) +
            guides(fill = FALSE)
    } else {
        p <- ggplot(result, aes(x = log2FoldChange, y = -log10(p_val))) +
            geom_point(data = result, aes(x = log2FoldChange, y = -log10(p_val), color = change)) +
            geom_label_repel(
                data = result[which(result$gene %in% label_geneset), ],
                aes(x = log2FoldChange, y = -log10(p_val), label = gene, fill = change), color = "white", fontface = "italic"
            ) +
            theme_bw() +
            geom_vline(xintercept = c(-logFC, logFC), lty = 2) +
            geom_hline(yintercept = c(-log10(p_val)), lty = 2) +
            scale_x_continuous(limits = c(-xlim, xlim)) +
            coord_fixed(ratio = (2 * xlim) / (max(-log10(result$p_val), na.rm = T))) +
            theme(
                panel.grid = element_blank(), legend.title = element_blank(),
                panel.grid.minor = element_blank(),
                axis.text = element_text(color = "black")
            ) +
            xlab("log2FoldChange") +
            ylab("-log10P-value") +
            scale_color_manual(values = c("UP" = "red", "DOWN" = "blue", "NONE" = "grey")) +
            scale_fill_manual(values = c("UP" = "red", "DOWN" = "blue", "NONE" = "grey")) +
            guides(fill = FALSE)
    }
    return(p)
}

colnames(meta) = c("Reject", "Pred_score", "Predict")
table(meta$Predict)
seu <- AddMetaData(seu, meta)

# p1 <- DimPlot(seu)
# png(paste0(figurePath, "UMAP-Clutering.png"))
# print(p1)
# dev.off()

p2 <- DimPlot(seu, group.by = c("Predict"), cols = c("Background" = "gray91", "Rank-" = "#4cb1c4", "Rank+" = "#b5182b"))
png(paste0(figurePath, "UMAP-Rank.png"))
print(p2)
dev.off()

# 差异基因火山图
Idents(seu) <- seu$Predict
all_marker <- FindMarkers(seu, ident.1 = "Rank+", ident.2 = "Rank-")
# all_marker <- FindMarkers(seu, ident.1 = "Rank+", ident.2 = "Background")

marker <- all_marker[all_marker$p_val <= 0.01, ]
# marker <- all_marker[all_marker$p_val_adj <= 0.05, ]

# 根据logFC排序
marker <- marker[order(marker$avg_log2FC, decreasing = TRUE), ]
marker$gene <- rownames(marker)

colnames(marker) <- c("p_val", "log2FoldChange", "pct.1", "pct.2", "padj", "gene")

p <- plot_Volcano_2(marker, label_geneset = c(head(marker, 5)[, "gene"], tail(marker, 5)[, "gene"]), logFC = 0.5)

png(paste0(figurePath, "DEGs Volcano.png"), width = 600, height = 800)
print(p)
dev.off()

saveRDS(marker, paste0(figurePath, "marker.rds"))

## Boxplot
marker <- readRDS(paste0(figurePath, "marker.rds"))

## up-regulated in NR
NRmarkers <- marker[marker$log2FoldChange >= log2(1.5), "gene"]
Rmarkers <- marker[marker$log2FoldChange <= (-log2(1.5)), "gene"]

# marker <- marker[1:50]
NRmarkers <- list(NRmarkers)
Rmarkers <- list(Rmarkers)

## GSEA function
GSEAfunc <- function(meta, score) {
    data_combined <- cbind(meta, score = as.numeric(score))
    p <- ggboxplot(data_combined,
        x = "Response", y = "score",
        color = "Response", palette = c("#00AFBB", "#E7B800"),
        add = "jitter"
    )
    my_comparisons <- list(c("R", "NR"))
    p <- p + stat_compare_means(comparisons = my_comparisons, label = "p.label", size = 5, method = "wilcox.test")

    return(p)
}

### Buld
dataPath <- "/mnt/data/lyx/scRankv2/data/RNAseq_treatment/Melanoma/"
datasets <- sapply(list.files(dataPath), function(x) {
    return(strsplit(x, "_")[[1]][1])
})
datasets <- unique(unname(datasets))

plot_feature_distribution <- function(data_matrix) {
    rand_x <- sample(1:nrow(data_matrix), 50)
    rand_y <- sample(1:ncol(data_matrix), 20)
    data_matrix <- data_matrix[rand_x, rand_y]
    # Check if data_matrix is indeed a matrix
    if (!is.matrix(data_matrix)) {
        stop("The input data must be a matrix.")
    }

    # Convert the matrix to a long format data frame suitable for ggplot2
    data_long <- melt(data_matrix, varnames = c("Feature", "Sample"))

    # Create the boxplot
    p <- ggplot(data_long, aes(x = as.factor(Sample), y = value)) +
        geom_boxplot(outlier.shape = NA) + # Hide outliers to avoid overplotting
        geom_jitter(width = 0.2, alpha = 0.5, size = 0.1) + # Add jitter to show actual points
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + # Adjust text angle for x-axis labels
        labs(x = "Sample", y = "Feature Value", fill = "Feature", title = "Feature Distribution per Sample")

    return(p)
}

plot_gene_diff <- function(exp,cli,genes) {
    data_combine <- as.data.frame(cbind(t(exp[genes,]),cli$Response))
    colnames(data_combine) <- c(genes,"Response")

    data_long <- tidyr::pivot_longer()

    # Create the boxplot
    p <- ggplot(data_long, aes(x = as.factor(Sample), y = value)) +
        geom_boxplot(outlier.shape = NA) + # Hide outliers to avoid overplotting
        geom_jitter(width = 0.2, alpha = 0.5, size = 0.1) + # Add jitter to show actual points
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + # Adjust text angle for x-axis labels
        labs(x = "Sample", y = "Feature Value", fill = "Feature", title = "Feature Distribution per Sample")

    return(p)
}

for (data in datasets) {
    exp <- read.csv(paste0(dataPath, data, "_exp.csv"), row.names = 1)
    meta <- read.csv(paste0("/home/lyx/project/scRankv2/bulk2sc/SKCM/tempfiles/bulk_pred_score/", data, "_predict_score.csv"), row.names = 1)

    colnames(meta) <- c("Response", "OS_status", "OS_time", "scRank_Label")
    meta$Response <- ifelse(meta$Response == 0, "R", "NR")
    meta$scRank_Label <- ifelse(meta$scRank_Label == 0, "R", "NR")

    exp <- as.matrix(exp)

    # p <- plot_feature_distribution(exp)
    # png(paste0(figurePath, data, " feature distribution.png"), width = 600, height = 450)
    # print(p)
    # dev.off()

    NRscore <- gsva(exp, gset.idx.list = NRmarkers, method = "ssgsea")
    Rscore <- gsva(exp, gset.idx.list = Rmarkers, method = "ssgsea")

    p1 <- GSEAfunc(meta, NRscore)
    png(paste0(figurePath, data, " NR genes enrich boxplot.png"), width = 450, height = 450)
    print(p1)
    dev.off()

    p1 <- GSEAfunc(meta, Rscore)
    png(paste0(figurePath, data, " Response genes enrich boxplot.png"), width = 450, height = 450)
    print(p1)
    dev.off()

    ## Survival
    data_combined <- cbind(meta, NRscore = as.numeric(NRscore))
    data_combined <- cbind(data_combined, Rscore = as.numeric(Rscore))

    NRscore_cutoff <- surv_cutpoint(data = data_combined, time = "OS_time", event = "OS_status", variables = "NRscore")$cutpoint[1, 1]
    Rscore_cutoff <- surv_cutpoint(data = data_combined, time = "OS_time", event = "OS_status", variables = "Rscore")$cutpoint[1, 1]

    data_combined$NRscore_group <- ifelse(data_combined$NRscore > NRscore_cutoff, "High", "Low")
    data_combined$Rscore_group <- ifelse(data_combined$Rscore > Rscore_cutoff, "High", "Low")

    fit1 <- survfit(Surv(time = data_combined$OS_time, event = data_combined$OS_status) ~ NRscore_group, data = data_combined)
    fit2 <- survfit(Surv(time = data_combined$OS_time, event = data_combined$OS_status) ~ Rscore_group, data = data_combined)

    p <- ggsurvplot(fit1,
        data = data_combined,
        pval = TRUE, # 如果你想显示p值
        risk.table = TRUE
    ) # 如果你想显示置信区间
    png(paste0(figurePath, data, " NR genes enrichment score with OS.png"), width = 450, height = 450)
    print(p)
    dev.off()

    p <- ggsurvplot(fit2,
        data = data_combined,
        pval = TRUE, # 如果你想显示p值
        risk.table = TRUE
    ) # 如果你想显示置信区间
    png(paste0(figurePath, data, " R genes enrichment score with OS.png"), width = 450, height = 450)
    print(p)
    dev.off()

    # fit <- survfit(Surv(time = data_combined$OS_time, event = data_combined$OS_status) ~ Response, data = data_combined)

    # p <- ggsurvplot(fit,
    #     data = data_combined,
    #     pval = TRUE, # 如果你想显示p值
    #     risk.table = TRUE
    # ) # 如果你想显示置信区间
    # png(paste0(figurePath, data, " Response with OS.png"), width = 450, height = 450)
    # print(p)
    # dev.off()

    # fit <- survfit(Surv(time = data_combined$OS_time, event = data_combined$OS_status) ~ scRank_Label, data = data_combined)

    # p <- ggsurvplot(fit,
    #     data = data_combined,
    #     pval = TRUE, # 如果你想显示p值
    #     risk.table = TRUE
    # ) # 如果你想显示置信区间
    # png(paste0(figurePath, data, " Pred response with OS.png"), width = 450, height = 450)
    # print(p)
    # dev.off()

    ## Determine certain DEGs

}
