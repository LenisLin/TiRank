# Visualization
library(pheatmap)
library(RColorBrewer)
library(ggplot2)
library(ggpubr)
library(ggrepel)

## Line chart for deltaS and C-GPs
LineChart_GPs <- function(mat, savePath) {
    num_ <- c()
    cutoff <- c(0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.65)
    for (cutoff_ in cutoff) {
        mat2 <- ifelse(mat > cutoff_, 1, 0)
        num_ <- c(num_, sum(mat2))
    }
    plotdf <- data.frame("Cutoff" = cutoff, "Numpairs" = num_)

    p <- ggplot(plotdf, aes(x = Cutoff, y = Numpairs)) +
        geom_line() +
        geom_point()

    png(paste0(savePath, "Line Chart for different deltaS cutoff.png"), width = 800, height = 600)
    print(p)
    dev.off()

    return(NULL)
}

## Heatmap for celltype-specific gene pairs
Heatmap_CGPs <- function(deltaS_Mat, CGPList, savePath) {
    celltypes <- names(CGPList)
    deltaS_Mat <- deltaS_Mat[, celltypes]

    CGPs <- c()
    for (i in 1:length(CGPList)) {
        CGPs <- c(CGPs, names(CGPList[[i]]))
    }

    plot_df <- deltaS_Mat[CGPs, ]
    plot_df <- as.data.frame(plot_df)

    color <- colorRampPalette(c("#436eee", "white", "#EE0000"))(100)
    p <- pheatmap(plot_df,
        color = color, scale = "row",
        cluster_rows = F, cluster_cols = T,
        legend_labels = c("ΔS high", "ΔS low"), legend = T,
        show_rownames = F, show_colnames = T
    )

    pdf(paste0(savePath, "C-GP heatmap.pdf"), width = 6, height = 8)
    print(p)
    dev.off()
    return(NULL)
}

## Heatmap for Robust celltype-specific gene pairs
Heatmap_RCGPs <- function(GPMatrix, celllabel, savePath) {
    deltaS_Mat <- GetCandidateCGPs(GPMatrix, celllabel)

    color <- colorRampPalette(c("#436eee", "white", "#EE0000"))(100)
    p <- pheatmap(deltaS_Mat,
        color = color, scale = "row",
        cluster_rows = F, cluster_cols = T,
        legend_labels = c("ΔS high", "ΔS low"), legend = T,
        show_rownames = F, show_colnames = T
    )

    pdf(paste0(savePath, "RC-GP heatmap.pdf"), width = 6, height = 8)
    print(p)
    dev.off()
    return(NULL)
}

## Boxplot for Prognostic-associated Celltype specific gene pairs
Boxplot_PRCGP <- function(resultDF, savePath) {
    plot_df_ <- resultDF
    colnames(plot_df_) <- c("Celltype", "HR", "deltaS", "Dataset", "Majortype")
    plot_df_$HR <- log(as.numeric(plot_df_$HR), base = exp(1))
    plot_df_$Celltype <- as.factor(plot_df_$Celltype)
    plot_df_$Majortype <- as.factor(plot_df_$Majortype)

    p_tem <- ggplot(data = plot_df_, aes(x = Celltype, y = HR, fill = Celltype)) +
        geom_boxplot(alpha = 0.7) +
        scale_y_continuous(name = "ln(HR)") +
        scale_x_discrete(name = paste0("Cell subtypes")) +
        ggtitle(paste0("Prognostic value boxplot of cell subpopulations")) +
        geom_hline(yintercept = c(0), linetype = "dashed") +
        theme_classic() +
        theme(
            plot.title = element_text(size = 14, face = "bold"),
            text = element_text(size = 12),
            axis.title = element_text(face = "bold"),
            axis.text.x = element_text(size = 8, angle = 90)
        ) +
        coord_flip() +
        facet_grid(Majortype ~ Dataset, scales = "free_y", space = "free_y")

    pdf(paste0(savePath, "Boxplot of PR-CPG.pdf"), width = 10, height = 8)
    print(p_tem)
    dev.off()
    return(NULL)
}

## Pair T-test of entropy from two modes
PairTtestForEntro <- function(mat1, mat2) {
    mat1$mode <- rep("Scale", times = nrow(mat1))
    mat2$mode <- rep("Raw", times = nrow(mat2))

    plotdf <- rbind(mat1, mat2)
    plotdf$Entropy <- as.numeric(plotdf$Entropy)
    plotdf$mode <- as.factor(plotdf$mode)

    p <- ggpaired(
        plotdf,
        x = "mode", y = "Entropy", color = "mode", line.color = "gray",
        line.size = 0.4, palette = "jco"
    ) + stat_compare_means(paired = TRUE) +
        facet_wrap(~Label)
    pdf("test.pdf", width = 8, height = 6)
    print(p)
    dev.off()
}

## Bubble plot for visualizing celltypes properity
BubbleplotForCelltypePro <- function(mat1, mat2, savePath) {
    mat1$mode <- rep("Scale", times = nrow(mat1))
    mat2$mode <- rep("Raw", times = nrow(mat2))

    plotdf <- rbind(mat1, mat2)
    plotdf$Entropy <- as.numeric(plotdf$Entropy)
    plotdf$mode <- as.factor(plotdf$mode)
    plotdf$Entropy2 <- 1 - plotdf$Entropy
    plotdf$Label <- as.factor(plotdf$Label)

    levels(plotdf$Label) <- c("protective", "risk")

    p <- ggplot(plotdf, aes(x = Celltype, y = Entropy2)) +
        geom_point(aes(colour = factor(Label), size = Entropy2)) +
        theme(
            panel.background = element_blank(),
            panel.grid.major = element_line(colour = "white"),
            panel.border = element_rect(colour = "white", fill = NA),
            axis.text.x = element_text(size = 8, angle = 90)
        ) +
        scale_color_manual(values = c("#436eee", "#EE0000")) +
        coord_flip() +
        facet_grid(Majortype ~ mode, scales = "free_y", space = "free_y")
    pdf(savePath, height = 6, width = 8)
    print(p)
    dev.off()
}

## UMAP of PGP rank score
VisualizeForPGPinSC <- function(seuobj, savePath) {
    if (nrow(seuobj@reductions$umap@cell.embeddings) == 0) {
        seuobj <- RunUMAP(seuobj)
    }

    umap_cor <- seuobj@reductions$umap@cell.embeddings
    scRankScore <- seuobj@meta.data$scRankScore
    clusterid <- seuobj@meta.data$seurat_clusters
    scRankLabel <- seuobj@meta.data$scRankLabel

    plotdfTemp <- cbind(umap_cor, scRankScore, clusterid)
    plotdfTemp <- as.data.frame(plotdfTemp)
    colnames(plotdfTemp) <- c("UMAP_1", "UMAP_2", "scRankScore", "ClusterID")

    plotdfTemp$UMAP_1 <- as.numeric(plotdfTemp$UMAP_1)
    plotdfTemp$UMAP_2 <- as.numeric(plotdfTemp$UMAP_2)
    plotdfTemp$scRankScore <- as.numeric(plotdfTemp$scRankScore)
    plotdfTemp$ClusterID <- as.factor(plotdfTemp$ClusterID)
    plotdfTemp$scRankLabel <- as.factor(scRankLabel)


    p <- ggplot(plotdfTemp, aes(UMAP_1, UMAP_2)) +
        geom_point(aes(color = scRankScore), size = 0.1) +
        scale_color_gradient2(low = "#0040ff", mid = "grey", high = "#EE0000") +
        theme_classic()


    pdf(paste0(savePath, "scRank score of GSE144735.pdf"), height = 6, width = 8)
    print(p)
    dev.off()

    p <- ggplot(plotdfTemp, aes(UMAP_1, UMAP_2)) +
        geom_point(aes(color = scRankLabel), size = 0.1) +
        scale_color_manual(values = c("scRank-" = "#0040ff", "n.s." = "grey", "scRank+" = "#EE0000")) +
        theme_classic() +
        guides(color = guide_legend(override.aes = list(size = 8, alpha = 1)))


    pdf(paste0(savePath, "scRank label of GSE144735.pdf"), height = 6, width = 8)
    print(p)
    dev.off()

    p <- ggplot(plotdfTemp, aes(UMAP_1, UMAP_2)) +
        geom_point(aes(color = ClusterID), size = 0.1) +
        theme_classic() +
        guides(color = guide_legend(override.aes = list(size = 8, alpha = 1)))


    pdf(paste0(savePath, "clusterid of GSE144735.pdf"), height = 6, width = 8)
    print(p)
    dev.off()


    ## scRank label in cluster id
    plotdfTemp2 <- matrix(data = NA, nrow = 0, ncol = 3)
    plotdfTemp2 <- as.data.frame(plotdfTemp2)

    clusterids <- names(table(plotdfTemp$ClusterID))

    for (clusterid in clusterids) {
        TempDF <- subset(plotdfTemp, ClusterID == clusterid)
        tableDF <- as.data.frame(table(TempDF$scRankLabel))
        tableDF <- cbind(rep(clusterid, nrow(tableDF)), tableDF)
        plotdfTemp2 <- rbind(plotdfTemp2, tableDF)
    }

    colnames(plotdfTemp2) <- c("ClusterID", "scRankLabel", "Counts")
    plotdfTemp2 <- subset(plotdfTemp2, (scRankLabel == "scRank-" | scRankLabel == "scRank+"))

    p <- ggplot(data = plotdfTemp2, aes(x = ClusterID, y = Counts)) +
        geom_bar(aes(fill = scRankLabel), stat = "identity", width = 0.9, position = "dodge") +
        theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            axis.title = element_text(size = 12, face = "bold"),
            axis.text.x = element_text(angle = 90, hjust = 1),
            plot.margin = unit(rep(3, 4), "lines"),
            legend.position = "bottom", legend.box = "horizontal"
        ) +
        scale_fill_manual(values = c("scRank-" = "#0040ff", "scRank+" = "#EE0000")) +
        coord_flip()

    pdf(paste0(savePath, "scRank label in clusterid.pdf"), height = 8, width = 6)
    print(p)
    dev.off()
}

## volcano plot
VolcanoPlot <- function(MarkerDF, q_threshold = 0.01, log2FC_threshold = 1, labelN = 10, savePath) {
    MarkerDF$pointLabel <- NA
    for (i in 1:nrow(MarkerDF)) {
        temp <- MarkerDF[i, ]
        if (temp$avg_log2FC >= log2FC_threshold & temp$p_val_adj <= q_threshold) {
            MarkerDF[i, "pointLabel"] <- "Up-regulated"
        } else if (temp$avg_log2FC <= (-log2FC_threshold) & temp$p_val_adj <= q_threshold) {
            MarkerDF[i, "pointLabel"] <- "Down-regulated"
        } else {
            MarkerDF[i, "pointLabel"] <- "n.s."
        }
    }

    Top <- MarkerDF
    Top <- Top[which(Top$p_val_adj <= q_threshold), ]
    Top <- Top[order(Top$avg_log2FC), ]
    Top <- Top[c(1:labelN, nrow(Top):(nrow(Top) - labelN)), ]
    Top$TextLabel <- rownames(Top)

    ## plot
    p <- ggplot(MarkerDF, aes(x = avg_log2FC, y = -log10(p_val_adj))) +
        geom_point(size = 2, aes(color = pointLabel)) +
        scale_x_continuous(limits = c(-7, 7)) +
        scale_y_continuous(limits = c(0, 32)) +
        geom_hline(yintercept = -log10(q_threshold), linetype = 4) +
        geom_vline(xintercept = c(-log2(log2FC_threshold), log2(log2FC_threshold)), linetype = 4) +
        xlab(expression("log"[2] * " fold change")) +
        ylab(expression("-log"[10] * " q-value")) +
        scale_color_manual(values = c("Up-regulated" = "#FC4E07", "n.s." = "#C0C0C0", "Down-regulated" = "#0000FF"), name = "Label") +
        geom_label_repel(data = Top, aes(x = avg_log2FC, y = -log10(p_val_adj), label = TextLabel), size = 4) +
        theme(legend.text = element_text(size = 20)) +
        theme(legend.title = element_text(size = 20)) +
        theme(axis.title = element_text(size = 20)) +
        theme(axis.text = element_text(size = 15))

    pdf(paste0(savePath, "VolcanoPlot of DEGs in different scRank label.pdf"), height = 6, width = 8)
    print(p)
    dev.off()

    return(NULL)
}
