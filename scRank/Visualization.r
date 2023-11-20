# Functions for visualization
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

## RNA to scRNA

# Function to create volcano plots for differential gene expression analysis
createVolcanoPlot <- function(deResults, logFoldChangeThreshold = 0.5, pValueThreshold = 0.05, labeledGenes = NULL, outputPath) {
    # Classifying genes based on log fold change and p-value
    deResults$geneStatus <- "Unchanged"
    deResults$geneStatus[deResults$log2FoldChange >= logFoldChangeThreshold & deResults$p_val <= pValueThreshold] <- "Upregulated"
    deResults$geneStatus[deResults$log2FoldChange <= -logFoldChangeThreshold & deResults$p_val <= pValueThreshold] <- "Downregulated"

    # Setting plot limits
    xLimit <- max(abs(deResults$log2FoldChange))

    # Building the plot
    p <- ggplot(deResults, aes(x = log2FoldChange, y = -log10(p_val))) +
        geom_point(aes(color = geneStatus)) +
        theme_bw() +
        geom_vline(xintercept = c(-logFoldChangeThreshold, logFoldChangeThreshold), lty = 2) +
        geom_hline(yintercept = -log10(pValueThreshold), lty = 2) +
        scale_x_continuous(limits = c(-xLimit, xLimit)) +
        coord_fixed(ratio = (2 * xLimit) / max(-log10(deResults$p_val), na.rm = TRUE)) +
        theme(
            panel.grid = element_blank(), legend.title = element_blank(),
            panel.grid.minor = element_blank(), axis.text = element_text(color = "black")
        ) +
        xlab("Log2 Fold Change") +
        ylab("-Log10 P-value") +
        scale_color_manual(values = c("Unchanged" = "grey", "Upregulated" = "red", "Downregulated" = "blue"))

    # Adding gene labels if provided
    if (!is.null(labeledGenes)) {
        labeledData <- deResults[deResults$gene %in% labeledGenes, ]
        p <- p + geom_label_repel(
            data = labeledData, aes(label = gene, fill = geneStatus),
            color = "white", fontface = "italic"
        ) +
            scale_fill_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Unchanged" = "grey"))
    }

    # Saving the plot
    ggsave(outputPath, p, width = 6, height = 8)
}

# Function to plot the distribution of features in a data matrix
plotFeatureDistribution <- function(dataMatrix, outputPath) {
    # Validate if the input is a matrix
    if (!is.matrix(dataMatrix)) {
        stop("The input data must be a matrix.")
    }

    # Sampling a subset of the data for visualization
    sampledRows <- sample(1:nrow(dataMatrix), 50)
    sampledCols <- sample(1:ncol(dataMatrix), 20)
    dataMatrixSubset <- dataMatrix[sampledRows, sampledCols]

    # Converting the matrix to a long format data frame suitable for ggplot2
    dataLong <- reshape2::melt(dataMatrixSubset, varnames = c("Feature", "Sample"))

    # Creating the boxplot
    plot <- ggplot(dataLong, aes(x = as.factor(Sample), y = value)) +
        geom_boxplot(outlier.shape = NA) + # Hide outliers to avoid overplotting
        geom_jitter(width = 0.2, alpha = 0.5, size = 0.1) + # Add jitter to show actual points
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
        labs(x = "Sample", y = "Feature Value", title = "Feature Distribution per Sample")

    # Saving the plot
    ggsave(outputPath, plot, width = 8, height = 6)
}

# Function to create boxplots for feature distribution
createFeatureBoxplot <- function(expressionData, clinicalData, genes, outputPath) {
    # Combining expression and clinical data
    combinedData <- as.data.frame(cbind(t(expressionData[genes, ]), clinicalData$Response))
    colnames(combinedData) <- c(genes, "Response")

    # Reshaping the data to a long format for plotting (assuming this was the intent)
    longData <- tidyr::pivot_longer(combinedData, cols = genes, names_to = "Gene", values_to = "Expression")

    # Creating the boxplot
    plot <- ggplot(longData, aes(x = Gene, y = Expression)) +
        geom_boxplot(outlier.shape = NA) + # Hide outliers
        geom_jitter(width = 0.2, alpha = 0.5, size = 0.1) + # Add jitter for visibility
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + # Rotate x-axis labels
        labs(x = "Gene", y = "Expression Level", title = "Gene Expression by Response")

    # Saving the plot
    ggsave(outputPath, plot, width = 8, height = 6)
}

# Function to create boxplots for Gene Set Enrichment Analysis (GSEA)
plotGSEAResults <- function(metadata, scores, outputPath) {
    # Combining metadata and scores into a single data frame
    combinedData <- cbind(metadata, Score = as.numeric(scores))

    # Creating the boxplot with jitter for better visibility
    plot <- ggplot2::ggboxplot(combinedData,
        x = "Response", y = "Score", color = "Response",
        palette = c("#00AFBB", "#E7B800"), add = "jitter"
    ) +
        ggpubr::stat_compare_means(
            comparisons = list(c("R", "NR")), label = "p.label",
            size = 5, method = "wilcox.test"
        )

    # Saving the plot
    ggsave(outputPath, plot, width = 8, height = 6)
}

SelectMarkers_from_scRNA <- function(all_marker, FC_threshold = 1.5, P_threshold = 0.05, p.adj = FALSE, top_vars = NULL, savePath = NULL) {
    if (p.adj) {
        marker <- all_marker[all_marker$p_val_adj <= P_threshold, ]
    } else {
        marker <- all_marker[all_marker$p_val <= P_threshold, ]
    }

    # 根据logFC排序
    marker <- marker[order(marker$avg_log2FC, decreasing = TRUE), ]
    marker$gene <- rownames(marker)

    colnames(marker) <- c("p_val", "log2FoldChange", "pct.1", "pct.2", "padj", "gene")

    p <- createVolcanoPlot(
        marker,
        label_geneset = c(head(marker, 5)[, "gene"], tail(marker, 5)[, "gene"]), logFC = 0.5,
        savePath = savePath
    )

    write.csv(marker, paste0(strsplit(savePath, split = "\\.")[[1]][1], ".csv"), sep = ",", fileEncoding = "utf-8")


    NRmarkers <- marker[marker$log2FoldChange >= log2(FC_threshold), "gene"]
    Rmarkers <- marker[marker$log2FoldChange <= (-log2(FC_threshold)), "gene"]

    MarkerList <- list()
    MarkerList[["Highly"]] <- NRmarkers
    MarkerList[["Lowly"]] <- Rmarkers

    return(MarkerList)
}


# Function for Kaplan-Meier survival analysis visualization
plotSurvivalKM <- function(metadata, features, timeColumn, eventColumn, outputPath) {
    # Combining metadata and features into a single data frame
    combinedData <- cbind(metadata, Feature = as.numeric(features))

    # Determining the cutoff point for the feature
    cutoff <- survival::surv_cutpoint(
        data = combinedData, time = timeColumn,
        event = eventColumn, variables = "Feature"
    )$cutpoint[1, 1]
    combinedData$FeatureGroup <- ifelse(combinedData$Feature > cutoff, "High", "Low")

    # Performing survival analysis
    survivalFit <- survival::survfit(
        survival::Surv(
            time = combinedData[, timeColumn],
            event = combinedData[, eventColumn]
        ) ~ FeatureGroup,
        data = combinedData
    )

    # Creating the plot
    plot <- survminer::ggsurvplot(survivalFit, data = combinedData, pval = TRUE, risk.table = TRUE)

    # Saving the plot
    ggsave(outputPath, plot, width = 6, height = 6)
}

# Function to analyze and visualize scRank predicted subgroup distribution
analyzeScRankDistribution <- function(seuratObject, outputPath) {
    # Creating a frequency table
    freqTable <- table(seuratObject$seurat_clusters, seuratObject$Predict)
    df <- as.data.frame(freqTable)

    # Calculating cluster totals and proportions
    clusterTotals <- aggregate(Freq ~ Var1, data = df, sum)
    df <- merge(df, clusterTotals, by = "Var1")
    df$Proportion <- df$Freq.x / df$Freq.y
    df <- subset(df, Var2 == "Rank+")
    orderedClusters <- df$Var1[order(df$Proportion, decreasing = TRUE)]

    # Adjusting data frame for plotting
    df <- subset(df, Rank_Label != "Background")
    df$Cluster <- factor(df$Cluster, levels = orderedClusters)
    colnames(df) <- c("Cluster", "RankLabel", "Number", "TotalNumber", "Proportion")

    # Creating the bar plot
    plot <- ggplot(data = df, aes(x = Cluster, y = Proportion, fill = RankLabel)) +
        geom_bar(stat = "identity") +
        scale_fill_manual(values = c("Rank-" = "#4cb1c4", "Rank+" = "#b5182b"), limits = c("Rank+", "Rank-")) +
        theme_classic() +
        theme(panel.background = element_blank(), axis.line = element_line())

    # Saving the plot
    ggsave(outputPath, plot, width = 8, height = 6)
}
