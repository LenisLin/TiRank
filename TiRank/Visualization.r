# Functions for visualization
library(GSVA)
library(Seurat)
library(msigdbr)
library(fgsea)

library(ggplot2)
library(ggpubr)
library(ggtext)
library(ggrepel)
library(ggbeeswarm)

# library(ComplexHeatmap)
library(cowplot)
library(reshape2)
library(dplyr)
library(tidyr)

library(survival)
library(survminer)

## RNA to scRNA

# Function to create volcano plots for differential gene expression analysis
createVolcanoPlot <- function(deResults, FoldChangeThreshold = 1.5, pValueThreshold = 0.05, labeledGenes = NULL, outputPath) {
    # Classifying genes based on log fold change and p-value
    deResults$geneStatus <- "Unchanged"
    deResults$geneStatus[deResults$log2FoldChange >= log2(FoldChangeThreshold) & deResults$p_val <= pValueThreshold] <- "Upregulated"
    deResults$geneStatus[deResults$log2FoldChange <= -log2(FoldChangeThreshold) & deResults$p_val <= pValueThreshold] <- "Downregulated"

    # Setting plot limits
    xLimit <- max(abs(deResults$log2FoldChange))

    # Building the plot
    p <- ggplot(deResults, aes(x = log2FoldChange, y = -log10(p_val))) +
        geom_point(aes(color = geneStatus)) +
        theme_bw() +
        geom_vline(xintercept = c(-log2(FoldChangeThreshold), log2(FoldChangeThreshold)), lty = 2) +
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
    png(outputPath, width = 6, height = 6)
    print(plot)
    dev.off()
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
    png(outputPath, width = 6, height = 6)
    print(plot)
    dev.off()
}

# Function to create boxplots for Gene Set Enrichment Analysis (GSEA)
plotGSEAResults <- function(metadata, score, outputPath) {
    # Combining metadata and scores into a single data frame
    combinedData <- cbind(metadata, Score = as.numeric(score))

    # Creating the boxplot with jitter for better visibility
    plot <- ggboxplot(combinedData,
        x = "Response", y = "Score", color = "Response",
        palette = c("#00AFBB", "#E7B800"), add = "jitter"
    ) +
        stat_compare_means(
            comparisons = list(c("R", "NR")), label = "p.label",
            size = 5, method = "wilcox.test"
        )

    # Saving the plot
    png(outputPath, width = 300, height = 400)
    print(plot)
    dev.off()
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
        FoldChangeThreshold = FC_threshold, pValueThreshold = P_threshold,
        labeledGenes = c(head(marker, 5)[, "gene"], tail(marker, 5)[, "gene"]),
        outputPath = savePath
    )

    write.csv(marker, paste0(strsplit(savePath, split = "\\.")[[1]][1], ".csv"), sep = ",", fileEncoding = "utf-8")


    NRmarkers <- marker[marker$log2FoldChange >= log2(FC_threshold), "gene"]
    Rmarkers <- marker[marker$log2FoldChange <= (-log2(FC_threshold)), "gene"]

    if (!is.null(top_vars)) {
        if (length(NRmarkers)>top_vars) {
           NRmarkers <- NRmarkers[1:top_vars]
        }
        if (length(Rmarkers)>top_vars) {
        Rmarkers <- Rmarkers[(length(Rmarkers)-top_vars+1):length(Rmarkers)]
        }
    }

    MarkerList <- list()
    MarkerList[["Highly"]] <- NRmarkers
    MarkerList[["Lowly"]] <- Rmarkers

    return(MarkerList)
}


# Function for Kaplan-Meier survival analysis visualization
GroupingPatients <- function(metadata, features, timeColumn, eventColumn) {
    # Combining metadata and features into a single data frame
    if (!is.character(features)) {
        combinedData <- cbind(metadata, Feature = as.numeric(features))
        # Determining the cutoff point for the feature
        cutoff <- survminer::surv_cutpoint(
            data = combinedData, time = timeColumn,
            event = eventColumn, variables = "Feature"
        )$cutpoint[1, 1]
        combinedData$Group <- ifelse(combinedData$Feature > cutoff, "High", "Low")
    } else {
        combinedData <- cbind(metadata, Group = features)
    }

    return(combinedData)
}

plotSurvivalKM <- function(survivalFit, outputPath) {
    # Creating the plot
    plot <- ggsurvplot(survivalFit, data = combinedData, pval = TRUE, risk.table = TRUE)

    # Saving the plot
    png(outputPath, height = 400, width = 400)
    print(plot)
    dev.off()
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
    df_temp <- subset(df, Var2 == "Rank+")
    orderedClusters <- df_temp$Var1[order(df_temp$Proportion, decreasing = TRUE)]

    # Adjusting data frame for plotting
    colnames(df) <- c("Cluster", "RankLabel", "Number", "TotalNumber", "Proportion")
    df <- subset(df, RankLabel != "Background")
    df$Cluster <- factor(df$Cluster, levels = orderedClusters)

    # Creating the bar plot
    plot <- ggplot(data = df, aes(x = Cluster, y = Proportion, fill = RankLabel)) +
        geom_bar(stat = "identity") +
        scale_fill_manual(values = c("Rank-" = "#4cb1c4", "Rank+" = "#b5182b"), limits = c("Rank+", "Rank-")) +
        theme_classic() +
        theme(panel.background = element_blank(), axis.line = element_line())

    # Saving the plot
    ggsave(outputPath, plot, width = 8, height = 6)
}

plotGeneExpressionSwarm <- function(seurat_obj, gene_list, class_category) {
  # Extract data for the specified genes
  data <- FetchData(seurat_obj, vars = c("ident", gene_list))
  data <- data[data$ident %in% class_category,]

  # Melt data into a long format for ggplot
  long_data <- data %>%
    pivot_longer(cols = gene_list, names_to = "gene", values_to = "expression") %>%
    rename(class = ident)

  # Generate swarm plot
p <- ggplot(long_data, aes(x = class, y = expression, color = class)) +
    geom_violin(trim = FALSE) +
    stat_summary(fun = mean, geom = "crossbar", aes(group = interaction(class, gene)), 
                 color = "black", size = 1, width = 0.5) +  # Crossbar for mean value
    stat_compare_means(method = 'wilcox.test', label = 'p.signif', aes(label = ..p.format..), label.x = 1.5) +
    theme_minimal() +
    labs(title = "Gene Expression by Class", y = "Expression Level") +
    scale_color_manual(values = c("Rank+" = "red", "Rank-" = "blue")) +
    facet_wrap(~ gene)


  return(p)
}

## Analysis subclusters
analysis_subcluster <- function(seu, subclusters, cluster_name, ident.1, ident.2, figurePath){
    ## subset
    seu_ <- subset(seu, seurat_clusters %in% subclusters)
    Idents(seu_) <- seu_$Predict

    ## DEGs
    marker <- FindMarkers(seu_, ident.1 = ident.1, ident.2 = ident.2)
    marker_ <- subset(marker,p_val<=0.05)
    marker_ <- marker_[order(marker_$avg_log2FC, decreasing = TRUE), ]

    ## Plot DEGs
    colnames(marker) <- c("p_val", "log2FoldChange", "pct.1", "pct.2", "padj")
    marker$gene <- rownames(marker)
    marker <- marker[order(marker$log2FoldChange,decreasing = T),]
    p <- createVolcanoPlot(marker,
    FoldChangeThreshold = 1.5, pValueThreshold = 0.05,
    labeledGenes = c(head(rownames(marker), 5), tail(rownames(marker), 5)),
    outputPath = paste0(figurePath, "DEGs Volcano of ",cluster_name,".png"))

    ## Enrichment analysis
    GeneSet <- msigdbr(species = "Homo sapiens",category = "H") %>% dplyr::select(gs_name,gene_symbol)
    gs_name <- names(table(GeneSet$gs_name))
    GeneSet_list <- list()
    for(GeneSet_name in gs_name){
        GeneSet_list[[GeneSet_name]] <- subset(GeneSet,gs_name==GeneSet_name)$gene_symbol
    }

    degs <- marker_$avg_log2FC
    names(degs) <- rownames(marker_)

    gsvaResult <- fgsea(pathways = GeneSet_list, stats = degs,
                  minSize=15,maxSize=500,nperm=100000)

    # plotEnrichment(GeneSet_list[[head(gsvaResult[order(pval), ], 1)$pathway]],degs) + 
    #            labs(title=head(gsvaResult[order(pval), ], 1)$pathway)

    plot_gsva_results(gsvaResult,outputPath=paste0(figurePath, cluster_name," enrichment bubble of Hallmark.png"))

    return (seu_)
}

plot_gsva_results <- function(gsva_data, outputPath) {
  # Convert the pathway column to a factor for plotting
  gsva_data$pathway <- factor(gsva_data$pathway, levels = gsva_data$pathway)

  p <- ggplot(gsva_data, aes(x = NES, y = pathway)) +
    geom_point(aes(size = size, color = -log10(padj)), alpha = 0.6) +
    scale_color_gradient(low = "blue", high = "red", name = "-log10(Adj P-Value)") +
    theme_light(base_size = 12) +
    theme(axis.text.y = element_text(size = 8),
          plot.title = element_text(size = 14, face = "bold"),
          plot.subtitle = element_text(size = 12)) +
    labs(title = "Gene Set Variation Analysis (GSVA) Results",
         subtitle = "Pathways Ranked by Normalized Enrichment Score (NES)",
         x = "NES",
         y = "Pathway",
         size = "Gene Set Size") +
    guides(color = guide_colorbar(title.position = "top", title.hjust = 0.5))

    png(outputPath,width = 600,height = 800)
    print(p)
    dev.off()
   return(NULL)
}
