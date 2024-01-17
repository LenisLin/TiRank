library(Seurat)

library(ggplot2)
library(ggbeeswarm)
library(gghalves)
library(ggpubr)
library(ggrepel)
library(RColorBrewer)

library(corrplot)

library(dplyr)
library(tidyr)

slicesPath <- "/home/lenislin/Experiment/data/scRankv2/data/ST/GC_24/slices/"
savePath <- "./tempfiles_GC/"

slices <- list.files(slicesPath)
results <- list()
clinical <- read.csv("/home/lenislin/Experiment/data/scRankv2/data/ST/GC_24/GC_ST_Clinical.csv")

## Output the information of matrix
for (slice_ in slices) {
    # ## Load spatial slice
    # score <- read.csv(paste0(savePath, slice_, "/", "spot_predict_score.csv"), row.names = 1)
    # deconv <- read.csv(paste0(savePath, slice_, "/", "q05_cell_abundance_w_sf.csv"), row.names = 1)

    # stAnndata <- Load10X_Spatial(paste0(slicesPath, slice_))

    # ## shift the coordinates as number
    # if (class(stAnndata@images$slice1@coordinates[1, 2]) == class("a")) {
    #     coordinates <- apply(stAnndata@images$slice1@coordinates, MARGIN = 2, function(x) {
    #         return(as.numeric(x))
    #     })
    #     rownames(coordinates) <- rownames(stAnndata@images$slice1@coordinates)
    #     stAnndata@images$slice1@coordinates <- as.data.frame(coordinates)
    #     rm(coordinates)
    # }

    # ## Add information
    # stAnndata <- AddMetaData(stAnndata, score)
    # predScore <- score$Rank_Score

    # Rank_Label <- (1 - stAnndata$Reject) * stAnndata$Rank_Score
    # Rank_Label <- ifelse(Rank_Label == 0, "Background", ifelse(Rank_Label > 0.5, "scRank+", "scRank-"))
    # table(Rank_Label)
    # stAnndata <- AddMetaData(stAnndata, as.factor(Rank_Label), "Rank_Label")
    # # stAnndata <- AddMetaData(stAnndata, patho)
    # stAnndata <- AddMetaData(stAnndata, deconv)

    # ## Run dimension-reduction
    # stAnndata <- FindVariableFeatures(stAnndata, n_features = 2000)
    # stAnndata <- SCTransform(stAnndata, assay = "Spatial", verbose = FALSE)
    # stAnndata <- RunPCA(stAnndata, assay = "SCT", verbose = FALSE)
    # stAnndata <- FindNeighbors(stAnndata, reduction = "pca", dims = 1:20)
    # stAnndata <- FindClusters(stAnndata, verbose = FALSE)
    # stAnndata <- RunUMAP(stAnndata, reduction = "pca", dims = 1:20)

    # ## Spatial feature plot
    # Idents(stAnndata) <- as.factor(stAnndata$clusters)
    # p1 <- SpatialDimPlot(stAnndata, pt.size.factor = 0.9) + ggtitle("Gene cluster Label")
    # p1_ <- DimPlot(stAnndata) + ggtitle("Gene cluster Label")

    # # Idents(stAnndata) <- stAnndata$patho_class
    # # p2 <- SpatialDimPlot(stAnndata, pt.size.factor = 1.2) + ggtitle("Pathology Label")
    # # p2_ <- DimPlot(stAnndata) + ggtitle("Pathology Label")

    # Idents(stAnndata) <- stAnndata$Rank_Label
    # p3 <- SpatialDimPlot(stAnndata, pt.size.factor = 0.9) + ggtitle("scRank Label")
    # p3_ <- DimPlot(stAnndata) + ggtitle("scRank Label")

    # pdf(paste0(savePath, slice_, "/", "clusters plot on UMAP.pdf"), height = 6, width = 10)
    # print(p1_ + p3_)
    # dev.off()

    # pdf(paste0(savePath, slice_, "/", "clusters plot on spatial.pdf"), height = 6, width = 10)
    # print(p1 + p3)
    # dev.off()

    stAnndata <- results[[slice_]]
    ## Rank score feature plot
    stAnndata$Rank_Score_neg <- 1 - stAnndata$Rank_Score
    p <- SpatialFeaturePlot(stAnndata, features = "Rank_Score_neg", pt.size.factor = 0.75, alpha = c(0.25,0.75),image.alpha = 0) +
        scale_fill_gradient2(low = "#8C8CFF",mid = "#FFF6F6",high = "#FF8C8C",midpoint = 0.5) +
        theme(legend.position = "right")

    pdf(paste0(savePath, slice_, "/", "minus Rank score featureplot.pdf"), height = 6, width = 8)
    print(p)
    dev.off()

    p <- SpatialFeaturePlot(stAnndata, features = "Rank_Score", pt.size.factor = 0.75, alpha = c(0.25,0.75),image.alpha = 0) +
        scale_fill_gradient2(low = "#8C8CFF",mid = "#FFF6F6",high = "#FF8C8C",midpoint = 0.5) +
        theme(legend.position = "right")

    pdf(paste0(savePath, slice_, "/", "Rank score featureplot.pdf"), height = 6, width = 8)
    print(p)
    dev.off()


    ## HE plot
    # p <- SpatialFeaturePlot(stAnndata, features = "Rank_Score", alpha = c(0, 0)) + theme(legend.position = "None")

    # pdf(paste0(savePath, slice_, "/", "HE plot.pdf"), height = 6, width = 8)
    # print(p)
    # dev.off()

    # ## Celltypes fractions feature plot
    # celltypes <- colnames(deconv)
    # savePathTemp <- paste0(paste0(savePath, slice_, "/", "celltypes_fraction/"))
    # if (!dir.exists(savePathTemp)) {
    #     dir.create(savePathTemp, recursive = T)
    # }

    # for (celltype in celltypes) {
    #     p <- SpatialFeaturePlot(stAnndata, features = celltype, alpha = c(0.05, 1), pt.size.factor = 1) +
    #         theme(legend.position = "right")

    #     pdf(paste0(savePathTemp, celltype, " fraction featureplot.pdf"), height = 6, width = 8)
    #     print(p)
    #     dev.off()
    # }


    # results[[slice_]] <- stAnndata
}
saveRDS(results, paste0(savePath, "All Slices.rds"))

## visualize the score distribution
# Initialize an empty data frame
combined_data <- data.frame()

# Loop through each Seurat object in the results list
for (name in names(results)) {
    # Extract metadata
    metadata <- results[[name]]@meta.data

    # Add a column with the name of the Seurat object
    metadata$Sample <- name

    # Combine with the main data frame
    combined_data <- rbind(combined_data, metadata)
}

dim(combined_data)

# Bind clinical information
# colnames(clinical)[1] <- "Sample"
# combined_data <- left_join(combined_data, clinical, by = "Sample")

head(combined_data)
# write.csv(combined_data,paste0(savePath,"data_for_downstream.csv"))
combined_data <- read.csv(paste0(savePath, "data_for_downstream.csv"))

## compare the prediction results
## All zone stack barplot
if (1) {
    # Creating a frequency table
    freqTable <- table(combined_data$Sample, combined_data$Rank_Label)
    df <- as.data.frame(freqTable)

    # merge label
    colnames(df) <- c("Sample", "scRank_label", "Freq")
    # df <- Match_Pathology_label(df = df, merge_rules = merge_rules)

    # Calculating cluster totals and proportions
    clusterTotals <- aggregate(Freq ~ Sample, data = df, sum)
    df <- merge(df, clusterTotals, by = "Sample")
    df$Proportion <- df$Freq.x / df$Freq.y

    # Calculate the entropy
    regions <- as.character(clusterTotals[, 1])
    i <- 1
    entrodf <- matrix(data = NA, nrow = length(regions), ncol = 1)
    for (region in regions) {
        sub_df <- subset(df, Sample == region)
        x <- as.numeric(sub_df[, ncol(sub_df)]) + 1e-16
        entropy_value <- (-(x %*% log10(x)))
        entrodf[i, 1] <- entropy_value
        i <- i + 1

        rownames(entrodf) <- regions
        colnames(entrodf) <- "entropy"
    }

    # Order
    colnames(df) <- c("Sample", "RankLabel", "Number", "TotalNumber", "Proportion")
    df <- df[order(df$Sample), ]
    orderedClusters <- unique(df$Sample)
    df$Sample <- factor(df$Sample, levels = orderedClusters)

    # df$NormalizedEntropy <- entrodf[match(df$Region,rownames(entrodf)),1]

    ## Bar Plot
    plot <- ggplot(data = df, aes(x = Sample, y = Proportion, fill = RankLabel)) +
        geom_bar(stat = "identity", position = position_stack()) +
        # geom_text(aes(label = sprintf("%.2f", NormalizedEntropy), y = 1.05), vjust = -0.5) + # Display entropy as text
        scale_fill_manual(values = c("scRank-" = "#4cb1c4", "scRank+" = "#b5182b", "Background" = "grey"), limits = c("scRank+", "scRank-", "Background")) +
        theme_classic() +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 10),
            panel.background = element_blank(),
            axis.line = element_line()
        )

    # Saving the plot
    ggsave(paste0(savePath, "Rank label distribution in Samples.pdf"), plot, width = 6, height = 6)


    ## Boxplot show the proportion difference between each Rank category
    df$PGroup <- ifelse(startsWith(as.character(df$Sample), "N"), "Neg", "Pos")
    plot <- ggplot(data = df, aes(x = RankLabel, y = Proportion, fill = PGroup)) +
        geom_boxplot() +
        # geom_text(aes(label = sprintf("%.2f", NormalizedEntropy), y = 1.05), vjust = -0.5) + # Display entropy as text
        theme_classic() +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 10),
            panel.background = element_blank(),
            axis.line = element_line()
        ) +
        stat_compare_means(method = "t.test")

    # Saving the plot
    ggsave(paste0(savePath, "Rank label distribution in Groups boxplot.pdf"), plot, width = 6, height = 6)

    ## Boxplot of the ratio
    plotdf <- df %>%
        group_by(Sample) %>%
        summarize(
            scRankPlusProportion = sum(Proportion[scRank_label == "scRank+"]),
            OtherProportion = sum(Proportion[scRank_label != "scRank+"]),
            Ratio = scRankPlusProportion / OtherProportion
        )
    plotdf <- as.data.frame(plotdf)
    plotdf$PGroup <- ifelse(startsWith(as.character(plotdf$Sample), "N"), "Neg", "Pos")

    plot <- ggplot(data = plotdf, aes(x = PGroup, y = Ratio, fill = PGroup)) +
        geom_boxplot() +
        theme_classic() +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 10),
            panel.background = element_blank(),
            axis.line = element_line()
        ) +
        stat_compare_means(method = "t.test")

    # Saving the plot
    ggsave(paste0(savePath, "Rank label ratio in Groups boxplot.pdf"), plot, width = 6, height = 6)

    ## Compare Rank score
    df <- combined_data
    plotdf <- df %>%
        group_by(Sample) %>%
        summarise(across(c(1:(ncol(df) - 1)), mean, na.rm = TRUE))
    plotdf <- as.data.frame(plotdf)
    plotdf <- plotdf[, c("Rank_Score", "Sample")]
    plotdf$PGroup <- ifelse(startsWith(as.character(plotdf$Sample), "N"), "Neg", "Pos")

    plot <- ggplot(data = plotdf, aes(x = PGroup, y = Rank_Score, fill = PGroup)) +
        geom_boxplot() +
        # geom_text(aes(label = sprintf("%.2f", NormalizedEntropy), y = 1.05), vjust = -0.5) + # Display entropy as text
        theme_classic() +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 10),
            panel.background = element_blank(),
            axis.line = element_line()
        ) +
        stat_compare_means(method = "t.test")

    # Saving the plot
    ggsave(paste0(savePath, "Rank score in Groups boxplot.pdf"), plot, width = 4, height = 4)
}

## Volcanno plot of differential celltypes in scRank+ and scRank-
if (1) {
    df <- combined_data
    df$PGroup <- ifelse(startsWith(as.character(df$Sample), "N"), "Neg", "Pos")
    head(df)

    ## Celltypes
    celltypes <- colnames(df)[14:47]

    frac_df <- df[, match(c(celltypes, "Rank_Label"), colnames(df))]
    # frac_df <- subset(frac_df, Rank_Label != "Background")
    frac_df$Rank_Label <- ifelse(frac_df$Rank_Label == "scRank+", 1, 0)

    result_df <- FCandPvalueCal(mat = frac_df, xCol = c(1, (ncol(frac_df) - 1)), yCol = ncol(frac_df))
    VolcanoPlot(result_df,
        pthreshold = 0.05, fcthreshold = 1.4,
        filename = paste0(savePath, "Volcano of Celltypes fraction difference of Rank_pos versus Rank_neg.pdf")
    )
}

## Volcanno plot of differential genes in scRank+ and scRank-
if (1) {
    for (slice_ in names(results)) {
        stAnndata_ <- results[[slice_]]
        df <- as.matrix(stAnndata_@assays$SCT@data)
        df <- as.data.frame(t(df))
        df$Rank_Label <- as.character(stAnndata_@meta.data$Rank_Label)
        df$Rank_Label <- ifelse(df$Rank_Label == "scRank+", 1, 0)

        result_df <- FCandPvalueCal(mat = df, xCol = c(1, (ncol(df) - 1)), yCol = ncol(df))
        VolcanoPlot(result_df,
            pthreshold = 0.05, fcthreshold = 2,
            filename = paste0(savePath, slice_, "/", "Volcano of HVG expression difference of Rank_pos versus Rank_neg.pdf")
        )
    }
}

## Focus on scRank- spots
if (1) {
    for (slice_ in names(results)) {
        stAnndata_ <- results[[slice_]]

        ## Subset
        stAnndata_ <- stAnndata_[, stAnndata_$Rank_Label %in% c("scRank-", "Background")]
        stAnndata_$Rank_Label <- as.character(stAnndata_$Rank_Label)
        if (dim(stAnndata_)[2] < 1) {
            cat("The slice", slice_, "did not contain interesting region!", "\n")
            next
        }

        ## Rank score feature plot
        p1 <- SpatialFeaturePlot(stAnndata_, features = ("Rank_Score"), pt.size.factor = 0.75) +
            scale_fill_gradient(low = "red", high = "white") +
            theme(legend.position = "top")

        pdf(paste0(savePath, slice_, "/", "scRank- spots spatial plot.pdf"), height = 6, width = 6)
        print(p1)
        dev.off()

        p4 <- SpatialFeaturePlot(stAnndata_, features = "Rank_Score", alpha = c(0, 0)) + theme(legend.position = "None")
        pdf(paste0(savePath, slice_, "/", "scRank- correspond HE plot.pdf"), height = 6, width = 6)
        print(p4)
        dev.off()
    }
}

## Correlation in zoom region and certain label
if (1) {
    ## all celltypes
    print(colnames(combined_data))

    ## define the interesting pairs of celltype
    types1 <- colnames(combined_data)[15:48]
    types2 <- types1
    # types1 <- c("SPP1.A", "Pro.inflammatory", "SPP1.B", "Anti.inflammatory") ## TAM
    # types2 <- c("Pericytes", "Stromal.2", "Stromal.3", "Myofibroblasts") ##

    ## compare the immune related cell subpopulation deconvolution abundance in different zone
    ### !Rank+
    if (T) {
        row_idx <- (combined_data$Rank_Label != "scRank+")
        col_idx <- match(c("Sample", unique(c(types1, types2))), colnames(combined_data))
        plotdf <- combined_data[row_idx, col_idx]
        # dim(plotdf)

        ## Take the mean
        plotdf <- plotdf %>%
            group_by(Sample) %>%
            summarise(across(c(1:(ncol(plotdf) - 2)), mean, na.rm = TRUE))
        plotdf <- as.data.frame(plotdf)

        mat <- as.matrix(plotdf[, 2:ncol(plotdf)])
        tdc <- cor(mat, method = c("spearman"))
        testRes <- cor.mtest(mat, method = "spearman", conf.level = 0.95)

        addcol <- colorRampPalette(c("blue", "white", "red"))

        pdf(paste0(savePath, "Types abundance correlation in scRank- region.pdf"), height = 15, width = 15)
        corrplot(tdc,
            method = "color", col = addcol(100),
            tl.col = "black", tl.cex = 0.8, tl.srt = 45, tl.pos = "lt",
            p.mat = testRes$p, diag = T, type = "upper",
            sig.level = c(0.001, 0.01, 0.05), pch.cex = 1.2,
            insig = "label_sig", pch.col = "grey20", order = "AOE"
        )
        corrplot(tdc,
            method = "number", type = "lower", col = addcol(100),
            tl.col = "n", tl.cex = 0.5, tl.pos = "n", order = "AOE",
            add = T
        )
        dev.off()
    }

    ### Rank+
    if (T) {
        row_idx <- (combined_data$Rank_Label == "scRank+")
        col_idx <- match(c("Sample", unique(c(types1, types2))), colnames(combined_data))
        plotdf <- combined_data[row_idx, col_idx]
        # dim(plotdf)

        ## Take the mean
        plotdf <- plotdf %>%
            group_by(Sample) %>%
            summarise(across(c(1:(ncol(plotdf) - 2)), mean, na.rm = TRUE))
        plotdf <- as.data.frame(plotdf)

        mat <- as.matrix(plotdf[, 2:ncol(plotdf)])
        tdc <- cor(mat, method = c("spearman"))
        testRes <- cor.mtest(mat, method = "spearman", conf.level = 0.95)

        addcol <- colorRampPalette(c("blue", "white", "red"))

        pdf(paste0(savePath, "Types abundance correlation in scRank+ region.pdf"), height = 15, width = 15)
        corrplot(tdc,
            method = "color", col = addcol(100),
            tl.col = "black", tl.cex = 0.8, tl.srt = 45, tl.pos = "lt",
            p.mat = testRes$p, diag = T, type = "upper",
            sig.level = c(0.001, 0.01, 0.05), pch.cex = 1.2,
            insig = "label_sig", pch.col = "grey20", order = "AOE"
        )
        corrplot(tdc,
            method = "number", type = "lower", col = addcol(100),
            tl.col = "n", tl.cex = 0.5, tl.pos = "n", order = "AOE",
            add = T
        )
        dev.off()
    }
}


## Compare the correlation of given pairs of celltype in Rank+ and Rank- spots
if (1) {
    df <- combined_data
    df <- df[df$Pathologist.Annotation != "", ]
    head(df)

    df <- subset(df, Pathologist.Annotation %in% interesting_region)
    celltypes <- colnames(df)[15:54]

    ## define the interesting pairs of celltype
    types1 <- c("SPP1.A", "Pro.inflammatory", "SPP1.B", "Anti.inflammatory") ## TAM
    types2 <- c("Pericytes", "Stromal.2", "Stromal.3", "Myofibroblasts")

    celltypes_pair <- combn_types_pair(types1, types2)

    ## self-correlation
    result_df <- self_correlation(df, celltypes_pair)

    ## boxplot
    result_df$Rank_Label <- as.factor(result_df$Rank_Label)
    # result_df$Coef <- as.numeric(result_df$Coef)
    result_df$Cooc_Score <- as.numeric(result_df$Cooc_Score)

    my_comparisons <- list(
        c("scRank+", "scRank-"),
        c("scRank+", "Background"),
        c("Background", "scRank-")
    )

    p <- ggplot(result_df, aes(x = Rank_Label, y = Cooc_Score, fill = Rank_Label)) +
        geom_violin(trim = FALSE) + # Draw violin plots
        geom_jitter(width = 0.2, size = 1, alpha = 0.5, color = "black") + # Add jittered points for individual data representation
        scale_fill_brewer(palette = "Set1") + # Use a color palette for better visual distinction
        theme_minimal() + # Use a minimal theme for a cleaner look
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
            axis.text.y = element_text(size = 10),
            legend.position = "none", # Hide legend if not needed
            plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
            plot.subtitle = element_text(hjust = 0.5, size = 12)
        ) +
        labs(
            title = "Risk Score Distribution Across Samples",
            subtitle = "Violin plots with jittered data points",
            x = "Sample",
            y = "Risk Score"
        ) +
        stat_compare_means(comparisons = my_comparisons) +
        facet_grid(Pair ~ ., scales = "free")

    pdf(paste0(savePath, "test.pdf"), height = 18, width = 4)
    print(p)
    dev.off()
}

## Plot the gene expression value and celltypes abundance in certain HE region
if (1) {
    all_slice_seurat <- readRDS("/home/lenislin/Experiment/projects/scRankv2/bulk2st/tempfiles/All Slices.rds")
    slices <- list.files(slicesPath)

    interesting_region <- c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC low", "stroma_fibroblastic_IC med")
    interesting_types <- c("SPP1.B", "Anti.inflammatory", "Stromal.2", "Myofibroblasts")
    interesting_genes <- c("ACTA2", "COL3A1", "SPP1", "CD163")

    for (slice_ in slices) {
        stAnndata <- all_slice_seurat[[slice_]]
        # stAnndata
        savePath1 <- paste0(savePath, slice_, "/")
        if (!dir.exists(savePath1)) {
            dir.create(savePath1, recursive = T)
        }

        ## Subset
        stAnndata_ <- stAnndata[, stAnndata$`Pathologist Annotation` %in% interesting_region]
        stAnndata_$Rank_Label <- as.character(stAnndata_$Rank_Label)
        if (dim(stAnndata_)[2] < 1) {
            cat("The slice", slice_, "did not contain interesting region!", "\n")
            next
        }

        ## celltypes
        savePath2 <- paste0(savePath1, "interestingTypes", "/")
        if (!dir.exists(savePath2)) {
            dir.create(savePath2, recursive = T)
        }
        for (celltype in interesting_types) {
            p <- SpatialFeaturePlot(stAnndata_, features = (celltype), pt.size.factor = 1.2) +
                scale_fill_gradient(low = "white", high = "red") +
                theme(legend.position = "top")

            pdf(paste0(savePath2, celltype, " abundance featureplot.pdf"), height = 6, width = 6)
            print(p)
            dev.off()
        }


        ## features
        savePath3 <- paste0(savePath1, "interestingGenes", "/")
        if (!dir.exists(savePath3)) {
            dir.create(savePath3, recursive = T)
        }
        for (feature in interesting_genes) {
            p <- SpatialFeaturePlot(stAnndata_, features = feature, pt.size.factor = 1.2) +
                scale_fill_gradient(low = "white", high = "red") +
                theme(legend.position = "top")

            pdf(paste0(savePath3, feature, " expression featureplot.pdf"), height = 6, width = 6)
            print(p)
            dev.off()
        }
    }
}

## Where is the Spp1+ cell and CAF?
if (F) {
    celltypes <- colnames(combined_data)[15:54]
    SPP1_types <- c("SPP1.A", "SPP1.B")
    CAF_types <- c("Myofibroblasts", "Pericytes", "Proliferating") # "Stromal.2", "Stromal.3"

    df <- combined_data[, match(c("Sample", "Pathologist.Annotation", "Pathologist.Annotation2", SPP1_types, CAF_types), colnames(combined_data))]
    df <- df[df$Pathologist.Annotation != "", ]
    head(df)

    plotdf <- pivot_longer(df, cols = 4:ncol(df), values_to = "Fraction", names_to = "Celltypes")
    plotdf <- as.data.frame(plotdf)

    color_vec <- c(brewer.pal(9, "Pastel1"), brewer.pal(8, "Pastel2"))

    plotdf$Pathologist.Annotation <- factor(plotdf$Pathologist.Annotation,
        levels = unique(plotdf$Pathologist.Annotation[order(plotdf$Pathologist.Annotation2)])
    )

    p <- ggplot(plotdf, aes(x = Pathologist.Annotation, y = Fraction, fill = Pathologist.Annotation)) +
        geom_violin(trim = FALSE) +
        geom_jitter(width = 0.2, size = 0.1, alpha = 0.5, color = "grey") +
        scale_fill_manual(values = color_vec) +
        theme_minimal() +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 10),
            axis.text.y = element_text(size = 10),
            plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
            plot.subtitle = element_text(hjust = 0.5, size = 12)
        ) +
        facet_grid(Celltypes ~ .)

    pdf(paste0(savePath, "SPP1 and Fibroblast of histology group.pdf"), height = 8, width = 10)
    print(p)
    dev.off()
}

## Zoom in certain HE region
if (F) {
    all_slice_seurat <- readRDS("/home/lenislin/Experiment/projects/scRankv2/bulk2st/tempfiles/All Slices.rds")
    slices <- list.files(slicesPath)

    interesting_region <- c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC low", "stroma_fibroblastic_IC med")

    for (slice_ in slices) {
        stAnndata <- all_slice_seurat[[slice_]]
        # stAnndata

        if (!dir.exists(paste0(savePath, slice_, "/"))) {
            dir.create(paste0(savePath, slice_, "/"), recursive = T)
        }

        ## Plot the interesting region
        Idents_ <- stAnndata@meta.data["Pathologist Annotation"][, 1]
        Idents(stAnndata) <- as.factor(Idents_)
        p <- SpatialDimPlot(stAnndata, cells.highlight = CellsByIdentities(
            object = stAnndata, idents = interesting_region
        ), pt.size.factor = 1, alpha = c(0.5, 1), facet.highlight = TRUE)

        pdf(paste0(savePath, slice_, "/", "Interesting Region.pdf"), height = 6, width = 8)
        print(p)
        dev.off()

        ## Subset
        stAnndata_ <- stAnndata[, stAnndata$`Pathologist Annotation` %in% interesting_region]
        stAnndata_$Rank_Label <- as.character(stAnndata_$Rank_Label)
        if (dim(stAnndata_)[2] < 1) {
            cat("The slice", slice_, "did not contain interesting region!", "\n")
            next
        }

        ## Rank score feature plot
        p1 <- SpatialFeaturePlot(stAnndata_, features = ("Rank_Score"), pt.size.factor = 1.2) +
            scale_fill_gradient(low = "white", high = "red") +
            theme(legend.position = "top")

        Idents(stAnndata_) <- stAnndata_$Rank_Label
        p2 <- SpatialDimPlot(stAnndata_, pt.size.factor = 1.2) +
            scale_fill_manual(values = c("scRank+" = "red", "scRank-" = "blue", "Background" = "grey")) +
            theme(legend.position = "top")

        Idents(stAnndata_) <- stAnndata_$`Pathologist Annotation`
        p3 <- SpatialDimPlot(stAnndata_, pt.size.factor = 1.2) +
            theme(legend.position = "top")


        pdf(paste0(savePath, slice_, "/", "Interesting Region Rank score featureplot.pdf"), height = 6, width = 15)
        print(p1 + p2 + p3)
        dev.off()

        p4 <- SpatialFeaturePlot(stAnndata_, features = "Rank_Score", alpha = c(0, 0)) + theme(legend.position = "None")
        pdf(paste0(savePath, slice_, "/", "Interesting Region HE plot.pdf"), height = 6, width = 6)
        print(p4)
        dev.off()
    }
}


## compare cell subpopulations fraction in different hitopathology region
if (1) {
    ## compare groups
    compare_groups <- list()
    compare_groups[["stroma_fibroblastic_IC"]] <- list(
        c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC low"),
        c("stroma_fibroblastic_IC med", "stroma_fibroblastic_IC low"),
        c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC med")
    )

    compare_groups[["stroma_desmoplastic_IC"]] <- list(
        c("stroma_desmoplastic_IC med to high", "stroma_desmoplastic_IC low")
    )

    compare_groups[["tumor&stroma_IC"]] <- list(
        c("tumor&stroma", "tumor&stroma_IC low"),
        c("tumor&stroma", "tumor&stroma_IC med to high"),
        c("tumor&stroma_IC low", "tumor&stroma_IC med to high")
    )

    compare_groups[["tissue"]] <- list(
        c("tumor", "submucosa"),
        c("tumor", "tumor&stroma"),
        c("tumor&stroma", "submucosa"),
        c("submucosa", "epithelium&submucosa"),
        c("epithelium&submucosa", "non neo epithelium")
    )

    compare_groups[["submucosa"]] <- list(
        c("submucosa", "IC aggregate_submucosa")
    )

    for (i in 1:length(compare_groups)) {
        my_comparisons <- compare_groups[[i]]
        group_name <- names(compare_groups)[i]

        ## compare the immune related cell subpopulation deconvolution abundance in different zone
        row_idx <- (combined_data$Pathologist.Annotation %in% unique(unlist(my_comparisons)))
        col_idx <- match(c("Sample", "Pathologist.Annotation", colnames(combined_data)[15:54]), colnames(combined_data))
        plotdf <- combined_data[row_idx, col_idx]
        # dim(plotdf)

        ## Take the mean
        plotdf <- plotdf %>% pivot_longer(cols = 3:ncol(plotdf), names_to = "Celltypes", values_to = "Fraction")
        plotdf <- as.data.frame(plotdf)

        plotdf <- plotdf %>%
            group_by(Sample, Pathologist.Annotation, Celltypes) %>%
            summarise(across(c(1:(ncol(plotdf) - 3)), mean, na.rm = TRUE))
        plotdf <- as.data.frame(plotdf)

        p <- ggplot(plotdf, aes(x = Pathologist.Annotation, y = Fraction, fill = Pathologist.Annotation)) +
            geom_violin(trim = FALSE) +
            geom_jitter(width = 0.2, size = 1, alpha = 0.5, color = "black") +
            scale_fill_brewer(palette = "Set1") +
            theme_minimal() +
            theme(
                axis.text.x = element_text(angle = 30, hjust = 0.5, vjust = 0.5, size = 10),
                axis.text.y = element_text(size = 10),
                legend.position = "none",
                plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
                plot.subtitle = element_text(hjust = 0.5, size = 12)
            ) +
            facet_wrap(~Celltypes, ncol = 6, scales = "free") +
            stat_compare_means(comparisons = my_comparisons)

        pdf(paste0(savePath, "Abundance difference of group ", group_name, ".pdf"), height = 25, width = 15)
        print(p)
        dev.off()
    }
}

## Compare with Rank value
if (1) {
    ## compare groups
    compare_groups <- list()
    compare_groups[["stroma_fibroblastic_IC"]] <- list(
        c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC low"),
        c("stroma_fibroblastic_IC med", "stroma_fibroblastic_IC low"),
        c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC med")
    )

    compare_groups[["stroma_desmoplastic_IC"]] <- list(
        c("stroma_desmoplastic_IC med to high", "stroma_desmoplastic_IC low")
    )

    compare_groups[["tumor&stroma_IC"]] <- list(
        c("tumor&stroma", "tumor&stroma_IC low"),
        c("tumor&stroma", "tumor&stroma_IC med to high"),
        c("tumor&stroma_IC low", "tumor&stroma_IC med to high")
    )

    compare_groups[["tissue"]] <- list(
        c("tumor", "submucosa"),
        c("tumor", "tumor&stroma"),
        c("tumor&stroma", "submucosa"),
        c("submucosa", "epithelium&submucosa"),
        c("epithelium&submucosa", "non neo epithelium")
    )

    compare_groups[["submucosa"]] <- list(
        c("submucosa", "IC aggregate_submucosa")
    )

    combined_data_ <- combined_data[combined_data$Rank_Label != "Background", ]

    for (i in 1:length(compare_groups)) {
        my_comparisons <- compare_groups[[i]]
        group_name <- names(compare_groups)[i]

        ## compare the immune related cell subpopulation deconvolution abundance in different zone
        row_idx <- (combined_data_$Pathologist.Annotation %in% unique(unlist(my_comparisons)))
        col_idx <- match(c("Sample", "Pathologist.Annotation", "Rank_Score"), colnames(combined_data_))
        plotdf <- combined_data_[row_idx, col_idx]
        # dim(plotdf)

        ## Take the mean
        plotdf <- plotdf %>%
            group_by(Sample, Pathologist.Annotation) %>%
            summarise(across(c(1:(ncol(plotdf) - 2)), mean, na.rm = TRUE))

        plotdf <- as.data.frame(plotdf)
        plotdf$Pathologist.Annotation <- as.factor(plotdf$Pathologist.Annotation)

        p <- ggplot(plotdf, aes(x = Pathologist.Annotation, y = Rank_Score, fill = Pathologist.Annotation)) +
            geom_violin(trim = FALSE) +
            geom_jitter(width = 0.2, size = 1, alpha = 0.5, color = "black") +
            scale_fill_brewer(palette = "Set1") +
            theme_minimal() +
            theme(
                axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
                axis.text.y = element_text(size = 10),
                legend.position = "none",
                plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
                plot.subtitle = element_text(hjust = 0.5, size = 12)
            ) +
            stat_compare_means(comparisons = my_comparisons)

        pdf(paste0(savePath, "Risk Score difference of group ", group_name, ".pdf"), height = 6, width = 4)
        print(p)
        dev.off()
    }
}

## Compare with Rank label
if (1) {
    ## compare groups
    compare_groups <- list()
    compare_groups[["stroma_fibroblastic_IC"]] <- list(
        c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC low"),
        c("stroma_fibroblastic_IC med", "stroma_fibroblastic_IC low"),
        c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC med")
    )

    compare_groups[["stroma_desmoplastic_IC"]] <- list(
        c("stroma_desmoplastic_IC med to high", "stroma_desmoplastic_IC low")
    )

    compare_groups[["tumor&stroma_IC"]] <- list(
        c("tumor&stroma", "tumor&stroma_IC low"),
        c("tumor&stroma", "tumor&stroma_IC med to high"),
        c("tumor&stroma_IC low", "tumor&stroma_IC med to high")
    )

    compare_groups[["tissue"]] <- list(
        c("tumor", "submucosa"),
        c("tumor", "tumor&stroma"),
        c("tumor&stroma", "submucosa"),
        c("submucosa", "epithelium&submucosa"),
        c("epithelium&submucosa", "non neo epithelium")
    )

    compare_groups[["submucosa"]] <- list(
        c("submucosa", "IC aggregate_submucosa")
    )

    combined_data_ <- combined_data

    for (i in 1:length(compare_groups)) {
        my_comparisons <- compare_groups[[i]]
        group_name <- names(compare_groups)[i]

        ## compare the immune related cell subpopulation deconvolution abundance in different zone
        row_idx <- (combined_data_$Pathologist.Annotation %in% unique(unlist(my_comparisons)))
        col_idx <- match(c("Sample", "Pathologist.Annotation", "Rank_Label"), colnames(combined_data_))
        plotdf <- combined_data_[row_idx, col_idx]
        # dim(plotdf)

        ## Take the mean
        plotdf <- plotdf %>%
            group_by(Sample, Pathologist.Annotation, Rank_Label) %>%
            summarise(Count = n()) %>%
            mutate(Fraction = Count / sum(Count))

        plotdf <- as.data.frame(plotdf)

        for (ranklabel in c("scRank-", "scRank+")) {
            plotdf_ <- plotdf[plotdf$Rank_Label == ranklabel, ]
            plotdf_$Pathologist.Annotation <- as.factor(plotdf_$Pathologist.Annotation)

            p <- ggplot(plotdf_, aes(x = Pathologist.Annotation, y = Fraction, fill = Pathologist.Annotation)) +
                geom_violin(trim = FALSE) +
                geom_jitter(width = 0.2, size = 1, alpha = 0.5, color = "black") +
                scale_fill_brewer(palette = "Set1") +
                theme_minimal() +
                theme(
                    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
                    axis.text.y = element_text(size = 10),
                    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
                    plot.subtitle = element_text(hjust = 0.5, size = 12),
                    legend.position = "None"
                ) +
                stat_compare_means(comparisons = my_comparisons)

            pdf(paste0(savePath, ranklabel, " fraction difference of group ", group_name, ".pdf"), height = 6, width = 4)
            print(p)
            dev.off()
        }
    }
}

## Correlation
if (1) {
    ## compare the immune related cell subpopulation deconvolution abundance in different zone
    row_idx <- (combined_data$Rank_Label != "Background")
    col_idx <- match(c("Sample", "Pathologist.Annotation", "Rank_Score", colnames(combined_data)[15:54]), colnames(combined_data))
    plotdf <- combined_data[row_idx, col_idx]
    # dim(plotdf)

    ## Take the mean
    plotdf <- plotdf %>%
        group_by(Sample, Pathologist.Annotation) %>%
        summarise(across(c(1:(ncol(plotdf) - 2)), mean, na.rm = TRUE))
    plotdf <- as.data.frame(plotdf)

    mat <- as.matrix(plotdf[, 3:ncol(plotdf)])
    library(corrplot)
    tdc <- cor(mat, method = c("spearman"))
    testRes <- cor.mtest(mat, method = "spearman", conf.level = 0.95)

    addcol <- colorRampPalette(c("blue", "white", "red"))

    pdf(paste0(savePath, "Rank score and types abundance correlation.pdf"), height = 15, width = 20)
    corrplot(tdc,
        method = "color", col = addcol(100),
        tl.col = "black", tl.cex = 0.8, tl.srt = 45, tl.pos = "lt",
        p.mat = testRes$p, diag = T, type = "upper",
        sig.level = c(0.001, 0.01, 0.05), pch.cex = 1.2,
        insig = "label_sig", pch.col = "grey20", order = "AOE"
    )
    corrplot(tdc,
        method = "number", type = "lower", col = addcol(100),
        tl.col = "n", tl.cex = 0.5, tl.pos = "n", order = "AOE",
        add = T
    )
    dev.off()
}

## Compare Rank label Counts diff in each histology zone
if (1) {
    combined_data_ <- combined_data[combined_data$Rank_Label != "Background", ]

    ## compare the immune related cell subpopulation deconvolution abundance in different zone
    row_idx <- (!combined_data_$Pathologist.Annotation %in% c("", "tumor&stroma"))
    col_idx <- match(c("Sample", "Pathologist.Annotation", "Rank_Label"), colnames(combined_data_))
    plotdf <- combined_data_[row_idx, col_idx]
    # dim(plotdf)

    ## Take the mean
    plotdf <- plotdf %>%
        group_by(Sample, Pathologist.Annotation, Rank_Label) %>%
        summarise(Count = n()) %>%
        mutate(Fraction = Count / sum(Count))

    plotdf <- as.data.frame(plotdf)
    plotdf <- plotdf[plotdf$Rank_Label != "Background", ]

    plotdf$Pathologist.Annotation <- as.factor(plotdf$Pathologist.Annotation)

    p <- ggplot(plotdf, aes(x = Pathologist.Annotation, y = Fraction, fill = Rank_Label)) +
        geom_violin(trim = FALSE) +
        geom_jitter(width = 0.2, size = 1, alpha = 0.5, color = "black") +
        scale_fill_brewer(palette = "Set1") +
        theme_minimal() +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
            axis.text.y = element_text(size = 10),
            plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
            plot.subtitle = element_text(hjust = 0.5, size = 12)
        ) +
        stat_compare_means(
            label = "p.format"
        )

    pdf(paste0(savePath, "Risk Label Counts difference of histology group.pdf"), height = 4, width = 12)
    print(p)
    dev.off()
}


# Plotting
p <- ggplot(combined_data, aes(x = Sample_number, y = Rank_Score, fill = Sample_number)) +
    geom_violin(trim = FALSE) + # Draw violin plots
    geom_jitter(width = 0.2, size = 1, alpha = 0.5, color = "black") + # Add jittered points for individual data representation
    scale_fill_brewer(palette = "Set1") + # Use a color palette for better visual distinction
    theme_minimal() + # Use a minimal theme for a cleaner look
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
        axis.text.y = element_text(size = 10),
        legend.position = "none", # Hide legend if not needed
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 12)
    ) +
    labs(
        title = "Risk Score Distribution Across Samples",
        subtitle = "Violin plots with jittered data points",
        x = "Sample",
        y = "Risk Score"
    )

pdf("test.pdf", height = 6, width = 10)
print(p)
dev.off()

# The order of tumor invasion in colorectal cancer (CRC) typically follows a progression through the layers of the colon or rectum wall.
# This progression is crucial for staging the cancer and determining treatment approaches. Here's a general order of CRC tumor invasion:

# Mucosa: This is the innermost layer of the colon. Initially, CRC may begin as small, benign clumps of cells called polyps.
# Over time, some of these polyps can become cancerous, invading the mucosal layer.

# Submucosa: After invading the mucosa, the next layer cancer cells typically invade is the submucosa.
# This layer contains blood vessels, nerves, and glands.

# Muscularis Propria: This is the thick layer of muscle that helps in the movement of the colon and rectum.
# Invasion into the muscularis propria indicates a deeper, potentially more serious spread of cancer.

# Subserosa and Serosa: The subserosa is a layer of connective tissue, and the serosa is the outermost layer of the intestines.
# In the colon, the serosa covers only part of the organ, while the rectum does not have a serosa.
# Involvement of these layers often indicates advanced disease and potential spread beyond the colon or rectum.

# Adjacent Structures: The cancer may then spread to nearby organs or structures, such as the bladder, uterus, or the abdominal wall.

# Lymph Nodes: The involvement of nearby lymph nodes is a common route for the spread of CRC.
# The number and location of affected lymph nodes are critical factors in cancer staging.

# Distant Metastasis: In its advanced stages, CRC can spread to distant organs,
# such as the liver, lungs, stAnndata, or bones, through the bloodstream or lymphatic system.

# The extent of the invasion and the spread of the cancer are key components of the TNM staging system (Tumor, Nodes, Metastasis),
# which is used to determine the stage of colorectal cancer.


## Match the pathology label
merge_rules <- list(
    "tumor&stroma" = c("tumor&stroma_IC med to high", "tumor&stroma_IC low", "tumor&stroma"),
    "stroma_desmoplastic" = c("stroma_desmoplastic_IC low", "stroma_desmoplastic_IC med to high"),
    "stroma_fibroblastic" = c("stroma_fibroblastic_IC high", "stroma_fibroblastic_IC low", "stroma_fibroblastic_IC med"),
    "muscularis" = c("muscularis_IC med to high", "IC aggregate_muscularis or stroma"),
    "epithelium" = c("epithelium&submucosa", "non neo epithelium"),
    "submucosa" = c("submucosa", "IC aggregate_submucosa"),
    "tumor" = c("tumor"),
    "exclude" = c("exclude")
    # You can add or adjust the categories as needed
)


Match_Pathology_label <- function(df, merge_rules) {
    # Ensure the column exists
    if (!"Pathologist.Annotation" %in% names(df)) {
        stop("Pathologist.Annotation column not found in the dataframe")
    }

    # Ensure merge_rules is a list and not empty
    if (!is.list(merge_rules) || length(merge_rules) == 0) {
        stop("merge_rules must be a non-empty list")
    }

    # Iterate over each merge rule and apply
    df$Pathologist.Annotation2 <- NA
    for (new_category in names(merge_rules)) {
        categories_to_merge <- merge_rules[[new_category]]
        df$Pathologist.Annotation2[df$Pathologist.Annotation %in% categories_to_merge] <- new_category
    }

    return(df)
}

## calculate fold-change and p-value
FCandPvalueCal <- function(mat, xCol, yCol) {
    groups <- names(table(mat[, yCol]))
    groups <- as.character(sort(as.numeric(groups), decreasing = F))
    if (length(groups) < 2) {
        return(0)
    }

    returnMat <- matrix(data = NA, nrow = length(xCol[1]:xCol[2]), ncol = 3)
    returnMat <- as.data.frame(returnMat)
    colnames(returnMat) <- c("Celltype", "Foldchange", "P.value")

    group1mat <- mat[which(mat[, yCol] == groups[1]), ]
    group2mat <- mat[which(mat[, yCol] == groups[2]), ]

    for (i in xCol[1]:xCol[2]) {
        typeTemp <- colnames(mat)[i]

        v1 <- group1mat[, i]
        v2 <- group2mat[, i]

        ## relaps versus no relaps
        foldchange <- mean(v2) / mean(v1)
        pvalue <- t.test(v2, v1)$p.value

        returnMat[i, ] <- c(typeTemp, foldchange, pvalue)
    }

    return(returnMat)
}

## Plot volcano plot
VolcanoPlot <- function(df, pthreshold = 0.05, fcthreshold = 1.4, filename = NULL) {
    df$Foldchange <- as.numeric(df$Foldchange)
    df$P.value <- as.numeric(df$P.value)

    df$change <- as.factor(ifelse(df$P.value <= pthreshold & abs(log2(df$Foldchange)) >= log2(fcthreshold),
        ifelse(log2(df$Foldchange) >= log2(fcthreshold), "Up-regulate", "Down-regulate"), "Non-significant"
    ))

    # 样本标签
    df$label <- ifelse(df[, 3] <= pthreshold & abs(log2(df$Foldchange)) >= log2(fcthreshold), as.character(df[, 1]), "")

    # 绘制火山图
    p.vol <- ggplot(
        data = df,
        aes(x = log2(Foldchange), y = -log10(P.value), colour = change, fill = change)
    ) +
        scale_color_manual(values = c("Down-regulate" = "blue", "Non-significant" = "grey", "Up-regulate" = "red")) +
        geom_point(alpha = 0.4, size = 3.5) +
        # 标签
        geom_text_repel(aes(x = log2(Foldchange), y = -log10(P.value), label = label),
            size = 3,
            box.padding = unit(0.6, "lines"), point.padding = unit(0.7, "lines"),
            segment.color = "black", show.legend = FALSE
        ) +
        # 辅助线
        geom_vline(xintercept = c(-(log2(fcthreshold)), (log2(fcthreshold))), lty = 4, col = "black", lwd = 0.8) +
        geom_hline(yintercept = -log10(pthreshold), lty = 4, col = "black", lwd = 0.8) +
        theme_bw() +
        labs(x = "log2(Fold Change)", y = "-log10(P value)", title = paste0("Volcano Plot of Different types abundance in Rank+ and Rank-")) +
        # 坐标轴标题、标签和图例相关设置
        theme(
            axis.text = element_text(size = 11), axis.title = element_text(size = 13), # 坐标轴标签和标题
            plot.title = element_text(hjust = 0.5, size = 15, face = "bold"), # 标题
            legend.text = element_text(size = 11), legend.title = element_text(size = 13), # 图例标签和标题
            plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
        ) # 图边距

    ggsave(p.vol, filename = filename)

    return(NULL)
}

## combine the celltypes group
combn_types_pair <- function(types_vec1, types_vec2) {
    num_pair <- length(types_vec1) * length(types_vec2)
    df <- as.data.frame(matrix(data = NA, ncol = 2, nrow = num_pair))
    idx <- 1

    for (type_a in types_vec1) {
        for (type_b in types_vec2) {
            df[idx, 1] <- type_a
            df[idx, 2] <- type_b
            idx <- idx + 1
        }
    }
    return(df)
}

## self-correlation
self_correlation <- function(df, celltypes_pair) {
    celltypes <- unique(c(celltypes_pair[, 1], celltypes_pair[, 2]))
    cor_df <- df[, match(c("Sample", "Rank_Label", celltypes), colnames(df))]

    samples <- names(table(cor_df$Sample))
    labels <- names(table(cor_df$Rank_Label))

    # result_df <- as.data.frame(matrix(data = NA,nrow = 0,ncol = 5))
    result_df <- as.data.frame(matrix(data = NA, nrow = 0, ncol = 4))
    for (i in 1:nrow(celltypes_pair)) {
        type1 <- celltypes_pair[i, 1]
        type2 <- celltypes_pair[i, 2]

        # result_df_ <- as.data.frame(matrix(data = NA,nrow=length(samples)*length(labels),ncol = 4))
        result_df_ <- as.data.frame(matrix(data = NA, nrow = length(samples) * length(labels), ncol = 3))

        j <- 1

        for (sample in samples) {
            for (label in labels) {
                tempdf <- cor_df[(cor_df$Sample == sample) & (cor_df$Rank_Label == label), ]
                if (dim(tempdf)[1] < 1) {
                    next
                } else {
                    ## correlation
                    #    cor_coef <- cor(tempdf[,match(type1,colnames(tempdf))],tempdf[,match(type2,colnames(tempdf))],method = "spearman")
                    #    cor_pvalue <- cor.test(tempdf[,match(type1,colnames(tempdf))],tempdf[,match(type2,colnames(tempdf))],method = "spearman")
                    #    result_df_[j,] <- c(sample,label,cor_coef,cor_pvalue$p.value)

                    ## co-occurrence
                    score <- mean(tempdf[, match(type1, colnames(tempdf))] * tempdf[, match(type2, colnames(tempdf))])
                    result_df_[j, ] <- c(sample, label, score)
                    j <- j + 1
                }
            }
        }
        result_df_$Pair <- paste0(type1, "_", type2)
        result_df <- rbind(result_df, result_df_)
    }

    result_df <- na.omit(result_df)
    # colnames(result_df) <- c("Sample","Rank_Label","Coef","P_value","Pair")
    colnames(result_df) <- c("Sample", "Rank_Label", "Cooc_Score", "Pair")

    return(result_df)
}
