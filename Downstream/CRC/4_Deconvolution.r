## Decovolution

library(Seurat)
library(CARD)

library(tidyr)
library(dplyr)
library(reshape2)

library(ggplot2)
library(ggsci)
library(ggpubr)
library(ggrepel)

FigurePath <- "/home/lenislin/Experiment/projectsResult/TiRank/bulk2st/CRC/figures/"

## Prepare scRNA-seq data as reference
if (F) {
    scDatarawPath <- "/mnt/raid5/ProjectData/TiRank/scRNAseq/CRC/"
    exp_ <- read.table(paste0(scDatarawPath, "GSE144735_processed_KUL3_CRC_10X_raw_UMI_count_matrix.txt"), sep = "\t", header = T, row.names = 1)
    anno_ <- read.table(paste0(scDatarawPath, "GSE144735_processed_KUL3_CRC_10X_annotation.txt"), sep = "\t", header = T, row.names = 1)

    table(anno_$Class)
    table(anno_$Cell_subtype)

    set.seed(619)
    sampleSize <- 5e3
    sampleidx <- sample(1:nrow(anno_), size = sampleSize)

    exp_subset <- exp_[, sampleidx]
    anno_subset <- anno_[sampleidx, ]

    saveRDS(exp_subset, paste0(FigurePath, "scRNA_CRC_exp.rds"))
    saveRDS(anno_subset, paste0(FigurePath, "scRNA_CRC_meta.rds"))
}

# Volcano plot function
volcano_plot_degs <- function(plotdf, category_col_name, logFC_threshold = 0.75, pval_threshold = 0.05) {
    # Ensure category_col_name exists in plotdf
    if (!category_col_name %in% colnames(plotdf)) {
        stop("Category column not found in the dataframe.")
    }

    # Extract the gene columns (all except the category column)
    gene_columns <- colnames(plotdf)[!colnames(plotdf) %in% category_col_name]

    # Split the dataframe based on the categories in the last column
    #   categories <- unique(plotdf[[category_col_name]])
    categories <- c("Rank-", "Rank+")
    if (length(categories) != 2) {
        stop("There should be exactly two categories to compare.")
    }

    # Split into two groups
    group1 <- plotdf[plotdf[[category_col_name]] == categories[1], gene_columns]
    group2 <- plotdf[plotdf[[category_col_name]] == categories[2], gene_columns]

    # Initialize a data frame to store differential expression results
    deg_results <- data.frame(
        gene = gene_columns,
        FC = numeric(length(gene_columns)),
        p_value = numeric(length(gene_columns))
    )

    # Perform differential expression analysis for each gene
    for (gene in gene_columns) {
        # Calculate the log fold change
        mean_group1 <- mean(group1[[gene]])
        mean_group2 <- mean(group2[[gene]])
        FC <- (mean_group2) / (mean_group1)

        # Perform a t-test (or Wilcoxon test) for significance
        test_result <- t.test(group1[[gene]], group2[[gene]])
        p_value <- test_result$p.value

        # Store the results
        deg_results[deg_results$gene == gene, "FC"] <- FC
        deg_results[deg_results$gene == gene, "p_value"] <- p_value
    }

    # Add -log10(p-value) for plotting
    deg_results$neg_log_pval <- -log10(deg_results$p_value)
    deg_results$log2FC <- log2(deg_results$FC)

    deg_results$log2FC <- ifelse(deg_results$log2FC > 3, 3, deg_results$log2FC)
    deg_results$log2FC <- ifelse(deg_results$log2FC < (-3), -3, deg_results$log2FC)
    deg_results$neg_log_pval <- ifelse(deg_results$neg_log_pval > 10, 10, deg_results$neg_log_pval)

    # Add significance and direction columns
    deg_results$significant <- with(deg_results, p_value <= pval_threshold & abs(log2FC) >= logFC_threshold)
    deg_results$regulation <- ifelse(deg_results$log2FC >= logFC_threshold & deg_results$p_value <= pval_threshold, "up",
        ifelse(deg_results$log2FC <= -logFC_threshold & deg_results$p_value <= pval_threshold, "down", "not_significant")
    )

    # Select the top 5 up- and down-regulated genes
    top_up_genes <- deg_results %>%
        dplyr::filter(regulation == "up") %>%
        dplyr::top_n(5, wt = log2FC)
    top_up_genes <- top_up_genes$gene

    top_down_genes <- deg_results %>%
        dplyr::filter(regulation == "down") %>%
        dplyr::top_n(-5, wt = log2FC)
    top_down_genes <- top_down_genes$gene

    # Plotting the volcano plot
    p <- ggplot(deg_results, aes(x = log2FC, y = neg_log_pval)) +
        geom_point(aes(color = regulation), size = 2) +
        scale_color_manual(values = c("up" = "#BC3C29FF", "down" = "#0072B5FF", "not_significant" = "grey")) + # Red for up, blue for down, grey for non-significant
        labs(
            title = paste0("Volcano Plot of ", categories[2], " versus ", categories[1]),
            x = "Log2 Fold Change",
            y = "-log10(p-value)",
            color = "Regulation"
        ) +
        theme_minimal() +
        geom_text_repel(
            data = subset(deg_results, gene %in% c(top_up_genes, top_down_genes)),
            aes(label = gene),
            size = 3
        )

    return(p)
}

# Visualize the spatial distribution of cell type proportion
CARD.visualize.prop <- function(proportion, spatial_location, ct.visualize = ct.visualize, colors = c("lightblue", "lightyellow", "red"), NumCols, pointSize = 3.0) {
    if (is.null(colors)) {
        colors <- c("lightblue", "lightyellow", "red")
    } else {
        colors <- colors
    }
    res_CARD <- as.data.frame(proportion)
    res_CARD <- res_CARD[, order(colnames(res_CARD))]
    location <- as.data.frame(spatial_location)
    if (sum(rownames(res_CARD) == rownames(location)) != nrow(res_CARD)) {
        stop("The rownames of proportion data does not match with the rownames of spatial location data")
    }
    ct.select <- ct.visualize
    res_CARD <- res_CARD[, ct.select]
    if (!is.null(ncol(res_CARD))) {
        res_CARD_scale <- as.data.frame(apply(res_CARD, 2, function(x) {
            (x - min(x)) / (max(x) - min(x))
        }))
    } else {
        res_CARD_scale <- as.data.frame((res_CARD - min(res_CARD)) / (max(res_CARD) - min(res_CARD)))
        colnames(res_CARD_scale) <- ct.visualize
    }
    res_CARD_scale$x <- as.numeric(location$x)
    res_CARD_scale$y <- as.numeric(location$y)
    mData <- melt(res_CARD_scale, id.vars = c("x", "y"))
    colnames(mData)[3] <- "Cell_Type"
    b <- c(0, 1)
    p <- suppressMessages(ggplot(mData, aes(x, y)) +
        geom_point(aes(colour = value), size = pointSize) +
        scale_color_gradientn(colours = colors) +
        # scale_color_viridis_c(option = 2)+
        scale_x_discrete(expand = c(0, 1)) +
        scale_y_discrete(expand = c(0, 1)) +
        facet_wrap(~Cell_Type, ncol = NumCols) +
        # coord_fixed() +
        theme(
            plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"),
            # legend.position=c(0.14,0.76),
            panel.background = element_blank(),
            plot.background = element_blank(),
            panel.border = element_rect(colour = "grey89", fill = NA, size = 0.5),
            axis.text = element_blank(),
            axis.ticks = element_blank(),
            axis.title = element_blank(),
            legend.title = element_text(size = 14, face = "bold"),
            legend.text = element_text(size = 11),
            strip.text = element_text(size = 12, face = "bold"),
            legend.key = element_rect(colour = "transparent", fill = "white"),
            legend.key.size = unit(0.45, "cm")
        ))
    return(p)
}

## Define the merge region
if (T) {
    # Define merging categories
    merged_categories <- list(
        "IC Aggregate" = c("IC aggregate_submucosa", "IC aggragate_stroma or muscularis", "IC aggregregate_submucosa", "IC aggregate submucosa", "IC aggregate_muscularis or stroma", "IC aggregate_stroma or muscularis", "IC aggregates_stroma or muscularis", "IC aggreates_stroma or muscularis"),
        "Stroma Fibroblastic(IC_Aggre)" = c("stroma_fibroblastic_IC med", "stroma_fibroblastic_IC_med", "stroma_fibroblastic_IC high", "stroma_fibroblastic_IC_high", "muscularis_IC med to high"),
        "Stroma Fibroblastic" = c("stroma_fibroblastic_IC low", "stroma_fibroblastic_IC_low"),
        "Stroma Desmoplastic(IC_Aggre)" = c("stroma_desmoplastic_IC med to high", "stroma desmoplastic_IC med to high"),
        "Stroma Desmoplastic" = c("stroma_desmoplastic_IC low", "stroma desmoplastic_IC low"),
        # "Tumor & Stroma(IC_Aggre)" = c("tumor&stroma_IC med to high", "tumor&stroma IC med to high"),
        # "Tumor & Stroma" = c("tumor&stroma_IC low", "tumor&stroma", "tumor"),
        "Tumor & Stroma" = c("tumor&stroma_IC low", "tumor&stroma", "tumor", "tumor&stroma_IC med to high", "tumor&stroma IC med to high"),
        "Epithelium & Submucosa" = c("epitehlium&submucosa", "submucosa", "non neo epithelium", "epithelium&submucosa"),
        "Exclude" = c("exclude", "")
    )

    merged_categories_df <- data.frame(
        "Original" = unlist(merged_categories),
        "Merge" = rep(names(merged_categories), times = sapply(merged_categories, length))
    )
}

## Load Spatial Data and deconvlution
sc_count <- readRDS(paste0(FigurePath, "scRNA_CRC_exp.rds"))
sc_count <- as.matrix(sc_count)
colnames(sc_count) <- gsub(colnames(sc_count), pattern = "\\.", replacement = "-")

sc_meta <- readRDS(paste0(FigurePath, "scRNA_CRC_meta.rds"))

st_list <- readRDS(paste0(FigurePath, "all_seurat_obj.rds"))
slicesName <- names(st_list)

for (slice_ in slicesName) {
    ## Load slice and create CARD object
    seurat_object <- st_list[[slice_]]

    st_count <- seurat_object@assays$SCT@counts
    st_clocation <- seurat_object@meta.data[, c("array_row", "array_col")]
    colnames(st_clocation) <- c("x", "y")

    CARD_obj <- createCARDObject(
        sc_count = sc_count,
        sc_meta = sc_meta,
        spatial_count = st_count,
        spatial_location = st_clocation,
        ct.varname = "Cell_subtype",
        ct.select = NULL,
        sample.varname = NULL,
        minCountGene = 100,
        minCountSpot = 5
    )

    ## Deconvolution using CARD
    CARD_obj <- CARD_deconvolution(CARD_object = CARD_obj)
    deconvdf <- CARD_obj@Proportion_CARD

    ## Save
    saveRDS(CARD_obj, paste0(FigurePath, slice_, "/CARD_deconv.rds"))
    write.csv(deconvdf, paste0(FigurePath, slice_, "/CARD_Deconv_result.csv"))

    # CARD_obj <- readRDS(paste0(FigurePath, slice_, "/CARD_deconv.rds"))

    ## Visualize Interesting Types
    ct.visualize <- c(
        "Myofibroblasts", "Stromal 1", "Stromal 2", "Stromal 3", "Pericytes",
        "CMS3", "CMS2", "SPP1+A", "SPP1+B", "Anti-inflammatory", "Pro-inflammatory",
        "CD8+ T cells", "CD19+CD20+ B", "CD4+ T cells"
    )

    p2 <- CARD.visualize.prop(
        proportion = CARD_obj@Proportion_CARD,
        spatial_location = CARD_obj@spatial_location,
        ct.visualize = ct.visualize, ### selected cell types to visualize
        colors = c("lightblue", "lightyellow", "red"), ### if not provide, we will use the default colors
        NumCols = 5, ### number of columns in the figure panel
        pointSize = 1.2 ### point size in ggplot2 scatterplot
    )
    pdf(paste0(FigurePath, slice_, "/interesting_type_fractrion.pdf"), height = 10, width = 16)
    print(p2)
    dev.off()

    ## Visualize the cell type proportion correlation
    p4 <- CARD.visualize.Cor(CARD_obj@Proportion_CARD, colors = NULL) # if not provide, we will use the default colors
    pdf(paste0(FigurePath, slice_, "/CARD_celltype_correlation_allregion.pdf"), height = 15, width = 20)
    print(p4)
    dev.off()
}

## Compare the celltype abundance
if (T) {
    celltypes <- unique(sc_meta$Cell_subtype)

    allAbundance <- as.data.frame(matrix(data = NA, nrow = 0, ncol = length(celltypes)))
    colnames(allAbundance) <- celltypes

    allRankLabel <- c()
    allRankScore <- c()

    ## Combine Gene Expression
    for (slice_ in slicesName) {
        ## Load data
        abundanceDF <- read.csv(paste0(FigurePath, slice_, "/CARD_Deconv_result.csv"), header = T, row.name = 1)

        seurat_object <- st_list[[slice_]]
        seurat_object$Patho_anno_merge <- merged_categories_df[match(seurat_object$Patho_anno, merged_categories_df$Original), "Merge"]

        ## Subset spot
        idx <- seurat_object$Patho_anno_merge %in% c("Stroma Fibroblastic(IC_Aggre)", "Stroma Fibroblastic")

        abundanceDF_subset <- abundanceDF[idx, ]

        allRankLabel <- c(allRankLabel, as.character(seurat_object$Rank_Label)[idx])
        allRankScore <- c(allRankScore, as.numeric(seurat_object$Rank_Score)[idx])

        rownames(abundanceDF_subset) <- paste0(slice_, "_", rownames(abundanceDF_subset))
        allAbundance <- rbind(allAbundance, abundanceDF_subset)
    }

    colnames(allAbundance) <- celltypes

    ## Volcano plot for Diff cell types
    plotdf <- allAbundance
    plotdf$RankLabel <- as.factor(allRankLabel)

    p <- volcano_plot_degs(plotdf = plotdf, category_col_name = "RankLabel", logFC_threshold = 0.25, pval_threshold = 0.05)
    pdf(paste0(FigurePath, "Differential Celltypes of TiRank Label within Fibro_Stromal region.pdf"), height = 6, width = 8)
    print(p)
    dev.off()

    ## Boxplot for Key subpopulations
    if (T) {
        selectTypes <- c(
            "SPP1+B", "Myofibroblasts", "CMS2",
            "CD8+ T cells", "NK cells", "Pro-inflammatory"
        )
        plotdf2 <- plotdf[, match(c(selectTypes, "RankLabel"), colnames(plotdf))]
        plotdf2 <- subset(plotdf2, RankLabel %in% c("Rank-", "Rank+"))
        # plotdf2$Patient <- sapply(rownames(plotdf2),function(x){
        #     ids <- strsplit(x,split = "_")[[1]]
        #     return(paste0(ids[1:(length(ids)-1)],collapse = "_"))
        # })

        # plotdf3 <- plotdf2 %>%
        # group_by(Patient,RankLabel) %>%
        # summarise(across(c(1:(ncol(plotdf2) - 2)), mean, na.rm = TRUE))
        # plotdf3 <- as.data.frame(plotdf3)

        # plotdf3 <- pivot_longer(plotdf3,cols = 3:8,values_to = "Abundance",names_to = "Celltype")
        plotdf3 <- pivot_longer(plotdf2, cols = 1:length(selectTypes), values_to = "Abundance", names_to = "Celltype")
        plotdf3 <- as.data.frame(plotdf3)
        plotdf3$Celltype <- factor(plotdf3$Celltype, levels = selectTypes)
        plotdf3$RankLabel <- as.factor(plotdf3$RankLabel)


        p <- ggplot(plotdf3, aes(x = RankLabel, y = Abundance, fill = RankLabel)) +
            geom_boxplot(alpha = 0.7, color = "black", outlier.shape = NA) +
            geom_jitter(color = "darkgrey", position = position_jitter(width = 0.2), size = 0.1, alpha = 0.3) +
            scale_fill_manual(values = ggsci::pal_jco("default")(2)) +
            theme_minimal() +
            theme(
                plot.title = element_text(size = 16, face = "bold"),
                text = element_text(size = 12),
                axis.title = element_text(face = "bold", size = 14),
                axis.text.x = element_text(size = 11, angle = 90, hjust = 1, vjust = 0.5),
                strip.background = element_blank()
            ) +
            stat_summary(
                fun = mean,
                geom = "text",
                aes(label = round(..y.., 5)),
                position = position_nudge(y = 0.05),
                size = 3,
                color = "black"
            ) +
            stat_compare_means(aes(group = RankLabel),
                method = "wilcox",
                hide.ns = FALSE,
                label.y.npc = "middle"
            ) +
            facet_wrap(~Celltype, scales = "free_y")

        pdf(paste0(FigurePath, "Boxplot of Differential Celltypes of TiRank Label within Fibro_Stromal region.pdf"), height = 6, width = 8)
        print(p)
        dev.off()
    }

    ## Visualize the cell type proportion correlation
    ct.visualize <- c("Myofibroblasts", "Pericytes", "SPP1+A", "SPP1+B", "Anti-inflammatory", "Pro-inflammatory")

    plotdf <- allAbundance[, match(ct.visualize, colnames(allAbundance))]
    p4 <- CARD.visualize.Cor(plotdf, colors = NULL) # if not provide, we will use the default colors
    pdf(paste0(FigurePath, "CARD_celltype_correlation_within Fibro_Stromal region.pdf"), height = 6, width = 8)
    print(p4)
    dev.off()
}
