library(Seurat)
library(Matrix)
library(dplyr)
library(tidyr)
library(jsonlite)
library(stringr)

library(ggplot2)
library(ggsci)
library(ggpubr)
library(ggrepel)

## Set path
ResultPath <- "/home/lenislin/Experiment/projectsResult/TiRank/bulk2st/CRC/results/"
FigurePath <- "/home/lenislin/Experiment/projectsResult/TiRank/bulk2st/CRC/figures/"
AnnofilePath <- "/mnt/raid5/ProjectData/TiRank/ST/CRC/slices/"
slicesName <- list.files(ResultPath)

## Self defined functions
load_create_seurat <- function(rankdataPath, rawdataPath) {
    # sparse matrix
    # expression_profile <- Matrix::readMM(paste0(dataPath, "4_Convert2R/", "expression_profile.mtx"))
    # expression_profile <- t(expression_profile)

    # row_names <- read.csv(paste0(dataPath, "4_Convert2R/", "row_names.csv"), header = FALSE, stringsAsFactors = FALSE)$V1
    # col_names <- read.csv(paste0(dataPath, "4_Convert2R/", "column_names.csv"), header = FALSE, stringsAsFactors = FALSE)$V1

    # rownames(expression_profile) <- row_names
    # colnames(expression_profile) <- col_names

    # metadata
    metadata <- read.csv(paste0(rankdataPath, "4_Convert2R/", "metadata.csv"), row.names = 1)

    # Create a Seurat object
    seurat_object <- Load10X_Spatial(paste0(rawdataPath))
    seurat_object <- seurat_object[, match(rownames(metadata), colnames(seurat_object))]
    seurat_object@meta.data <- metadata

    # Preprocessing
    seurat_object <- SCTransform(seurat_object, assay = "Spatial", verbose = FALSE)

    seurat_object <- RunPCA(seurat_object, assay = "SCT", verbose = FALSE)
    seurat_object <- FindNeighbors(seurat_object, reduction = "pca", dims = 1:15)
    seurat_object <- FindClusters(seurat_object, verbose = FALSE)
    seurat_object <- RunUMAP(seurat_object, reduction = "pca", dims = 1:15)

    return(seurat_object)
}

# Function to select highly variable genes
select_highly_variable_genes <- function(expression_matrix, num_genes = 2000) {
    # Calculate mean and variance for each gene (row)
    gene_means <- rowMeans(expression_matrix)
    gene_vars <- apply(expression_matrix, 1, var)

    # Compute coefficient of variation (CV) as a measure of variability
    gene_cv <- gene_vars / gene_means

    # Log-transform the mean and CV to stabilize variance for further selection
    log_gene_means <- log1p(gene_means)
    log_gene_cv <- log1p(gene_cv)

    # Fit a loess regression model to capture the relationship between mean and CV
    loess_fit <- loess(log_gene_cv ~ log_gene_means)

    # Compute residuals from the loess fit to determine gene variability
    residuals <- log_gene_cv - predict(loess_fit)

    # Rank the genes by their residuals (highest residuals represent highly variable genes)
    ranked_genes <- order(residuals, decreasing = TRUE)

    # Select the top 'num_genes' genes
    top_genes <- rownames(expression_matrix)[ranked_genes][1:num_genes]

    return(top_genes)
}

# Function to compute DEGs and plot a volcano plot
volcano_plot_degs <- function(degs, logFC_threshold = 0.75, pval_threshold = 0.05) {
    # Check if the required columns exist
    if (!all(c("p_val", "avg_log2FC", "p_val_adj") %in% colnames(degs))) {
        stop("The dataframe must contain 'p_val', 'avg_log2FC', and 'p_val_adj' columns.")
    }

    # Add gene names as a column
    degs$gene <- rownames(degs)

    # Add -log10(p-value) for plotting
    degs$neg_log_pval <- -log10(degs$p_val_adj)

    # Clip the log fold changes and negative log p-values for display
    degs$avg_log2FC <- ifelse(degs$avg_log2FC > 3, 3, degs$avg_log2FC)
    degs$avg_log2FC <- ifelse(degs$avg_log2FC < -3, -3, degs$avg_log2FC)
    degs$neg_log_pval <- ifelse(degs$neg_log_pval > 10, 10, degs$neg_log_pval)

    # Add significance and regulation columns
    degs$significant <- with(degs, p_val_adj <= pval_threshold & abs(avg_log2FC) >= logFC_threshold)
    degs$regulation <- ifelse(degs$avg_log2FC >= logFC_threshold & degs$p_val_adj <= pval_threshold, "up",
        ifelse(degs$avg_log2FC <= -logFC_threshold & degs$p_val_adj <= pval_threshold, "down", "not_significant")
    )

    # Remove mitochondrial genes if needed
    degs <- degs[!startsWith(degs$gene, "MT."), ]
    degs <- degs[degs$neg_log_pval > 1e-2, ]

    # Select the top 5 up- and down-regulated genes
    top_up_genes <- degs %>%
        filter(regulation == "up") %>%
        top_n(5, wt = avg_log2FC) %>%
        pull(gene)

    top_down_genes <- degs %>%
        filter(regulation == "down") %>%
        top_n(-5, wt = avg_log2FC) %>%
        pull(gene)

    # Plotting the volcano plot
    p <- ggplot(degs, aes(x = avg_log2FC, y = neg_log_pval)) +
        geom_point(aes(color = regulation), size = 1.5) +
        scale_color_manual(values = c("up" = "#BC3C29FF", "down" = "#0072B5FF", "not_significant" = "grey")) + # Red for up, blue for down, grey for non-significant
        labs(
            title = "Volcano Plot",
            x = "Log2 Fold Change",
            y = "-log10(Adjusted p-value)",
            color = "Regulation"
        ) +
        theme_minimal() +
        geom_label_repel(
            data = subset(degs, gene %in% c(top_up_genes, top_down_genes)),
            aes(label = gene),
            size = 3
        )

    return(p)
}

# Function to load pathway information
load_pathway_from_kegg <- function(prefix_) {
    ## Prepare list of gene set
    # Retrieve KEGG pathway links for Homo sapiens (hsa)
    # hsa_path <- keggLink("pathway", "hsa")
    # meta <- unique(hsa_path)[grepl("hsa00", unique(hsa_path))]

    all_pathways <- keggList("pathway", "hsa") # "hsa" for Homo sapiens
    # meta_1 <- names(all_pathways[startsWith(names(all_pathways) ,prefix = "hsa01")])
    meta <- names(all_pathways)[sapply(names(all_pathways), function(x) {
        any(startsWith(x, prefix_))
    })]
    # Filter unique pathways that contain 'hsa00'
    hsa_info <- lapply(meta, keggGet) # Get detailed information for each filtered pathway
    nm <- unlist(lapply(hsa_info, function(x) x[[1]]$NAME)) # Extract pathway names

    # Extract genes associated with each pathway
    genes <- unlist(lapply(hsa_info, function(x) {
        g <- x[[1]]$GENE
        # Split gene information and collapse into a single string
        paste(str_split(g[seq(2, length(g), by = 2)], ";", simplify = TRUE)[, 1], collapse = ";")
    }))

    # Create a data frame with pathway IDs, names, and associated genes
    geneset <- data.frame(hsa = meta, nm = nm, genes = genes)
    geneset$hsa <- sapply(geneset$hsa, function(x) {
        return(strsplit(x, ":")[[1]][2])
    })
    geneset$nm <- sapply(geneset$nm, function(x) {
        return(strsplit(x, " - ")[[1]][1])
    })

    geneset_list <- list()
    for (i in 1:nrow(geneset)) {
        geneset_list[[geneset[i, 2]]] <- strsplit(geneset[i, 3], ";")[[1]]
    }

    return(geneset_list)
}

## Load all ST data
if (F) {
    seurat_object_list <- list()
    regionNames <- c()
    for (slice_ in slicesName) {
        sliceResultPath <- paste0(ResultPath, slice_, "/")
        rawDataPath <- paste0(AnnofilePath, slice_, "/")

        ## Load data
        seurat_object <- load_create_seurat(sliceResultPath, rawDataPath)
        regionNames <- c(regionNames, names(table(seurat_object$Patho_anno)))

        ## Merge
        seurat_object_list[[slice_]] <- seurat_object
    }
    saveRDS(seurat_object_list, paste0(FigurePath, "all_seurat_obj.rds"))

    regionNames <- unique(regionNames)
}

seurat_object_list <- readRDS(paste0(FigurePath, "all_seurat_obj.rds"))
slicesName <- names(seurat_object_list)

# Define merging categories
if (T) {
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

## Statistical the fraction of region
if (T) {
    results <- as.data.frame(matrix(data = 0, nrow = length(regionNames), ncol = 3))
    colnames(results) <- c("Rank+", "Rank-", "Background")
    rownames(results) <- regionNames

    for (slice_ in slicesName) {
        PredictResultPath <- paste0(ResultPath, slice_, "/", "3_Analysis/Patho_anno_category_dict.json")

        ## Load Pcluster result
        data <- fromJSON(PredictResultPath)

        ## Merge
        for (name_ in names(data)) {
            regions <- data[[name_]]
            results[match(regions, rownames(results)), match(name_, colnames(results))] <- results[region, name_] + 1
        }
    }

    # Initialize a data frame to store mean values
    mean_values <- as.data.frame(matrix(data = 0, nrow = length(merged_categories), ncol = 3))
    rownames(mean_values) <- names(merged_categories)
    colnames(mean_values) <- c("Rank+", "Rank-", "Background")

    # Calculate the mean for each merged category
    for (category in names(merged_categories)) {
        regions <- merged_categories[[category]]
        # Get the values for the corresponding regions
        region_values <- results[rownames(results) %in% regions, ] / rowSums(results[rownames(results) %in% regions, ]) # Divide by 8
        # Calculate the mean of each value for the merged regions
        mean_value <- colMeans(region_values, na.rm = TRUE)

        # Add to mean_values dataframe
        mean_values[category, ] <- mean_value
    }

    # Output the mean values table
    print(mean_values)
    mean_values$Regions <- rownames(mean_values)
    mean_values <- mean_values[order(-mean_values[, "Rank+"], mean_values[, "Rank-"]), ]
    mean_values$Regions <- factor(mean_values$Regions, levels = mean_values$Regions)

    # Visualization the fraction of rank label within each region
    cate_colors <- c("Rank+" = "#E64B35FF", "Rank-" = "#4DBBD5FF", "Background" = "lightgrey")

    plotdf <- pivot_longer(mean_values, cols = 1:3, values_to = "Fraction", names_to = "Rank_Label")
    plotdf <- as.data.frame(plotdf)

    p <- ggplot(plotdf, aes(x = Regions, y = Fraction, fill = Rank_Label)) +
        geom_bar(stat = "identity", position = "fill") + # Use position = "fill" for fractions
        scale_fill_manual(values = cate_colors) +
        labs(
            title = "Contribution of Rank Labels within Different Regions",
            x = "Regions",
            y = "Fraction",
            fill = "Rank Label"
        ) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels for better visibility

    pdf(paste0(FigurePath, "Pathological region rank label distribution.pdf"), height = 6, width = 8)
    print(p)
    dev.off()
}


## Plot Label on each slice
if (T) {
    seurat_object_list <- readRDS(paste0(FigurePath, "all_seurat_obj.rds"))
    slicesName <- names(seurat_object_list)

    ## Set color
    cate_colors <- c("Rank+" = "#E64B35FF", "Rank-" = "#4DBBD5FF", "Background" = "darkgrey")
    region_colors <- setNames(pal_ucscgb("default")(8), names(merged_categories))
    values_colors <- colorRampPalette(colors = c("white", "red"))(100)

    for (slice_ in slicesName) {
        ## Load data
        seurat_object <- seurat_object_list[[slice_]]
        sliceResultSavePath <- paste0(FigurePath, slice_, "/")
        if (!dir.exists(sliceResultSavePath)) {
            dir.create(sliceResultSavePath, recursive = T)
        }

        ## Reassign Region category
        seurat_object$Patho_anno_merge <- merged_categories_df[match(seurat_object$Patho_anno, merged_categories_df$Original), "Merge"]

        ## Plot HE Image with Annotation
        p1 <- SpatialDimPlot(seurat_object, group.by = "Patho_anno_merge", cols = region_colors, image.alpha = 0.5, label = FALSE, label.size = 15, pt.size.factor = 2)
        pdf(paste0(sliceResultSavePath, "Patho_anno_region.pdf"), height = 6, width = 8)
        print(p1)
        dev.off()

        ## Plot HE Image
        p1 <- SpatialDimPlot(seurat_object, group.by = "Patho_anno_merge", label = FALSE, alpha = 0.0, label.size = 12)
        pdf(paste0(sliceResultSavePath, "HE_original.pdf"), height = 6, width = 8)
        print(p1)
        dev.off()

        ## Plot the TiRank Label and value
        p1 <- SpatialDimPlot(seurat_object, group.by = "Rank_Label", cols = cate_colors, image.alpha = 0.5, label = FALSE, label.size = 12, pt.size.factor = 2)
        pdf(paste0(sliceResultSavePath, "TiRank_Label.pdf"), height = 6, width = 8)
        print(p1)
        dev.off()

        p1 <- SpatialFeaturePlot(seurat_object, features = "Rank_Score", image.alpha = 0.5, pt.size.factor = 2) &
            scale_fill_gradientn(colours = values_colors)

        pdf(paste0(sliceResultSavePath, "TiRank_Score.pdf"), height = 6, width = 8)
        print(p1)
        dev.off()

        ## Plot TiRank label with P cluster
        JSONPath <- paste0(ResultPath, slice_, "/", "3_Analysis/Patho_anno_category_dict.json")
        data <- fromJSON(JSONPath)

        seurat_object@meta.data$Rank_Label_Pcluster <- NA

        ## Merge
        for (name_ in names(data)) {
            regions <- data[[name_]]
            for (region in regions) {
                seurat_object@meta.data$Rank_Label_Pcluster[seurat_object@meta.data$Patho_anno %in% region] <- name_
            }
        }

        ## Plot the TiRank Label and value
        p1 <- SpatialDimPlot(seurat_object, group.by = "Rank_Label_Pcluster", cols = cate_colors, image.alpha = 0.5, label = FALSE, label.size = 12, pt.size.factor = 2)
        pdf(paste0(sliceResultSavePath, "TiRank_Pcluster_Label.pdf"), height = 6, width = 8)
        print(p1)
        dev.off()
    }
}

## Collect the Spots from Fibro_stromal region
if (T) {
    ## Intersect Genes
    genenames <- rownames(seurat_object_list[[1]]@assays$SCT@counts)
    for (i in 2:length(seurat_object_list)) {
        genenames <- intersect(genenames, rownames(seurat_object_list[[i]]@assays$SCT@counts))
    }

    allExp <- as.data.frame(matrix(data = NA, nrow = length(genenames), ncol = 0))
    rownames(allExp) <- genenames

    allRankLabel <- c()
    allRankScore <- c()

    ## Combine Gene Expression
    for (slice_ in slicesName) {
        ## Load data
        seurat_object <- seurat_object_list[[slice_]]
        seurat_object$Patho_anno_merge <- merged_categories_df[match(seurat_object$Patho_anno, merged_categories_df$Original), "Merge"]

        ## Subset spot
        seurat_object_FS <- subset(seurat_object, Patho_anno_merge %in% c("Stroma Fibroblastic(IC_Aggre)", "Stroma Fibroblastic"))
        if (ncol(seurat_object) == 0) {
            next
        }
        Exp_ <- seurat_object_FS@assays$SCT@counts

        allRankLabel <- c(allRankLabel, as.character(seurat_object_FS$Rank_Label))
        allRankScore <- c(allRankScore, as.numeric(seurat_object_FS$Rank_Score))

        colnames(Exp_) <- paste0(slice_, "_", colnames(Exp_))
        Exp_ <- Exp_[match(genenames, rownames(Exp_)), ]
        allExp <- cbind(allExp, Exp_)
    }

    ## Volcano plot for DEGs
    seu_deg <- CreateSeuratObject(allExp)
    seu_deg$RankLabel <- as.character(allRankLabel)

    seu_deg <- NormalizeData(seu_deg)
    seu_deg <- FindVariableFeatures(seu_deg, selection.method = "vst", nfeatures = 2000)
    seu_deg <- ScaleData(seu_deg)

    Idents(seu_deg) <- as.factor(allRankLabel)
    degs <- FindMarkers(seu_deg, ident.1 = "Rank+", ident.2 = "Rank-")
    plotdf <- degs

    write.csv(degs, paste0(FigurePath, "DEG of Rank+ versus Rank- within Fibro_Stromal.csv"))

    p <- volcano_plot_degs(degs = plotdf, logFC_threshold = 0.5, pval_threshold = 0.05)
    pdf(paste0(FigurePath, "DEGs of TiRank Label within Fibro_Stromal region.pdf"), height = 6, width = 8)
    print(p)
    dev.off()

    ## Boxplot for Gene expression
}

## Enrichment analysis
library(msigdbr)
library(enrichplot)
library(fgsea)
library(clusterProfiler)
library(GSVA)
library(KEGGREST)

if (T) {
    ## Load DEGs from Rank+ versus Rank-
    degs <- read.csv(paste0(FigurePath, "DEG of Rank+ versus Rank- within Fibro_Stromal.csv"), header = T, row.name = 1)
    # degs_df <- degs[order(degs$avg_log2FC, decreasing = T), ]

    degs_sig <- subset(degs, p_val_adj <= 0.05)
    # degs_df <- degs_sig[order(degs_sig$avg_log2FC, decreasing = T), ]
    degs_sig_up <- subset(degs_sig, avg_log2FC > 0.25)
    degs_sig_down <- subset(degs_sig, avg_log2FC < (-0.25))
    degs_df <- rbind(degs_sig_up, degs_sig_down)
    degs_df <- degs_df[order(degs_df$avg_log2FC, decreasing = T), ]


    ## Load pathway
    geneset_list <- load_pathway_from_kegg(prefix_ = c("hsa00", "hsa043", "hsa046", "hsa052"))

    ## Enrichment
    degs_vec <- setNames(degs_df$avg_log2FC, rownames(degs_df))

    ## fGSEA
    Agsea_res <- fgsea(
        pathways = geneset_list,
        stats = degs_vec,
        minSize = 5,
        maxSize = 500,
        nperm = 1000
    )

    # plot
    selectPathway <- c(
        "Glycerophospholipid metabolism",
        "Wnt signaling pathway",
        "VEGF signaling pathway",
        "TNF signaling pathway",
        "IL-17 signaling pathway"
    )
    p <- plotGseaTable(geneset_list[selectPathway], degs_vec, Agsea_res, gseaParam = 0.5)
    pdf(paste0(FigurePath, "Enrichment of DEGs from TiRank Label within Fibro_Stromal region.pdf"), height = 6, width = 8)
    print(p)
    dev.off()

    for (pathway_ in selectPathway) {
        p <- plotEnrichment(geneset_list[[pathway_]], degs_vec, ticksSize = 0.5) +
            labs(title = pathway_) +
            theme(panel.border = element_rect(fill = NA, color = "black", size = 1, linetype = "solid"))

        pdf(paste0(FigurePath, pathway_, " Enrichment of DEGs from TiRank Label within Fibro_Stromal region.pdf"), height = 3, width = 4)
        print(p)
        dev.off()
    }
}
