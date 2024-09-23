## Sptial Analysis through MISTy

library(Seurat)
library(mistyR)
library(decoupleR)
library(OmnipathR)
library(distances)

library(tidyr)
library(dplyr)
library(tibble)

library(ggplot2)
library(ggrepel)

library(reticulate)
library(janitor)

## Adjust future.global.maxSize?
options(future.globals.maxSize = 1073741824) ## 1024 MB * 1024 ^ 2

FigurePath <- "/home/lenislin/Experiment/projectsResult/TiRank/bulk2st/CRC/figures/"

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

## Load Spatial Data
st_list <- readRDS(paste0(FigurePath, "all_seurat_obj.rds"))
slicesName <- names(st_list)

sc_meta <- readRDS(paste0(FigurePath, "scRNA_CRC_meta.rds"))
celltypes <- unique(sc_meta$Cell_subtype)

## Interatively on each slide
for (slice_ in slicesName) {
    ## Load slice
    seurat_object <- st_list[[slice_]]
    FigurePath_ <- paste0(FigurePath, slice_, "/")

    expression_raw <- as.matrix(GetAssayData(seurat_object, layer = "counts", assay = "SCT"))
    geometry <- GetTissueCoordinates(seurat_object, scale = NULL)
    geometry <- geometry[, c(1:2)]

    # Only take genes that  expressed in at least 5% of the spots
    expression <- expression_raw[rownames(expression_raw[(rowSums(expression_raw > 0) / ncol(expression_raw)) >= 0.05, ]), ]
    # Highly variable genes
    hvg <- FindVariableFeatures(expression, selection.method = "vst", nfeatures = 1000) %>%
        filter(variable == TRUE)

    hvg_expr <- expression[rownames(hvg), ]

    ## Extract cell-type composition
    composition <- read.csv(paste0(FigurePath_, "CARD_Deconv_result.csv"), header = T, row.name = 1)
    # colnames(composition) <- celltypes
    composition <- as_tibble(composition)

    ## Celltypes abundance
    if (T) {
        # Calculating the radius
        geom_dist <- as.matrix(distances(geometry))
        dist_nn <- apply(geom_dist, 1, function(x) (sort(x)[2]))
        paraview_radius <- ceiling(mean(dist_nn + sd(dist_nn)))

        # Create views
        heart_views <- create_initial_view(composition) %>%
            add_paraview(geometry, l = paraview_radius, family = "gaussian")

        # Run misty and collect results
        run_misty(heart_views, paste0(FigurePath_, "result/structural"))

        misty_results <- collect_results(paste0(FigurePath_, "result/structural"))

        pdf(paste0(FigurePath_, "Celltypes individual contributions of the views.pdf"), height = 6, width = 8)
        misty_results %>% plot_interaction_heatmap(view = "intra", clean = TRUE)
        dev.off()
    }

    ## Pathway activities on cell-type composition
    if (T) {
        # Obtain genesets
        model <- get_progeny(organism = "human", top = 500)
        # Use multivariate linear model to estimate activity
        est_path_act <- run_mlm(expression, model, .mor = NULL)

        # Add the result to the Seurat Object
        seurat_object[["progeny"]] <- NULL

        # Put estimated pathway activities object into the correct format
        est_path_act_wide <- est_path_act %>%
            pivot_wider(id_cols = condition, names_from = source, values_from = score) %>%
            column_to_rownames("condition")

        # Clean names
        colnames(est_path_act_wide) <- est_path_act_wide %>%
            clean_names(parsing_option = 0) %>%
            colnames(.)

        # Add to Seurat
        seurat_object[["progeny"]] <- CreateAssayObject(counts = t(est_path_act_wide))

        ## MISTy Views
        # Clean names
        colnames(composition) <- composition %>%
            clean_names(parsing_option = 0) %>%
            colnames(.)

        # create intra from cell-type composition
        comp_views <- create_initial_view(composition)

        # juxta & para from pathway activity
        path_act_views <- create_initial_view(est_path_act_wide) %>%
            add_juxtaview(geometry, neighbor.thr = 130) %>%
            add_paraview(geometry, l = 200, family = "gaussian")

        # Combine views
        com_path_act_views <- comp_views %>%
            add_views(create_view("juxtaview.path.130", path_act_views[["juxtaview.130"]]$data, "juxta.path.130")) %>%
            add_views(create_view("paraview.path.200", path_act_views[["paraview.200"]]$data, "para.path.200"))

        run_misty(com_path_act_views, paste0(FigurePath_, "result/comp_path_act"))
        misty_results_com_path_act <- collect_results(paste0(FigurePath_, "result/comp_path_act/"))

        # Plot
        pdf(paste0(FigurePath_, "Pathway individual contributions of the views.pdf"), height = 6, width = 8)
        misty_results_com_path_act %>%
            plot_view_contributions()
        dev.off()

        # What are the specific relations that can explain the cell-type composition?
        pdf(paste0(FigurePath_, "juxta.path.130 to cell types.pdf"), height = 12, width = 8)
        misty_results_com_path_act %>%
            plot_interaction_heatmap("juxta.path.130", clean = TRUE)
        dev.off()
    }

    ## Ligand-Receptor
    if (T) {
        # Ligand Receptor Resource
        omni_resource <- read.csv("omni_resource.csv", row.names = 1) %>%
            filter(resource == "consensus")

        # Get highly variable ligands
        ligands <- omni_resource %>%
            pull(source_genesymbol) %>%
            unique()
        hvg_lig <- hvg_expr[rownames(hvg_expr) %in% ligands, ]

        # Get highly variable receptors
        receptors <- omni_resource %>%
            pull(target_genesymbol) %>%
            unique()
        hvg_recep <- hvg_expr[rownames(hvg_expr) %in% receptors, ]

        # Clean names
        rownames(hvg_lig) <- hvg_lig %>%
            clean_names(parsing_option = 0) %>%
            rownames(.)

        rownames(hvg_recep) <- hvg_recep %>%
            clean_names(parsing_option = 0) %>%
            rownames(.)

        # Misty Views
        # Create views and combine them
        receptor_view <- create_initial_view(as.data.frame(t(hvg_recep)))

        ligand_view <- create_initial_view(as.data.frame(t(hvg_lig))) %>%
            add_paraview(geometry, l = 200, family = "gaussian")

        lig_recep_view <- receptor_view %>% add_views(create_view("paraview.ligand.200", ligand_view[["paraview.200"]]$data, "para.lig.200"))

        run_misty(lig_recep_view, paste0(FigurePath_, "result/lig_recep"), bypass.intra = TRUE)
        misty_results_lig_recep <- collect_results(paste0(FigurePath_, "result/lig_recep"))

        ## Downstream Analysis
        pdf(paste0(FigurePath_, "para.lig.200 ligand receptors.pdf"), height = 6, width = 12)
        plot_interaction_heatmap(misty_results_lig_recep, "para.lig.200", clean = TRUE, cutoff = 2, trim.measure = "gain.R2", trim = 10)
        dev.off()
    }
}

## Fibro Stromal Region
for (slice_ in slicesName) {
    ## Load slice
    seurat_object <- st_list[[slice_]]
    FigurePath_ <- paste0(FigurePath, slice_, "/")

    ## Subset spot
    seurat_object$Patho_anno_merge <- merged_categories_df[match(seurat_object$Patho_anno, merged_categories_df$Original), "Merge"]
    idx <- seurat_object$Patho_anno_merge %in% c("Stroma Fibroblastic(IC_Aggre)", "Stroma Fibroblastic")
    seurat_object <- seurat_object[, idx]

    expression_raw <- as.matrix(GetAssayData(seurat_object, layer = "counts", assay = "SCT"))
    geometry <- GetTissueCoordinates(seurat_object, scale = NULL)
    geometry <- geometry[, c(1:2)]

    # Only take genes that  expressed in at least 5% of the spots
    expression <- expression_raw[rownames(expression_raw[(rowSums(expression_raw > 0) / ncol(expression_raw)) >= 0.05, ]), ]
    # Highly variable genes
    hvg <- FindVariableFeatures(expression, selection.method = "vst", nfeatures = 1000) %>%
        filter(variable == TRUE)

    hvg_expr <- expression[rownames(hvg), ]

    ## Extract cell-type composition
    composition <- read.csv(paste0(FigurePath_, "CARD_Deconv_result.csv"), header = T, row.name = 1)
    # colnames(composition) <- celltypes
    composition <- composition[idx, ]
    composition <- as_tibble(composition)

    ## Celltypes abundance
    if (T) {
        # Calculating the radius
        geom_dist <- as.matrix(distances(geometry))
        dist_nn <- apply(geom_dist, 1, function(x) (sort(x)[2]))
        paraview_radius <- ceiling(mean(dist_nn + sd(dist_nn)))

        # Create views
        heart_views <- create_initial_view(composition) %>%
            add_paraview(geometry, l = paraview_radius, family = "gaussian")

        # Run misty and collect results
        run_misty(heart_views, paste0(FigurePath_, "result_FS/structural"))

        misty_results <- collect_results(paste0(FigurePath_, "result_FS/structural"))

        pdf(paste0(FigurePath_, "FS Celltypes individual contributions of the views.pdf"), height = 6, width = 8)
        misty_results %>% plot_interaction_heatmap(view = "intra", clean = TRUE)
        dev.off()
    }

    ## Pathway activities on cell-type composition
    # Obtain genesets
    model <- get_progeny(organism = "human", top = 500)
    # Use multivariate linear model to estimate activity
    est_path_act <- run_mlm(expression, model, .mor = NULL)

    # Add the result to the Seurat Object
    seurat_object[["progeny"]] <- NULL

    # Put estimated pathway activities object into the correct format
    est_path_act_wide <- est_path_act %>%
        pivot_wider(id_cols = condition, names_from = source, values_from = score) %>%
        column_to_rownames("condition")

    # Clean names
    colnames(est_path_act_wide) <- est_path_act_wide %>%
        clean_names(parsing_option = 0) %>%
        colnames(.)

    # Add to Seurat
    seurat_object[["progeny"]] <- CreateAssayObject(counts = t(est_path_act_wide))

    ## MISTy Views
    # Clean names
    colnames(composition) <- composition %>%
        clean_names(parsing_option = 0) %>%
        colnames(.)

    # create intra from cell-type composition
    comp_views <- create_initial_view(composition)

    # juxta & para from pathway activity
    path_act_views <- create_initial_view(est_path_act_wide) %>%
        add_juxtaview(geometry, neighbor.thr = 130) %>%
        add_paraview(geometry, l = 200, family = "gaussian")

    # Combine views
    com_path_act_views <- comp_views %>%
        add_views(create_view("juxtaview.path.130", path_act_views[["juxtaview.130"]]$data, "juxta.path.130")) %>%
        add_views(create_view("paraview.path.200", path_act_views[["paraview.200"]]$data, "para.path.200"))

    run_misty(com_path_act_views, paste0(FigurePath_, "result_FS/comp_path_act"))
    misty_results_com_path_act <- collect_results(paste0(FigurePath_, "result_FS/comp_path_act/"))

    # What are the specific relations that can explain the cell-type composition?
    pdf(paste0(FigurePath_, "FS juxta.path.130 to cell types.pdf"), height = 12, width = 8)
    misty_results_com_path_act %>%
        plot_interaction_heatmap("juxta.path.130", clean = TRUE)
    dev.off()

    ## Ligand-Receptor
    # Ligand Receptor Resource
    omni_resource <- read.csv("omni_resource.csv", row.names = 1) %>%
        filter(resource == "consensus")

    # Get highly variable ligands
    ligands <- omni_resource %>%
        pull(source_genesymbol) %>%
        unique()
    hvg_lig <- hvg_expr[rownames(hvg_expr) %in% ligands, ]

    # Get highly variable receptors
    receptors <- omni_resource %>%
        pull(target_genesymbol) %>%
        unique()
    hvg_recep <- hvg_expr[rownames(hvg_expr) %in% receptors, ]

    # Clean names
    rownames(hvg_lig) <- hvg_lig %>%
        clean_names(parsing_option = 0) %>%
        rownames(.)

    rownames(hvg_recep) <- hvg_recep %>%
        clean_names(parsing_option = 0) %>%
        rownames(.)

    # Misty Views
    # Create views and combine them
    receptor_view <- create_initial_view(as.data.frame(t(hvg_recep)))

    ligand_view <- create_initial_view(as.data.frame(t(hvg_lig))) %>%
        add_paraview(geometry, l = 200, family = "gaussian")

    lig_recep_view <- receptor_view %>% add_views(create_view("paraview.ligand.200", ligand_view[["paraview.200"]]$data, "para.lig.200"))

    run_misty(lig_recep_view, paste0(FigurePath_, "result_FS/lig_recep"), bypass.intra = TRUE)
    misty_results_lig_recep <- collect_results(paste0(FigurePath_, "result_FS/lig_recep"))

    ## Downstream Analysis
    pdf(paste0(FigurePath_, "FS para.lig.200 ligand receptors.pdf"), height = 6, width = 12)
    plot_interaction_heatmap(misty_results_lig_recep, "para.lig.200", clean = TRUE, cutoff = 2, trim.measure = "gain.R2", trim = 10)
    dev.off()
}
