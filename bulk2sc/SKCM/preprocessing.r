datapath = "/mnt/data/lyx/scRankv2/data/RNAseq_treatment/Melanoma/ims_gene_signature/data/"
load_ids <- c("gide19", "hugo16", "liu19", "puch", "Riaz17", "van")
save_ids <- c("Gide2019", "Hugo2016", "Liu2019", "PUCH2021", "Riaz2017", "VanAllen2015")

for (i in 1:length(load_ids)) {
    load_id <- load_ids[i]
    save_id <- save_ids[i]

    savePath <- paste0("/mnt/data/lyx/scRankv2/data/RNAseq_treatment/Melanoma/")

    exp <- read.csv(paste0(datapath, "mel_", load_id, "_exp_data.csv"), row.names = 1)
    sur <- read.csv(paste0(datapath, "mel_", load_id, "_survival_data.csv"), row.names = 1)
    cli <- read.csv(paste0(datapath, "mel_", load_id, "_cli_data.csv"), row.names = 1)

    if (i == 4) {
        rownames(sur) <- paste0("X", rownames(sur))
        rownames(cli) <- paste0("X", rownames(cli))
        colnames(exp) <- sub(pattern = "\\.", replacement = "-", colnames(exp))
    }

    if (max(sur$OS) > 180) {
        sur$OS <- sur$OS / 30
    }
    sur$OS <- round(sur$OS, digits = 4)

    cli$response <- ifelse(cli$response == 1, "CRPR", ifelse(cli$response == 0, "PD", "SD"))

    intersectidx <- intersect(colnames(exp), intersect(rownames(sur), rownames(cli)))
    if (length(intersectidx) == 0) {
        cat("Data", load_id, "do not have match id!", "\n")
        next
    }

    exp <- exp[, match(intersectidx, colnames(exp))]
    sur <- sur[match(intersectidx, rownames(sur)), ]
    cli <- cli[match(intersectidx, rownames(cli)), ]

    meta <- cbind(cli$response, sur$status, sur$OS)
    meta <- as.data.frame(meta)
    colnames(meta) <- c("Response", "OS_status", "OS_time")

    cat("The size of", load_id, "is", length(intersectidx), "with CR-PR:", nrow(meta[meta$Response == "CRPR", ]), "PD:", nrow(meta[meta$Response == "PD", ]), "SD:", nrow(meta[meta$Response == "SD", ]), "\n")

    if (!dir.exists(savePath)) {
        dir.create(savePath, recursive = T)
    }

    write.csv(meta, paste0(savePath, save_id, "_meta.csv"))
    write.csv(exp, paste0(savePath, save_id, "_exp.csv"))
}
