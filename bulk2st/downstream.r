library(Seurat)

library(ggplot2)
library(ggpubr)
library(cowplot)

dataPath = "/home/lenislin/Experiment/data/scRankv2/data/ST/CRC/"
savePath = "./tempfiles/"
figurePath = paste0(savePath, "figures/")

slices <- list.files(dataPath)

for(slice_ in slices){
    ## Load spatial slice
    score <- read.csv(paste0(savePath,"pred_score/",slice_,"_predict_score.csv"),row.names = 1)
    stAnndata <- Load10X_Spatial(paste0(dataPath,slice_))

    ## Add information
    stAnndata <- AddMetaData(stAnndata,score)
    Rank_Label <- (1-stAnndata$Reject)*stAnndata$Rank_Score
    Rank_Label <- ifelse(Rank_Label == 0, "Background",ifelse(Rank_Label>0.5,"scRank+","scRank-"))
    table(stAnndata$Rank_Label)
    stAnndata <- AddMetaData(stAnndata,as.factor(Rank_Label),"Rank_Label")

    ## Spatial feature plot
    pdf(paste0(figurePath,slice_," clusters.pdf"),height = 6,width = 8)
    Idents(stAnndata) <- stAnndata$clusters
    SpatialDimPlot(stAnndata,)
    dev.off()

    pdf(paste0(figurePath,slice_," pathology cluster.pdf"),height = 6,width = 8)
    Idents(stAnndata) <- stAnndata$patho_class
    SpatialDimPlot(stAnndata)
    dev.off()

    pdf(paste0(figurePath,slice_," scRank cluster.pdf"),height = 6,width = 8)
    Idents(stAnndata) <- stAnndata$Rank_Label
    SpatialDimPlot(stAnndata)
    dev.off()
}