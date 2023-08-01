library(survival)
library(survminer)
library(ggpubr)
library(ggsci)

clinical <- read.csv("/mnt/data/lyx/scRankv2/data/bulkClinical/Clinical_GSE38832.txt",sep="\t",header = T)
GSE38832riskScore <- read.table("/home/lyx/project/scRankv2/bulk2sc/GSE38832riskScore.csv", sep = ",", header = T)
colnames(GSE38832riskScore) <- c("geo_accession","riskscore")

clinical <- subset(clinical,dfs_event==1)

df2 <- dplyr::left_join(clinical,GSE38832riskScore,by = 'geo_accession')

        cutpoint <- surv_cutpoint(data = df2, time = "dfs_time", event = "dfs_event", variables = "riskscore")
        cutpoint <- summary(cutpoint)$cutpoint

        df <- df2[, c("riskscore", "dfs_event", "dfs_time")]
        df[, 1] <- ifelse(df[, 1] >= median(df2$riskscore), "high", "low")

        df$dfs_event <- as.numeric(df$dfs_event)
        df$dfs_time <- as.numeric(df$dfs_time)

        ## km curve
        fit <- survfit(Surv(dfs_time, dfs_event) ~ riskscore, data = df)
        p <- ggsurvplot(fit,
            data = df,
            linetype = c("solid", "solid"),
            surv.median.line = "hv", surv.scale = "percent",
            pval = T, risk.table = T,
            conf.int = T, conf.int.alpha = 0.1, conf.int.style = "ribbon",
            risk.table.y.text = T,
            palette = c("#3300CC", "#CC3300"),
            xlab = "Recurrence time"
        )

        pdf("test.pdf", width = 8, height = 6)
        print(p)
        dev.off()

a = coxph(Surv(dfs_time, dfs_event) ~ riskscore,data = df2)
summary(a)
