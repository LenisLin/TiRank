# Gene-pairs (GP) extractor

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

from statsmodels.stats.multitest import multipletests


class GenePairExtractor():
    def __init__( bulk_expression, clinical_data, single_cell_expression, analysis_mode, top_var_genes=500, top_gene_pairs=2000, p_value_threshold=None, padj_value_threshold=None, max_cutoff=0.8, min_cutoff=-0.8):
        bulk_expression = bulk_expression
        clinical_data = clinical_data
        single_cell_expression = single_cell_expression
        analysis_mode = analysis_mode
        top_var_genes = top_var_genes
        top_gene_pairs = top_gene_pairs
        p_value_threshold = p_value_threshold
        padj_value_threshold = padj_value_threshold
        max_cutoff = max_cutoff
        min_cutoff = min_cutoff

    def run_extraction(self):
        print(f"Starting gene pair extraction.")

        # Find the intersection of genes in bulk and single-cell datasets
        intersect_genes = np.intersect1d(
            single_cell_expression.index, bulk_expression.index)
        intersect_single_cell_expression = single_cell_expression.loc[intersect_genes]

        # Sort genes by variance in the single-cell dataset
        gene_variances = np.var(intersect_single_cell_expression, axis=1)
        sorted_genes = gene_variances.sort_values(ascending=False)

        # Select the top variable genes
        top_variable_genes = sorted_genes[:top_var_genes].index.tolist()

        # Extract the candidate genes
        bulk_expression, single_cell_expression = extract_candidate_genes(
            top_variable_genes)

        print(f"Get candidate genes done.")

        # Obtain the list of candidate genes
        if analysis_mode == "Bionomial":
            regulated_genes_r, regulated_genes_p = calculate_binomial_gene_pairs()
        elif analysis_mode == "Cox":
            regulated_genes_r, regulated_genes_p = calculate_survival_gene_pairs()
        elif analysis_mode == "Regression":
            regulated_genes_r, regulated_genes_p = calculate_regression_gene_pairs()
        else:
            raise ValueError(f"Unsupported mode: {analysis_mode}")

        if (len(regulated_genes_r) == 0) or (len(regulated_genes_p) == 0):
            raise ValueError(
                "A set of genes is empty. Try increasing the 'top_var_genes' value or loosening the 'p.value' threshold.")

        print(f"Get candidate gene pairs done.")

        # Transform the bulk gene pairs
        bulk_gene_pairs = transform_bulk_gene_pairs(
            regulated_genes_r, regulated_genes_p)
        # Filter the gene pairs
        bulk_gene_pairs_mat = filter_gene_pairs(bulk_gene_pairs)

        # Transform the single-cell gene pairs
        single_cell_gene_pairs_mat = transform_single_cell_gene_pairs(
            bulk_gene_pairs_mat)

        print(f"Profile transformation done.")

        # Return the bulk and single-cell gene pairs
        return bulk_gene_pairs_mat, single_cell_gene_pairs_mat

    def extract_candidate_genes(gene_names):
        # Construct gene pairs
        single_cell_gene_subset = single_cell_expression.loc[gene_names]
        bulk_gene_subset = bulk_expression.loc[gene_names, :]

        # Remove rows in bulk dataset where all entries are 0
        bulk_gene_subset = bulk_gene_subset.loc[(
            bulk_gene_subset != 0).any(axis=1)]
        gene_names = bulk_gene_subset.index.tolist()
        single_cell_gene_subset = single_cell_gene_subset.loc[gene_names]

        return bulk_gene_subset, single_cell_gene_subset

    def calculate_binomial_gene_pairs(self):
        # Calculate group means and perform t-test
        group_labels = clinical_data.iloc[:, 0]
        group_0 = bulk_expression.loc[:, group_labels == 0]
        group_1 = bulk_expression.loc[:, group_labels == 1]

        # Calculate t-tests and log fold changes
        p_values = []
        t_stats = []
        # logFCs = []

        for gene in bulk_expression.index:
            t_stat, p_value = ttest_ind(group_0.loc[gene], group_1.loc[gene])
            t_stats.append(t_stat)
            p_values.append(p_value)

            mean_0 = group_0.loc[gene].mean()
            mean_1 = group_1.loc[gene].mean()

            # Add a small constant to avoid division by zero
            # logFC = np.log2((mean_1 + 1e-10) / (mean_0 + 1e-10))
            # logFCs.append(logFC)

        # Store the results in a DataFrame
        DEGs = pd.DataFrame({
            # 'logFC': logFCs,
            'AveExpr': bulk_expression.mean(axis=1),
            't': t_stats,
            'P.Value': p_values,
            'gene': bulk_expression.index
        })

        # Drop the row which p_values is NULL
        DEGs = DEGs.dropna()

        # Adjust p-values for multiple testing
        DEGs['adj.P.Val'] = multipletests(DEGs['P.Value'], method='fdr_bh')[1]

        # Filter significant genes
        if (p_value_threshold is None) and (padj_value_threshold is not None):
            DEGs = DEGs[DEGs['adj.P.Val'] < padj_value_threshold]
        elif (p_value_threshold is not None) and (padj_value_threshold is None):
            DEGs = DEGs[DEGs['P.Value'] < p_value_threshold]
        else:
            raise ValueError(
                "The significant value threshold was not defined.")

        # Separate up- and down-regulated genes
        # regulated_genes_r = DEGs[DEGs['logFC'] > 0]['gene'].tolist()
        # regulated_genes_p = DEGs[DEGs['logFC'] < 0]['gene'].tolist()

        regulated_genes_0 = DEGs[DEGs['t'] > 0]['gene'].tolist()
        regulated_genes_1 = DEGs[DEGs['t'] < 0]['gene'].tolist()
        return regulated_genes_0, regulated_genes_1

    def calculate_survival_gene_pairs(self):
        # Perform univariate Cox analysis on the bulk dataset using CoxPHFitter
        survival_results = pd.DataFrame(columns=["gene", "HR", "p_value"])
        for i in range(bulk_expression.shape[0]):
            exp_gene = bulk_expression.iloc[i, :].astype(float)
            clinical_temp = pd.concat([clinical_data, exp_gene], axis=1)
            cph = CoxPHFitter()
            cph.fit(
                clinical_temp, duration_col=clinical_data.columns[0], event_col=clinical_data.columns[1])
            hr = cph.summary["exp(coef)"].values[0]
            p_value = cph.summary["p"].values[0]
            survival_results = survival_results.append(
                {"gene": bulk_expression.index[i], "HR": hr, "p_value": p_value}, ignore_index=True)

        survival_results = survival_results.dropna()
        survival_results["HR"] = survival_results["HR"].astype(float)
        survival_results["p_value"] = survival_results["p_value"].astype(float)

        survival_results['adj.P.Val'] = multipletests(
            survival_results['p_value'], method='fdr_bh')[1]

        # Filter significant genes
        if (p_value_threshold is None) and (padj_value_threshold is not None):
            survival_results = survival_results[survival_results['adj.P.Val']
                                                < padj_value_threshold]
        elif (p_value_threshold is not None) and (padj_value_threshold is None):
            survival_results = survival_results[survival_results['p_value']
                                                < p_value_threshold]
        else:
            raise ValueError(
                "The significant value threshold was not defined.")

        # Construct gene pairs for HR>1 and HR<1 separately
        regulated_genes_r = survival_results[survival_results['HR'] > 1]['gene']
        regulated_genes_p = survival_results[survival_results['HR'] < 1]['gene']

        return regulated_genes_r, regulated_genes_p

    def calculate_regression_gene_pairs(self):
        # Bulk dataset Pearson correlation. Define gene pairs based on correlation coefficient and p-value
        correlation_results = pd.DataFrame(
            columns=["gene", "correlation", "pvalue"])
        for i in range(bulk_expression.shape[0]):
            exp_gene = bulk_expression.iloc[i, :].astype(float)
            correlation, pvalue = pearsonr(
                exp_gene, clinical_data.iloc[:, 0])

            correlation_results = pd.concat([
                correlation_results,pd.Series({"gene": bulk_expression.index[i], 
                "correlation": correlation, 
                "pvalue": pvalue}).to_frame().T], axis=0, ignore_index=True)


        correlation_results = correlation_results.dropna()
        correlation_results["correlation"] = correlation_results["correlation"].astype(
            float)
        correlation_results["pvalue"] = correlation_results["pvalue"].astype(
            float)

        correlation_results['adj.P.Val'] = multipletests(
            correlation_results['pvalue'], method='fdr_bh')[1]

        # Filter significant genes
        if (p_value_threshold is None) and (padj_value_threshold is not None):
            correlation_results = correlation_results[correlation_results['adj.P.Val']
                                                      < padj_value_threshold]

        elif (p_value_threshold is not None) and (padj_value_threshold is None):
            correlation_results = correlation_results[correlation_results['pvalue']
                                                      < p_value_threshold]

        else:
            raise ValueError(
                "The significant value threshold was not defined.")

        # Define gene pairs based on whether correlation is >0 or <0
        positive_correlation_genes = correlation_results[correlation_results['correlation'] > 0]['gene']
        negative_correlation_genes = correlation_results[correlation_results['correlation'] < 0]['gene']

        return positive_correlation_genes, negative_correlation_genes

    # Construct bulk gene pairs
    def transform_bulk_gene_pairs( genes_r, genes_p):
        # Get genes
        exp1 = bulk_expression.loc[genes_r]
        exp2 = bulk_expression.loc[genes_p]

        # Compute result matrix
        result_values = np.where(exp1.values[:, None] > exp2.values, 1, -1)
        result_values = np.vstack(result_values)

        # Create result DataFrame
        row_names = [f"{i}__{j}" for i in genes_r for j in genes_p]
        result_df = pd.DataFrame(
            result_values, index=row_names, columns=bulk_expression.columns)

        return result_df

    def filter_gene_pairs( bulk_GPMat):
        # Filter results of gene pair construction. max_cutoff and min_cutoff define the upper and lower proportions
        bulk_GPMat = bulk_GPMat[(np.sum(bulk_GPMat, axis=1) < max_cutoff * bulk_GPMat.shape[1]) &
                                (np.sum(bulk_GPMat, axis=1) > min_cutoff * bulk_GPMat.shape[1])]

        if bulk_GPMat.shape[0] >= top_gene_pairs:
            # Compute variance of gene pairs and sort
            gene_pair_variances = np.var(bulk_GPMat, axis=1)
            sorted_gene_pairs = gene_pair_variances.sort_values(
                ascending=False)
            # Select top variable gene pairs
            top_var_gene_pairs = sorted_gene_pairs[:top_gene_pairs].index.tolist(
            )
            bulk_GPMat = bulk_GPMat.loc[top_var_gene_pairs]

        return bulk_GPMat

    def transform_single_cell_gene_pairs( bulk_GPMat):
        # Get gene pairs
        gene_pairs = bulk_GPMat.index.tolist()
        # Split gene pairs
        genes_1, genes_2 = split_gene_pairs(gene_pairs)

        # Construct gene pairs
        exp1 = single_cell_expression.loc[genes_1]
        exp2 = single_cell_expression.loc[genes_2]

        result = np.where(exp1.values > exp2.values, 1, -1)

        # Create result DataFrame
        result_df = pd.DataFrame(
            result, index=bulk_GPMat.index, columns=single_cell_expression.columns)

        return result_df

    def split_gene_pairs( gene_pairs):
        # Split gene pairs to get two lists of genes, gene1 and gene2.
        # gene1 contains the genes in the first position of gene_pairs, gene2 contains the second genes.
        gene1 = [x.split("__")[0] for x in gene_pairs]
        gene2 = [x.split("__")[1] for x in gene_pairs]

        return gene1, gene2
