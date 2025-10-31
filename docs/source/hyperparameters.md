# Hyperparameters

### Feature Selection Hyperparameters

1.  **`top_var_genes`**:
    * Default: `2000`
    * Recommendation: Increase if gene selection is insufficient.

2.  **`p_value_threshold`**:
    * Default: `0.05`
    * Recommendation: Adjust based on significance needs.

3.  **`top_gene_pairs`**:
    * Default: `2000`

### Model Training Hyperparameters

1.  **`alphas`**:
    * Balances loss components during training.

2.  **`n_epochs`**:
    * Increase for better convergence.

3.  **`lr`**:
    * Lower values ensure stable convergence.