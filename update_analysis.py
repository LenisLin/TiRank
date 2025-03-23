import os

import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score, confusion_matrix

resultPath = "/home/lenislin/Experiment/projects/TiRankv2/github/TiRank/intera_0/Data4/3_Analysis"

r1 = pd.read_csv(os.path.join(resultPath,"spot_predict_score_MLP_1.csv"),index_col=0)
r2 = pd.read_csv(os.path.join(resultPath,"spot_predict_score_MLP_2.csv"),index_col=0)

r1.head()
r2.head()

# Step 1: Verify unique labels in both dataframes
labels = sorted(set(r1['Rank_Label'].unique()) | set(r2['Rank_Label'].unique()))
print("Unique labels in Rank_Label:", labels)

# Step 2: Calculate the agreement rate
agreement = (r1['Rank_Label'] == r2['Rank_Label']).mean()
print(f"Agreement rate: {agreement:.2%}")

# Step 3: Calculate Cohen's kappa
kappa = cohen_kappa_score(r1['Rank_Label'], r2['Rank_Label'])
print(f"Cohen's kappa: {kappa:.4f}")

# Step 4: Compute and display the confusion matrix
cm = confusion_matrix(r1['Rank_Label'], r2['Rank_Label'], labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\nConfusion matrix (rows: r1 labels, columns: r2 labels):")
print(cm_df)

# Code for Visualization Embedding
import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Combine the embeddings
combined_embeddings = np.vstack((embeddings_sc, embeddings_bulk))  # Shape: (804, 32)

# Create labels
sc_labels = ['scRNA-seq'] * embeddings_sc.shape[0]  # 507 entries

bulk_labels = bulkClinical.loc[:,"response"]
combined_labels = sc_labels + list(bulk_labels)     # 804 entries total (507 sc + 297 bulk)

# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(combined_embeddings)  # Shape: (804, 2)

# Define colors for each label
label_to_color = {
    'scRNA-seq': 'gray',
    1: 'red',
    0: 'blue'
}
colors = [label_to_color[label] for label in combined_labels]

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=colors, s=10, alpha=0.7)

# Add legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='scRNA-seq', markersize=5, markerfacecolor='gray'),
    Line2D([0], [0], marker='o', color='w', label='Rank+', markersize=5, markerfacecolor='red'),
    Line2D([0], [0], marker='o', color='w', label='Rank-', markersize=5, markerfacecolor='blue')
]
plt.legend(handles=legend_elements)

# Add titles and labels
plt.title('UMAP Visualization of scRNA-seq and Bulk RNA-seq Embeddings')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
#plt.savefig('test.png')
plt.show()