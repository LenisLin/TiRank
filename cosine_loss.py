import torch.nn.functional as F

sc.tl.louvain(scAnndata,resolution=2)
clusterlabels=scAnndata.obs['louvain']


def cosine_loss(embeddings, labels):
    unique_labels = torch.unique(labels)
    total_similarity = 0
    count = 0

    for label in unique_labels:
        # 选择当前cluster的embeddings
        mask = (labels == label)
        cluster_embeddings = embeddings[mask]

        # 计算所有embedding pair之间的cosine similarity
        sim_matrix = F.cosine_similarity(cluster_embeddings.unsqueeze(1), cluster_embeddings.unsqueeze(0), dim=2)

        # 移除对角线元素，因为它们总是1
        sim_matrix.fill_diagonal_(0)

        # 计算总的相似性
        total_similarity += sim_matrix.sum()
        count += sim_matrix.numel()

    # 计算平均相似性
    avg_similarity = total_similarity / count

    # 我们希望最大化相似性，所以返回其负值作为loss
    return (1-avg_similarity)