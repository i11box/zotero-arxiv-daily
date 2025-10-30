from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime
from paper import ArxivPaper
from sentence_transformers import SentenceTransformer

def rerank_paper(candidate: list[ArxivPaper], corpus: list[dict], model: str = "google/embeddinggemma-300m", n_classes = 3) -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    # sort corpus by date, from newest to oldest
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    
    # K-Means clustering on corpus features
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(corpus_feature)
    labels = kmeans.labels_
    class_counts = np.bincount(labels, minlength=n_classes)
    class_weights = class_counts / class_counts.sum()  # normalize
    cluster_weight = class_weights[labels]  # shape: [n_corpus]

    # Combine time decay and cluster size weight
    combined_weight = time_decay_weight * cluster_weight
    combined_weight = combined_weight / combined_weight.sum()  # re-normalize

    sim = encoder.similarity(candidate_feature, corpus_feature)  # [n_candidate, n_corpus]
    scores = (sim * combined_weight).sum(axis=1) * 10  # [n_candidate]
    
    # 归一化到0-10区间
    min_score = scores.min()
    max_score = scores.max()
    # 避免除零情况
    if max_score > min_score:
        scores = (scores - min_score) / (max_score - min_score) * 10
    else:
        scores = np.full_like(scores, 5.0)  # 如果所有分数相同，则设为中间值5.0
    
    for s, c in zip(scores, candidate):
        c.score = s.item()
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    return candidate