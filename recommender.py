import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import torch

def visualize_clusters(corpus: list[dict], model: str = "google/embeddinggemma-300m", n_classes = 2, save_path: str = "clusters.png"):
    """
    Visualize paper clusters with paper IDs labeled on the plot.

    Parameters:
    corpus: List of dictionaries containing paper metadata
    model: Model name for SentenceTransformer
    n_classes: Number of clusters
    save_path: Path to save the visualization
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model, device=device)
    
    # Sort corpus by date, from newest to oldest
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    
    # K-Means clustering on corpus features
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(corpus_feature)
    labels = kmeans.labels_
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    corpus_2d = pca.fit_transform(corpus_feature)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot papers with cluster colors
    for i in range(n_classes):
        cluster_points = corpus_2d[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.6, label=f'Cluster {i+1}')
    
    # Number papers by their index (1-based)
    for idx, point in enumerate(corpus_2d):
        plt.annotate(str(idx+1), (point[0], point[1]), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.title('Paper Clustering Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save mapping between paper numbers and titles to a JSON file
    paper_mapping = {}
    for idx, paper in enumerate(corpus):
        paper_mapping[idx+1] = {
            'title': paper['data']['title'],
            'date': paper['data']['dateAdded']
        }
    
    mapping_file = save_path.replace('.png', '_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(paper_mapping, f, indent=2, ensure_ascii=False)
    
    # Print mapping to console
    print("\nPaper Number to Title Mapping:")
    print("="*50)
    for idx, paper in enumerate(corpus):
        print(f"{idx+1}: {paper['data']['title']}")
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model, device=device)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate