import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances


def evaluate_clustering(X, labels):
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        return None
    return {
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
    }


def run_kmeans(X, k_list):
    results, inertias = [], []
    for k in k_list:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X)
        inertia = model.inertia_
        inertias.append(inertia)
        metrics = evaluate_clustering(X, labels)
        if metrics:
            results.append(
                {
                    "method": "KMeans",
                    "params": {"k": k},
                    "labels": labels,
                    "inertia": inertia,
                    **metrics,
                }
            )
    elbow_point = None
    if len(inertias) >= 3:
        second_derivative = np.diff(inertias, n=2)
        elbow_index = np.argmin(second_derivative) + 1
        elbow_point = k_list[elbow_index]
    return {"results": results, "elbow_point": elbow_point}


def run_dbscan(X, eps_list, min_samples=3):
    dists = cosine_distances(X)
    results = []
    for eps in eps_list:
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = model.fit_predict(dists)
        metrics = evaluate_clustering(X, labels)
        if metrics:
            results.append(
                {
                    "method": "DBSCAN",
                    "params": {"eps": eps, "min_samples": min_samples},
                    "labels": labels,
                    **metrics,
                }
            )
    return results


def run_optics(X, xi_list):
    results = []
    for xi in xi_list:
        model = OPTICS(min_samples=3, metric="cosine", xi=xi)
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        if metrics:
            results.append(
                {"method": "OPTICS", "params": {"xi": xi}, "labels": labels, **metrics}
            )
    return results


def run_agglomerative(X, k_list):
    results = []
    for k in k_list:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        if metrics:
            results.append(
                {
                    "method": "Agglomerative",
                    "params": {"k": k},
                    "labels": labels,
                    **metrics,
                }
            )
    return results


def cluster_abstracts(abstracts, algorithm="kmeans", params=None):
    if not isinstance(abstracts, list) or not abstracts:
        raise ValueError("Input 'abstracts' must be a non-empty list of strings.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(abstracts, show_progress_bar=True)

    params = params or {}
    results = []

    if algorithm == "kmeans":
        k_list = params.get("k_list", list(range(2, len(abstracts))))
        output = run_kmeans(embeddings, k_list)
        results = output["results"]
        best = max(
            results,
            key=lambda x: np.mean(
                [x["silhouette"], 1 - x["davies_bouldin"], x["calinski_harabasz"]]
            ),
        )
    elif algorithm == "dbscan":
        eps_list = params.get("eps_list", [0.3, 0.4, 0.5])
        min_samples = params.get("min_samples", 3)
        results = run_dbscan(embeddings, eps_list, min_samples)
        if not results:
            raise ValueError(
                "DBSCAN could not form any valid clusters with the given parameters."
            )
        best = max(
            results,
            key=lambda x: np.mean(
                [x["silhouette"], 1 - x["davies_bouldin"], x["calinski_harabasz"]]
            ),
        )
    elif algorithm == "optics":
        xi_list = params.get("xi_list", [0.03, 0.05, 0.07])
        results = run_optics(embeddings, xi_list)
        best = max(
            results,
            key=lambda x: np.mean(
                [x["silhouette"], 1 - x["davies_bouldin"], x["calinski_harabasz"]]
            ),
        )
    elif algorithm == "agglomerative":
        k_list = params.get("k_list", list(range(2, len(abstracts))))
        results = run_agglomerative(embeddings, k_list)
        best = max(
            results,
            key=lambda x: np.mean(
                [x["silhouette"], 1 - x["davies_bouldin"], x["calinski_harabasz"]]
            ),
        )
    else:
        raise ValueError("Invalid clustering algorithm specified.")

    clusters = {}
    for abstract, label in zip(abstracts, best["labels"]):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(abstract)
    cluster_list = [clusters[label] for label in sorted(clusters.keys())]
    return cluster_list


def plot_clusters(X_proj, labels, title, is_3d=False):
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    if is_3d:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                X_proj[mask, 2],
                label=f"Cluster {label}",
                s=20,
                color=colors(i),
            )
        ax.set_title(title)
        ax.legend()
    else:
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                label=f"Cluster {label}",
                s=20,
                color=colors(i),
            )
        plt.title(title)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
    plt.tight_layout()
    plt.show()
