from typing import List

import numpy as np
import plotly.express as px
from clustering import cluster_abstracts
from detailed_summary import compare_clusters
from detailed_summary import summarize_cluster as detailed_summarize_clusters
from fastmcp import FastMCP
from keywords import abstract_to_keywords
from sentence_transformers import SentenceTransformer
from simple_summary import summarize_clusters as simple_summarize_clusters
from sklearn.decomposition import PCA

mcp = FastMCP(
    name="AURA - Augmented Unsupervised Research Analyzer",
    instructions="""
    You are a very experienced and skilled researchered and reviewer.
    You can only use the tools AURA provides for summarizing, clustering, or compare.
    """,
)


class SimpleSummary:
    def __init__(
        self, positive_words: List[str], negative_words: List[str], summary: str
    ):
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.summary = summary

    def __str__(self) -> str:
        return f"positive_words: {self.positive_words}, negative_words: {self.negative_words}, summary: {self.summary}"

    def to_json(self) -> dict:
        return {
            "positive_words": self.positive_words,
            "negative_words": self.negative_words,
            "summary": self.summary,
        }


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return {"status": "healthy"}


@mcp.tool()
async def keyword_extractor(abstract: str) -> str:
    """
    Extract keywords from a given abstract.

    Args:
        abstract (str): The abstract text to extract keywords from.

    Returns:
        str: A list of keywords extracted from the abstract.
    """
    return abstract_to_keywords(abstract)


@mcp.tool()
async def clusterize_abstracts(abstracts: List[str]) -> List[List[str]]:
    """
    Given a list of abstracts, cluster them using the DBSCAN algorithm.
    Each cluster is represented as a list of abstracts.
    DO NOT GIVE ANY SUMMARY ON THE CLUSTERS
    You can offer a 2d or 3d visualization.
    Example:
    [
        "abstract1",
        "abstract2",
        "abstract3"
    ]

    Output example:
    [
        ["abstract1", "abstract2"],
        ["abstract3"]
    ]
    """
    clusters, embeddings, labels = cluster_abstracts(abstracts, algorithm="dbscan")
    return clusters, embeddings, labels


@mcp.tool()
async def viz_clusters(
    abstracts: List[str],
    labels: List[int],
    clusters_names: List[str],
    title: str,
    is_3d: bool,
) -> None:
    """
    Visualize clusters based on their abstracts, labels, and names.

    Args:
        abstracts (List[str]): A list of abstracts to compute embeddings for.
        labels (List[int]): A list of cluster labels corresponding to the embeddings.
        clusters_names (List[str]): A list with the clusters names.
        title (str): The title of the visualization.
        is_3d (bool): Whether to create a 3D visualization (True) or a 2D visualization (False).
    """

    def truncate(text: str, max_length: int = 200) -> str:
        return (
            text
            if len(text) <= max_length
            else text[:max_length].rsplit(" ", 1)[0] + "…"
        )

    truncated_abstracts = [truncate(a, max_length=75) for a in abstracts]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings: np.ndarray = model.encode(abstracts, show_progress_bar=True)

    n_components = 3 if is_3d else 2
    pca = PCA(n_components=n_components, random_state=42)
    reduced: np.ndarray = pca.fit_transform(embeddings)

    label_names = [
        clusters_names[l] if l < len(clusters_names) else f"Cluster {l}" for l in labels
    ]

    data = {
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": label_names,
        "abstract": truncated_abstracts,
    }

    if is_3d:
        data["z"] = reduced[:, 2]
        fig = px.scatter_3d(
            data,
            x="x",
            y="y",
            z="z",
            color="label",
            title=title,
            hover_data={"abstract": True},
        )
    else:
        fig = px.scatter(
            data,
            x="x",
            y="y",
            color="label",
            title=title,
            hover_data={"abstract": True},
        )

    fig.show()

    return "visualization ended"


@mcp.tool()
async def simple_cluster_summary(clusters: List[List[str]]) -> List[SimpleSummary]:
    """
    Given a list of clusters, each of which is represented as a list of abstracts,
    return a list of SimpleSummary objects.
    Each SimpleSummary object contains positive keywords, negative keywords,
    and a summary of the cluster.

    Example:
    [
        ["abstract1", "abstract2"],
        ["abstract3", "abstract4"]
    ]
    will return a list of SimpleSummary objects, one for each cluster.
    SimpleSummary objects contain:
    - positive_keywords: List of keywords that are present in the abstracts of the cluster
    - negative_keywords: List of keywords that are not present in the abstracts of the cluster
    - summary: A summary of the cluster
    """
    summaries = []
    for positive_keywords, negative_keywords, summary in simple_summarize_clusters(
        clusters
    ):
        summaries.append(SimpleSummary(positive_keywords, negative_keywords, summary))
    return summaries


@mcp.tool()
async def detailed_cluster_summary(
    docs: List[str], positive_terms: List[str], negative_terms: List[str]
) -> str:
    """
    Generate a detailed summary for a cluster of documents. DEPENDS ON SIMPLE SUMMARY

    Args:
        docs (List[str]): A list of documents (abstracts) in the cluster.
        positive_terms (List[str]): A list of positive terms associated with the cluster.
        negative_terms (List[str]): A list of negative terms associated with the cluster.

    Returns:
        str: A detailed summary of the cluster, including reasoning and key insights.
    """

    return detailed_summarize_clusters(docs, positive_terms, negative_terms)


@mcp.tool()
async def compare(
    cluster1_docs: List[str],
    cluster1_positive_terms: List[str],
    cluster1_negative_terms: List[str],
    cluster2_docs: List[str],
    cluster2_positive_terms: List[str],
    cluster2_negative_terms: List[str],
) -> str:
    """
    Compare two clusters of documents and generate a detailed comparison.

    Args:
        cluster1_docs (List[str]): A list of documents in the first cluster.
        cluster1_positive_terms (List[str]): A list of positive terms for the first cluster.
        cluster1_negative_terms (List[str]): A list of negative terms for the first cluster.
        cluster2_docs (List[str]): A list of documents in the second cluster.
        cluster2_positive_terms (List[str]): A list of positive terms for the second cluster.
        cluster2_negative_terms (List[str]): A list of negative terms for the second cluster.

    Returns:
        str: A detailed comparison of the two clusters, including reasoning, overlap, and differences.
    """
    return compare_clusters(
        cluster1_docs,
        cluster1_positive_terms,
        cluster1_negative_terms,
        cluster2_docs,
        cluster2_positive_terms,
        cluster2_negative_terms,
    )


def main():
    """
    Main function to run the FastMCP server.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
