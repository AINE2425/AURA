from typing import List

from src.clustering import cluster_abstracts
from src.keywords import abstract_to_keywords
from mcp.server.fastmcp import FastMCP
from src.simple_summary import summarize_clusters as simple_summarize_clusters

mcp = FastMCP("AURA - Augmented Unsupervised Research Analyzer")


class SimpleSummary:
    def __init__(
        self, positive_words: List[str], negative_words: List[str], summary: str
    ):
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.summary = summary

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
    Given an abstract, return a list of keywords.
    Example:
    "This is an abstract about machine learning and its applications."
    will return:
    [
        "machine learning",
        "applications"
    ]
    """
    return abstract_to_keywords(abstract)


@mcp.tool()
async def clusterize_abstracts(abstracts: List[str]) -> List[List[str]]:
    """
    Given a list of abstracts, cluster them using the DBSCAN algorithm.
    Each cluster is represented as a list of abstracts.
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
    return cluster_abstracts(abstracts, algorithm="dbscan")


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

def main():
    """
    Main function to run the FastMCP server.
    """
    mcp.run(transport="stdio")

def hello():
    """
    A simple hello world function.
    """
    print("hello world")

if __name__ == "__main__":
    main()