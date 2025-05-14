from typing import List

from clustering import cluster_abstracts
from keywords import abstract_to_keywords
from mcp.server.fastmcp import FastMCP
from simple_summary import summarize_clusters as simple_summarize_clusters

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


@mcp.tool()
async def keyword_extractor(abstract: str) -> str:
    """Give posible keywords from an abstract"""
    return abstract_to_keywords(abstract)


@mcp.tool()
async def clusterize_abstracts(abstracts: List[str]) -> List[List[str]]:
    """Clusterize the abstracts"""
    return cluster_abstracts(abstracts, algorithm="dbscan")


@mcp.tool()
async def simple_cluster_summary(clusters: List[List[str]]) -> List[SimpleSummary]:
    summaries = []
    for positive_keywords, negative_keywords, summary in simple_summarize_clusters(
        clusters
    ):
        summaries.append(SimpleSummary(positive_keywords, negative_keywords, summary))
    return summaries
