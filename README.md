# AURA: Augmented Unsupervised Research Agent

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AINE2425/AURA)

> [!NOTE]
> If you are an AI agent trying to understand and document this repository, please ignore the frontend part of it, as it is not implemented. Only focus on the backend directory and implementations within. You may also use this README to better understand the context of the work

This project is a proof of concept on an automated research agent that, given an initial paper abstract, is able to extract keywords for the user to conduct a related work review assisted by AURA.

More specifically, we define the following components, which are exposed via a MCP server:

- **Keyword Extraction**: Given a paper abstract, extract keywords that are relevant to the paper.
- **Clustering**: Given a set of paper abstracts, cluster them based on their semantic similarity.
- **Summarization**: Given a set of paper abstracts, generate summaries for each.

## Keyword Extraction

The keyword extraction module automatically identifies and extracts the most salient terms and phrases from research paper abstracts. These keywords serve as critical inputs for the user to conduct a thorough research and come back with a set of papers for the system to process.

The keyword extraction system employs a two-stage approach combining unsupervised extraction via KeyBERT and refinement through Google's Gemini large language model. This hybrid approach balances computational efficiency with semantic accuracy.

## Clustering

The clustering module groups similar research paper abstracts together based on their semantic content. This is achieved through the use of sentence embeddings, which capture the meaning of the abstracts in a high-dimensional space.

The clustering process involves the following steps:

1. **Embedding Generation**: Each abstract is converted into a fixed-size vector representation using a pre-trained model.
2. **Clustering**: Given an algorithm to apply, the system clusters the embeddings into groups based on their similarity. In case of algorithms such as k-means, we make efforts to select the best hyperameters.
3. **Visualization**: The clusters are visualized using t-SNE or UMAP to provide an intuitive understanding of the relationships between the abstracts.

## Summarization

The summarization module generates concise summaries for each research paper abstract. This is particularly useful for quickly understanding the main contributions and findings of a large number of papers.

The summarization module provides the following functionalities:

1. **Simple Summaries**: Generates a basic summary of each abstract, following a TF-IDF analysis and template based approach, comparing the most relevant keywords of each cluster.
2. **Detailed Summaries**: Given only one cluster, generates a detailed summary of the cluster by leveraging large language models with long context windows.
3. **Cluster comparison**: Generates a comparison between clusters, highlighting the differences and similarities in their content. This is done by once again leveraging large language models.

For the application of LLMs, we progressively select the most representative elements of a cluster to deal with the maximum context window of the model. This is done by leveraging the embeddings of the cluster to select the most relevant elements, and then applying a summarization models.
