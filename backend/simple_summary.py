import random
from typing import List

import pandas as pd
import spacy
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def cluster_phrases(
    phrases: List[str], distance_threshold: float = 1.0
) -> List[List[str]]:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if len(phrases) <= 1:
        return [phrases]

    embeddings = model.encode(phrases)

    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="euclidean",
        linkage="ward",
    )
    clustering_model.fit(embeddings)
    labels = clustering_model.labels_

    clustered_phrases = {}
    for phrase, label in zip(phrases, labels):
        clustered_phrases.setdefault(label, []).append(phrase)

    return list(clustered_phrases.values())


def summarize_cluster(
    positive_words: List[str], negative_words: List[str], cluster_id: int
) -> str:
    positive_intro_templates = [
        "This cluster focuses primarily on {}.",
        "Key topics in this cluster include {}.",
        "This cluster is characterized by discussions on {}.",
    ]

    negative_intro_templates = ["It differest the most from clusters including {}"]

    clustered_positives = cluster_phrases(positive_words)

    # Select one representative per cluster
    representatives = [group[0] for group in clustered_positives]

    positives = (
        ", ".join(representatives[:-1]) + ", and " + representatives[-1]
        if len(representatives) > 1
        else representatives[0]
    )
    negatives = (
        ", ".join(negative_words[:-1]) + ", and " + negative_words[-1]
        if len(negative_words) > 1
        else negative_words[0]
    )

    summary = f"Cluster {cluster_id}: "
    summary += random.choice(positive_intro_templates).format(positives)

    if negative_words:
        summary += " " + random.choice(negative_intro_templates).format(negatives)

    return summary


# We will want to detect phrases such as "neural_network" to better identify clusters
def detect_phrases(
    tokenized_docs: List[List[str]], min_count: int = 5, threshold: float = 10.0
) -> List[List[str]]:
    phrases = Phrases(tokenized_docs, min_count=min_count, threshold=threshold)
    phraser = Phraser(phrases)
    return [phraser[doc] for doc in tokenized_docs]


def get_cluster_terms(cluster_idx, clusters, cluster_keywords):
    positive_terms = (
        cluster_keywords[cluster_idx].head(10)["word"].tolist()
    )  # More terms to allow clustering
    negative_terms = []
    for j in range(len(clusters)):
        if cluster_idx != j:
            diff = (
                cluster_keywords[cluster_idx]
                .merge(cluster_keywords[j], on="word", how="left")
                .fillna(0)
            )
            diff["diff"] = diff["tfidf_x"] - diff["tfidf_y"]
            negatives = (
                diff.sort_values("diff", ascending=True).head(3)["word"].tolist()
            )
            negative_terms.extend(negatives)
    negative_terms = list(set(negative_terms))  # Remove duplicates
    return positive_terms, negative_terms


def calculate_positive_and_negative_keywords(docs, clusters: List[List[str]]) -> None:
    tokenized_docs_per_cluster: List[List[List[str]]] = []

    # We lowercase everything to improve consistancy and differentiation.
    # We also convert to tokens before applying the detection of phrases
    for cluster_docs in docs:
        cluster_tokens: List[List[str]] = []
        for doc in cluster_docs:
            lemmas = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and not token.is_stop
            ]
            cluster_tokens.append(lemmas)
        tokenized_docs_per_cluster.append(cluster_tokens)

    all_tokenized_docs: List[List[str]] = [
        lemma for cluster in tokenized_docs_per_cluster for lemma in cluster
    ]
    all_tokenized_docs_with_phrases: List[List[str]] = detect_phrases(
        all_tokenized_docs
    )

    index = 0
    preprocessed_docs: List[str] = []

    # We merge the tokens into Strings to apply TF-IDF
    for cluster in tokenized_docs_per_cluster:
        cluster_text_tokens: List[str] = []
        for _ in cluster:
            cluster_text_tokens.extend(all_tokenized_docs_with_phrases[index])
            index += 1
        preprocessed_docs.append(" ".join(cluster_text_tokens))

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(), preprocessor=lambda x: x
    )
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords: List[pd.DataFrame] = []

    for i in range(tfidf_matrix.shape[0]):
        tfidf_scores = pd.DataFrame(
            {"word": feature_names, "tfidf": tfidf_matrix[i].toarray()[0]}
        )
        tfidf_scores = tfidf_scores.sort_values(by="tfidf", ascending=False)
        cluster_keywords.append(tfidf_scores)

    positive_keywords: List[List[str]] = []
    negative_keywords: List[List[str]] = []

    for i in range(len(clusters)):
        positive_terms, negative_terms = get_cluster_terms(
            i, clusters, cluster_keywords
        )
        positive_keywords.append(positive_terms)
        negative_keywords.append(negative_terms)

    return positive_keywords, negative_keywords


def summarize_clusters(clusters):
    """
    Summarizes clusters using Spacy's NLP pipeline and TF-IDF.

    :param clusters: List of clusters to summarize
    :return: List of summarized clusters
    """
    # Cargamos el modelo de spacy
    # Se puede cambiar por otro modelo, como 'es_core_news_sm' para español
    nlp = spacy.load("en_core_web_lg")

    docs = [
        list(
            tqdm(  # Decoramos con tqdm para ver barra de progreso
                nlp.pipe(cluster, n_process=-1), total=len(cluster)
            )
        )
        for cluster in clusters
    ]

    positive_keywords, negative_keywords = calculate_positive_and_negative_keywords(
        docs, clusters
    )

    summaries = []
    for i in range(len(clusters)):
        positive_terms = positive_keywords[i]
        negative_terms = negative_keywords[i]
        summaries.append(summarize_cluster(positive_terms, negative_terms, i + 1))

    return zip(clusters, summaries)
