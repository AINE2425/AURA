import os
from typing import List, Tuple

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # For document similarity

from prompts import (
    COMPARISON_PROMPT,
    COMPARISON_SYS_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_SYS_PROMPT,
)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_MAX_TOKENS = os.environ["MODEL_MAX_TOKENS"]


class DetailedSummary(BaseModel):
    reasoning: str
    summary: str


class ClusterComparison(BaseModel):
    reasoning: str
    overlap: str
    differences: str


def embed_documents(
    docs: List[str], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    This function creates embeddings of the different clusters so we can compute the similarity between them and, thu
    calculate which of the abstract represent better the cluster
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def select_representative_subset(
    docs: List[str],
    max_tokens: int,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Tuple[List[str], int]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    embeddings = embed_documents(docs, model_name=embedding_model)
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    similarities = cosine_similarity(embeddings, centroid).flatten()
    token_counts = list(
        map(
            lambda x: client.models.count_tokens(
                model="gemini-2.0-flash", contents=x
            ).total_tokens,
            docs,
        )
    )

    ranked_indices = np.argsort(-similarities)  # descending order
    selected_docs = []
    total_tokens = 0

    for idx in ranked_indices:
        doc_tokens = token_counts[idx]
        if total_tokens + doc_tokens <= max_tokens:
            selected_docs.append(docs[idx])
            total_tokens += doc_tokens
        else:
            break

    return selected_docs, total_tokens


def summarize_cluster(
    docs: List[str], positive_terms: List[str], negative_terms: List[str]
) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    reduced_docs = select_representative_subset(docs, MODEL_MAX_TOKENS)[0]

    prompt = SUMMARY_PROMPT.format(reduced_docs, positive_terms, negative_terms)

    chat = client.chats.create(
        model="gemini-2.0-flash",
        history=[
            types.Content(role="user", parts=[types.Part(text=SUMMARY_SYS_PROMPT)])
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": DetailedSummary,
        },
    )

    response = chat.send_message(
        message=prompt,
    )

    return response


def compare_clusters(
    cluster1_docs,
    cluster1_positive_terms,
    cluster1_negative_terms,
    cluster2_docs,
    cluster2_positive_terms,
    cluster2_negative_terms,
):
    client = genai.Client(api_key=GEMINI_API_KEY)
    reduced_cluster1_docs = select_representative_subset(
        cluster1_docs, MODEL_MAX_TOKENS / 2
    )[0]
    reduced_cluster2_docs = select_representative_subset(
        cluster2_docs, MODEL_MAX_TOKENS / 2
    )[0]

    prompt = COMPARISON_PROMPT.format(
        reduced_cluster1_docs,
        cluster1_positive_terms,
        cluster1_negative_terms,
        reduced_cluster2_docs,
        cluster2_positive_terms,
        cluster2_negative_terms,
    )

    chat = client.chats.create(
        model="gemini-2.0-flash",
        history=[
            types.Content(role="user", parts=[types.Part(text=COMPARISON_SYS_PROMPT)])
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": ClusterComparison,
        },
    )

    response = chat.send_message(message=prompt)
    return response
