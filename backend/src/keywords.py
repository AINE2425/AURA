import os

import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from keybert import KeyBERT
from transformers import AutoModel, AutoTokenizer

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    pass
else:
    raise EnvironmentError("GEMINI_API_KEY environment variable not found.")

# Load model for sentence embedding
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Mean pooling function to get a single vector per text
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Function to get text embeddings
def embed_text(texts):
    if isinstance(texts, str):
        texts = [texts]

    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return embeddings


def extract_key_phrases(abstract):
    kw_model = KeyBERT(model=embed_text)
    keywords = kw_model.extract_keywords(
        abstract,
        keyphrase_ngram_range=(1, 3),  # Key phrases from 1 to 3 words
        stop_words="english",
        top_n=10,  # Number of key phrases to extract
    )

    # Extract key phrases from KeyBERT's result
    key_phrases = [kw[0] for kw in keywords]

    return key_phrases


# Function to build the prompt for Gemini
def build_prompt(abstract, key_phrases):
    key_phrases_str = ", ".join(key_phrases)
    prompt = f"""
    Given the following abstract and a list of detected key phrases, generate 5 clean and normalized keywords that best summarize the article.
    Only provide the keywords, separated by commas, with no additional information.

    Abstract:
    {abstract}

    Detected Key Phrases:
    {key_phrases_str}

    Keywords (separated by commas):
    """
    return prompt


def generate_keywords_gemini(prompt):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    chat = client.chats.create(
        model="gemini-2.0-flash",
        history=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config={
            "response_mime_type": "text/plain",
        },
    )
    response = chat.send_message(message=prompt)
    return response.text.strip()


def abstract_to_keywords(abstract):
    # Step 1: Extract key phrases from the abstract
    key_phrases = extract_key_phrases(abstract)

    # Step 2: Build the prompt for Gemini
    prompt = build_prompt(abstract, key_phrases)

    # Step 3: Generate keywords using Gemini
    keywords = generate_keywords_gemini(prompt)

    return keywords
