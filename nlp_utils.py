from transformers import AutoTokenizer, AutoModel
import torch
import html
import heapq
import utils


def initialize_model_and_tokenizer(model_name="xlm-roberta-base"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


# Function to generate embeddings for a given text
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling to get a single vector for the text
    embeddings = outputs.last_hidden_state.mean(1)
    return embeddings


# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2)
    return cos_sim.item()


def text_similarity(text1, text2, tokenizer, model):
    # Initialize tokenizer and model
    tokenizer, model = initialize_model_and_tokenizer()

    # Get embeddings for both texts
    embedding1 = get_embedding(text1, tokenizer, model)
    embedding2 = get_embedding(text2, tokenizer, model)

    # Calculate and return the cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)
    return similarity_score


def get_total_cosine(segments1, segments2, tokenizer, model):
    total_similarity = 0
    for segment1, segment2 in zip(segments1, segments2):
        similarity = text_similarity(segment1, segment2, tokenizer, model)
        total_similarity += similarity
    return total_similarity


def match_segments(original_segments, translated_segments, tokenizer, model):
    translated_segments = [html.unescape(segment) for segment in translated_segments]
    heap = []
    # Initial combination generation and heap population
    if len(translated_segments) <= 1:
        return translated_segments
    for i in range(len(translated_segments) - 1):
        merged_segments = (
            translated_segments[:i]
            + [translated_segments[i] + " " + translated_segments[i + 1]]
            + translated_segments[i + 2 :]
        )
        score = get_total_cosine(original_segments, merged_segments, tokenizer, model)
        heapq.heappush(heap, (-score, merged_segments))

    # Continue processing the heap
    while len(heap[0][1]) > len(original_segments):
        _, current_segments = heapq.heappop(heap)
        for i in range(len(current_segments) - 1):
            new_segments = (
                current_segments[:i]
                + [current_segments[i] + " " + current_segments[i + 1]]
                + current_segments[i + 2 :]
            )
            score = get_total_cosine(original_segments, new_segments, tokenizer, model)
            heapq.heappush(heap, (-score, new_segments))

    return heap[0][1]
