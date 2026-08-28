"""Single shared embedding-model instance, used by both indexing and retrieval."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from cga.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
