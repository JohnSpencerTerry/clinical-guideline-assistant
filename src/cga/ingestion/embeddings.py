"""Single shared embedding-model instance, used by both indexing and retrieval."""

from langchain_huggingface import HuggingFaceEmbeddings

from cga.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
