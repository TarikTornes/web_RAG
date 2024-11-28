from langchain_experimental.text_splitter import SemanticChunker
from typing import Optional
from .custom_embeddings import CustomEmbeddings

class SemanticChunkerConfig:
    @classmethod
    def create(cls, embeddings_path: str = "jinaai/jina-embeddings-v2-base-en",
                device: Optional[str] = "mps", 
                breakpoint_threshold_type: str = "percentile"):
        embeddings = CustomEmbeddings(embeddings_path, device)

        return SemanticChunker(embeddings, 
                               breakpoint_threshold_type=breakpoint_threshold_type
        )


def chunker(document):

    text_splitter = SemanticChunkerConfig().create()
    chunks = text_splitter.create_documents([document.strip()])

    return chunks

