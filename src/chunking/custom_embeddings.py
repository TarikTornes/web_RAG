from sentence_transformers import SentenceTransformer
from typing import List

class CustomEmbeddings:
    '''
    This class follows the structure of the embeddings interface given by the SemanticChunker.
    See https://python.langchain.com/v0.2/api_reference/core/embeddings/langchain_core.embeddings.
    embeddings.Embeddings.html#langchain_core.embeddings.embeddings.Embeddings
    Thus the embedding needs a function embed_documents and embed_query
    '''
    def __init__(self, model, device='mps'):
        # modified by TT
        self.model = SentenceTransformer(model, device=device, trust_remote_code=True)
        #self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        '''
        .toList allows us to transform the result, which is a list of numpy arrays, 
        into a list of lists thus the tensors (i.e. the numpy arrays will be lists of
        floats, which is necessary due to the given embeddings interface by the semantic chunker

        @param texts: Is the document which is a list of strings
        @returns : a list of lists, where the inner lists represent the embedding vectors
        '''
        return [self.model.encode(t).tolist() for t in texts]

    #def embed_documents_tt(self, texts: List[str]):
    #    return [self.model.encode(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])

