from transformers import AutoConfig, AutoModel
from ..utils.logging import log

class Embedding_Model:

    def __init__(self,embeddings_paths):
        self.embeddings_paths = embeddings_paths
        self.embeddings_config = AutoConfig.from_pretrained(self.embeddings_paths)
        self.embeddings_model = AutoModel.from_pretrained(self.embeddings_paths, trust_remote_code=True, device_map='auto')

    def embed_chunks(self, chunks, batch_size=2):
        log("INFO", "Embedding: Starting to embed chunks")
        embeddings = self.embeddings_model.encode(chunks, batch_size=batch_size)
        log("INFO", "Embedding: Finished embedding chunks")

        return embeddings

    def get_hidden_size(self):
        return self.embeddings_config.hidden_size

    def get_config(self):
        return self.embeddings_config

    def get_model(self):
        return self.embeddings_model


