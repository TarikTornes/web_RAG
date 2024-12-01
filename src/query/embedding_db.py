from ..utils.logging import log
from ..utils.computations import cos_sim

import faiss, re
from transformers import AutoConfig, AutoModel


class EmbeddingDB:

    def __init__(self, embeddings_path, embeddings, chunks_dict):
        self.embeddings_config = AutoConfig.from_pretrained(embeddings_path)
        self.embeddings_model = AutoModel.from_pretrained(embeddings_path)
        self.hidden_size = self.embeddings_config.hidden_size

        self.index = faiss.IndexFlatL2(self.hidden_size)
        self.index = self.index.add(embeddings)
        self.embeddings = embeddings
        self.chunks_dict = chunks_dict
        log("INFO", "Query: EmbeddingDB successfully loaded")


    def get_k_Results(self, QUERY, k):
        results = []

        query = self.embeddings_model.encode([QUERY])

        D, I = self.index.search(query, k)

        for i, j in enumerate(I[0]):
            vec = self.embeddings[j]
            chunk = self.chunks_dict[j]
            results.append((j,chunk))
            log("QUERY_RESULTS", "CHUNK", i, j, round(D[0][i], 3), \
                round(cos_sim(query[0], vec), 3), \
                re.sub('\n', ' ', self.chunks_dict[j]))

        return results





