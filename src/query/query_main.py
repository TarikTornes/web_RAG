from ..utils.logging import log
from ..utils.load_conf import load_conf
from ..utils.check_device import check_device
from .embedding_db import EmbeddingDB

import pickle, os, faiss


def main():

    check_device()

    config = load_conf()

    with open("data/embeddings.pkl", "rb") as f:
        data1 = pickle.load(f)

    with open("data/chunks.pkl", "rb") as f:
        data2 = pickle.load(f)



    db = EmbeddingDB(config["Paths"]["embeddings_path"], data1["embeddings"], data2["chunks_dict"])









if __name__=="__main__":
    main()
