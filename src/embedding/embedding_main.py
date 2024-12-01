from ..utils.check_device import check_device
from ..utils.logging import log
from ..utils.load_conf import load_conf
from .embedding_model import Embedding_Model

import pickle

def main():

    with open("data/chunks.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    # chunks_dict = loaded_data["chunks_dict"]
    chunks_all = loaded_data["chunks_all"]


    check_device()
    config = load_conf()

    embedding_mod = Embedding_Model(config['Paths']['embeddings_path'])
    check_device()
    embedded_chunks = embedding_mod.embed_chunks(chunks_all, 2)

    data_to_save = {
        "embeddings": embedded_chunks,
        "hidden_size": embedding_mod.get_hidden_size()
    }

    with open("data/embeddings.pkl", "wb") as f:
        pickle.dump(data_to_save, f)

    log("INFO", "Embedding: embeddings successfully pickled")



if __name__=="__main__":
    main()
