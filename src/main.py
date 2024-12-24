from .utils.logging import log
from .utils.load_conf import load_conf
from .utils.check_device import check_device
from .query.indexdb import IndexDB
from .model.llama_model import Llama_model

import pickle


def main():

    check_device()

    config = load_conf()

    with open("data/embeddings.pkl", "rb") as f:
        data1 = pickle.load(f)

    with open("data/chunks.pkl", "rb") as f:
        data2 = pickle.load(f)


    db = IndexDB(config["Paths"]["embeddings_path"], 
                 data1["embeddings"], 
                 data2["chunks_dict"], 
                 data2["web_page_dict"])


    model = Llama_model(config["Paths"]["llama_cpp_path"])


    # Q&A Loop
    while True:
        log("INFO", "Started loop")

        print("Whats your question (\"none\" to abort) ?")

        question = str(input())

        if question == "none":
            break

        query_res = db.get_k_Results(question, config["General"]["k-nearest"])
        
        answer = model.getAnswer(query_res, question)

        print(answer)


if __name__=="__main__":
    main()





