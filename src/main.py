from .utils.logging import log
from .utils.load_conf import load_conf
from .utils.check_device import check_device
from .query.indexdb import IndexDB
from .model.llama_model import Llama_model

import pickle, os


def main():

    
    config = load_conf()
    check_device()
    
    with open("data/embeddings.pkl", "rb") as f:
        data1 = pickle.load(f)

    with open("data/chunks.pkl", "rb") as f:
        data2 = pickle.load(f)


    db = IndexDB(config["Paths"]["embeddings_path"], 
                 data1["embeddings"], 
                 data2["chunks_dict"], 
                 data2["web_page_dict"])

    
    model = Llama_model(config["Paths"]["llama_cpp_path"])
    check_device()


    # Q&A Loop
    while True:
        log("INFO", "Started loop")

        print('#'*os.get_terminal_size().columns, "\n")

        print("    Whats your question (\"none\" to abort) ?")

        question = str(input("\n>>>  "))

        if question == "none":
            break
        
        print('\n'+'-'*os.get_terminal_size().columns)


        query_res = db.get_k_Results(question, config["General"]["k-nearest"])
        
        answer = model.getAnswer(query_res, question)

        print("\n")


        
if __name__=="__main__":
    main()





