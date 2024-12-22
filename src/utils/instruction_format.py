import pickle
from .load_conf import load_conf
from ..query.indexdb import IndexDB


def instruction_format(query_ls):
    instr = ""
    for _, chunk, url_link in query_ls:

        url_link = replace_text_before(url_link, "www", "https://")
        
        instr = instr + f"""FROM WEBPAGE: {url_link}\n{chunk}\n\n"""

    return instr


import re

def replace_text_before(target_string, marker, replacement):
    """
    Replace the text before a certain string (marker) with another string (replacement).

    :param target_string: The full string to modify.
    :param marker: The string that marks the point after which content remains unchanged.
    :param replacement: The string to replace the text before the marker.
    :return: Modified string with the replacement applied.
    """
    pattern = rf"^(.*?){re.escape(marker)}(.*?)(/index\.html)?$"
    result = re.sub(pattern, replacement + r"\2", target_string)
    return result



        
"""
def main():

    config = load_conf()

    with open("data/chunks.pkl", "rb") as f:
        data2 = pickle.load(f)


    with open("data/embeddings.pkl", "rb") as f:
        data1 = pickle.load(f)


    
    db = IndexDB(config["Paths"]["embeddings_path"], 
                 data1["embeddings"], 
                 data2["chunks_dict"], 
                 data2["web_page_dict"])

    query_res = db.get_k_Results("Martin Theobald", config["General"]["k-nearest"])


    print(instruction_format(query_res))


if __name__=="__main__":
    main()
"""
