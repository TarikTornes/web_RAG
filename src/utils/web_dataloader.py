import os
import pandas as pd
import re
from .logging import log


class WebPDataLoader:
    '''
    This class is responsible for loading the webdata into a dataframe
    '''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.df = pd.DataFrame()

    def load(self):
        data = []

        print(self.dir_path)

        for root, dirs, files in os.walk(self.dir_path):  # Adjust the root path if needed
            print(files)
            
            for file in files:
                if file.endswith('.txt'):  # Only process files named 'output.txt'
                    file_path = os.path.join(root, file)
                    url = file_path.replace(".txt", ".html")
                    #url = url.replace("./dataset/", "https://")
                    
                    try:
                        # Try reading the content of the file using 'ISO-8859-1' encoding
                        with open(file_path, 'r', encoding='ISO-8859-1') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        # If it fails, skip the file and print a warning
                        log("WARNING",f"Skipping file (cannot decode): {file_path}")
                        continue
                        
                    # Create a dictionary for the current file
                    file_info = {
                        'url': url,
                        'content': content
                    }
                    
                    # Append the dictionary to the data list
                    data.append(file_info)

        # Convert the list of dictionaries into a pandas DataFrame
        self.df = pd.DataFrame(data)
        print(self.df.head)
        log("INFO", "web data loaded")
        return None


    def preprocess_text(self,text):

        text = re.sub(r'\[.*?\]', '', text)  # Removes anything within square brackets
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with a single space

        # Remove leading/trailing spaces
        text = text.strip()

        # Lowercase the entire text (optional, based on your NLP needs)
        text = text.lower()

        return text


    def preprocess_df(self, preproc_func=None):
        if preproc_func is None:
            preproc_func = self.preprocess_text

        self.df['cleaned_content'] = self.df['content'].apply(lambda text: preproc_func(text))

    def get_df(self):
        return self.df

    def get_path(self):
        return self.dir_path


def load_data(root_dir):
    """ This function perform the necessary steps to retrieve a dataframe
        storing the website content.

    :param root_dir: file_path to the root directory which contains the whole websites
    :return: dataframe containing the website content
    """

    web_dl = WebPDataLoader(root_dir)
    web_dl.load()
    web_dl.preprocess_df()

    return web_dl.get_df()

