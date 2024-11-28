import os
import pandas as pd


class WebPDataLoader:
    '''
    This class is responsible for loading the webdata into a dataframe
    '''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.df = pd.DataFrame()

    def load(self):
        data = []
        for root, dirs, files in os.walk(self.dir_path):  # Adjust the root path if needed
            for file in files:
                if file.endswith('.txt'):  # Only process files named 'output.txt'
                    file_path = os.path.join(root, file)
                    url = file_path.replace(".txt", ".html")
                    #url = url.replace("./dataset/", "https://")
                    
                    try:
                        # Try reading the content of the file using 'ISO-8859-1' encoding
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        # If it fails, skip the file and print a warning
                        print(f"Skipping file (cannot decode): {file_path}")
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


    def preprocess_text(self):

            # Remove leading/trailing spaces
            text = text.strip()

            # Lowercase the entire text (optional, based on your NLP needs)
            text = text.lower()

            return text


    def preprocess_df(self, preproc_func=preprocess_text):
        self.df['cleaned_content'] = self.df['content'].apply(preproc_func)

    def get_df(self):
        return self.df

    def get_path(self):
        return self.dir_path


