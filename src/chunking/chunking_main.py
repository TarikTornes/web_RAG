from ..utils import web_dataloader
from ..utils.check_device import check_device
from . import text_splitter

import os, yaml, transformers, warnings, pickle

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# tells the logging utility which information to display(verbosity). 
# Here we have set that it only shows us errors
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning)

# options in order to parallelize
os.environ['OMP_NUM_THREADS'] = str(config['Env_Variables']['omp_num_threads'])
os.environ['TOKENIZERS_PARALLELISM'] = str(config['Env_Variables']['tokenizers_parallelism'])


def main():

    check_device()


    web_df = web_dataloader.load_data(config['Paths']['websites_root'])

    chunks_all, chunks_dict = text_splitter.chunk_data(web_df, config)

    data_to_save = {
        "chunks_all": chunks_all,
        "chunks_dict": chunks_dict
    }
    print("Current working directory:", os.getcwd())
    with open("data/chunks.pkl", 'wb') as f:
        pickle.dump(data_to_save,f)

    return data_to_save


if __name__ == "__main__":
    main()






    

    
