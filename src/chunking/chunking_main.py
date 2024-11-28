from ..utils import web_dataloader, check_device
import chunker

import os, yaml, transformers, warnings, pickle

with open('config.yaml', 'r') as file:
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

    websites_root = config['Paths']['websites_root']

    web_df = web_dataloader.load_data(websites_root)

    chunks_all, chunks_dict = chunker.chunk_data(web_df)

    data_to_save = {
        "chunks_all": chunks_all,
        "chunks_dict": chunks_dict
    }

    with open("../data/chunks.pkl") as f:
        pickle.dump(data_to_save,f)

    return data_to_save


if __name__ == "__main__":
    main()






    

    
