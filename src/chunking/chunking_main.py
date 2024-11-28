import os
from ..utils import web_dataloader
import yaml

from numpy import dot
from numpy.linalg import norm
import pandas as pd
import pickle
import csv, json, math, os, random, re, sys, time, torch, traceback, transformers, warnings
from datetime import datetime
import torch

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

    websites_root = config['Paths']['websites_root']

    web_dl = web_dataloader.WebPDataLoader(websites_root)
    web_dl.load()
    web_dl.preprocess_df()
    web_data_df = web_dl.get_df()


if __name__ == "__main__":
    main()






    

    
