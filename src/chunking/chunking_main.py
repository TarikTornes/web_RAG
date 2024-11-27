import os
from ..utils import web_dataloader

from numpy import dot
from numpy.linalg import norm
import pandas as pd
import pickle
import csv, json, math, os, random, re, sys, time, torch, traceback, transformers, warnings
from datetime import datetime
import torch


def set_chunking_env():
    # tells the logging utility which information to display(verbosity). 
    # Here we have set that it only shows us errors
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings('ignore', category=UserWarning)

    # options in order to parallelize
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():

    set_chunking_env()

    web_dl = web_dataloader.WebPDataLoader()

    df = web_dl.load() # NOt finished (input file path)





    

    
