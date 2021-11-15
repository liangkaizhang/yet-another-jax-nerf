import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './src')

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from dataset import DatasetConfig, DatasetBuilder

ds_config =  DatasetConfig(model_dir='./dataset/pinecone/sparse/0/',
                           images_dir='./dataset/pinecone/images/',
                           sample_per_image=100,
                           batch_size=256)
dataset = DatasetBuilder(ds_config)
next(dataset._parse_dataset())