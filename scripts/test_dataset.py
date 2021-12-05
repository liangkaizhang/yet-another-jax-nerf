import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')

from dataset import DatasetConfig, DatasetBuilder
import tensorflow as tf

if __name__ == "__main__":
    ds_config =  DatasetConfig(model_dir='../dataset/pinecone/sparse/0/',
                            images_dir='../dataset/pinecone/images/',
                            batch_from_single_image=True,
                            is_training=True,
                            batch_size=8)

    dataset = DatasetBuilder(ds_config)
    ds = dataset.build()