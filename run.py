import yaml
import argparse
import numpy as np

from models import *
from utils.dataset_CelebA import genDatasetCelebA_test
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    default=None)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try :
        config = yaml.load(file, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        print(e)

# --- dataset
train_gen, test_gen = genDatasetCelebA(**config['dataset_param'])

# --- model
model = vae_models[config['model_params']['name']](**config['model_params'])

# --- train
trainer(model, 
        train_gen, 
        test_gen, 
        tfk.optimizers.Adam(config['opt_param']['LR']),
        epochs=config['train_param']['epochs'],
        save_path=config['train_param']['save_path'])
