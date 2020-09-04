import yaml
import argparse
import numpy as np
from pathlib import Path

from models import *
from utils.dataset_CelebA import genDatasetCelebA
from utils.dataset_Satellite import genDatasetSatellite
import tensorflow as tf
import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    default=None)

parser.add_argument('--save_model', '-s',
                    dest="Is_save_model",
                    default='true')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try :
        config = yaml.load(file, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        print(e)

# --- dataset
if config['dataset_param']['datatype'] == 'satellite':
    train_gen, test_gen = genDatasetSatellite(**config['dataset_param'])    
else :
    train_gen, test_gen = genDatasetCelebA(**config['dataset_param'])

# --- model
model = vae_models[config['model_params']['name']](**config['model_params'])

# --- make file 
current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
RESULT_PATH = 'result/' + current_time + '_' + model.model_name + '/'


# --- train
trainer(model, 
        train_gen, 
        test_gen, 
        #tfk.optimizers.Adam(config['opt_param']['LR']),
        tfk.optimizers.Adam(learning_rate=tfk.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config['opt_param']['initial_learning_rate'],
            decay_rate=config['opt_param']['decay_rate'],
            decay_steps=config['opt_param']['decay_steps'],
        )),
        epochs=config['train_param']['epochs'],
        save_path=config['train_param']['save_path'],
        save_iter=config['train_param']['save_iter'],
        scale=config['dataset_param']['scale'],
        batch_size=config['dataset_param']['batch_size'],
        check_point_iter=config['train_param']['check_point_iter'],
        check_point_path=config['train_param']['check_point_path'],
        log_dir=config['train_param']['log_dir'],
        check_loss_cnt=config['train_param']['check_loss_cnt'],
        result_path = RESULT_PATH)

# --- save model
if args.Is_save_model == 'true':
    Path(RESULT_PATH+config['train_param']['save_model_path']).mkdir(parents=True, exist_ok=True)
    path = RESULT_PATH+config['train_param']['save_model_path'] + model.model_name +'.h5' 
    model.save_weights(path)