import yaml
import argparse
import numpy as np
from pathlib import Path

from models import *
from utils.dataset_CelebA import genDatasetCelebA
from utils.dataset_Satellite import genDatasetSatellite

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# --- parser
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

# --- HP for Tensorboard 
hp_log_dir = 'logs/hp/'
hp_summary_writer = tf.summary.create_file_writer(hp_log_dir)

HP_LOSS_FUNCTION = hp.HParam('loss_function_type', hp.Discrete(['MSE', 'BCE']))
HP_LR = hp.HParam('LR', hp.Discrete([0.001, 0.0005, 0.0001]))
HP_NETWORK_TYPE = hp.HParam('network_type', hp.Discrete(['deep', 'shallow']))
HP_LATENT_DIM = hp.HParam('latent_dim', hp.Discrete([32, 64, 128]))

with hp_summary_writer.as_default():
    hp.hparams_config(
        hparams = [HP_LATENT_DIM, HP_LOSS_FUNCTION, HP_LR, HP_NETWORK_TYPE],
        metrics = [hp.Metric('loss', display_name='loss')]
    )

origin_config = config

sess_num = 0
for _loss_type in HP_LOSS_FUNCTION.domain.values:
    for _lr in HP_LR.domain.values:
        for _network in HP_NETWORK_TYPE.domain.values:
            for _latent in HP_LATENT_DIM.domain.values:
                hparams = {
                    HP_LOSS_FUNCTION: _loss_type,
                    HP_LR: _lr,
                    HP_NETWORK_TYPE: _network,
                    HP_LATENT_DIM: _latent
                }
                run_name = 'run-%d' % sess_num
                modified_config = origin_config

                modified_config['train_param']['save_path'] = modified_config['train_param']['save_path'] + run_name + '/'
                modified_config['train_param']['save_model_path'] = modified_config['train_param']['save_model_path'] + run_name + '/'
                modified_config['train_param']['check_point_path'] = modified_config['train_param']['check_point_path'] + run_name + '/'
                modified_config['train_param']['log_dir'] = modified_config['train_param']['log_dir'] + run_name + '/'

                
                # ---
                modified_config['opt_param']['LR'] = _lr
                # ---
                if _network == 'deep':
                    modified_config['model_params'] = modified_config['model_params_deep']
                elif _network == 'shallow':
                    modified_config['model_params'] = modified_config['model_params_shallow']
                # ---
                if _loss_type == 'MSE':
                    modified_config['model_params']['loss_function_type'] = 'MSE'
                    modified_config['dataset_param']['scale'] = 'tanh'
                elif _loss_type == 'BCE':
                    modified_config['model_params']['loss_function_type'] = 'BCE'
                    modified_config['dataset_param']['scale'] = 'sigmoid'
                # ---
                elif _latent == 32:
                    modified_config['model_params'] = modified_config['model_params']['_32']
                elif _latent == 64:
                    modified_config['model_params'] = modified_config['model_params']['_64']
                elif _latent == 128:
                    modified_config['model_params'] = modified_config['model_params']['_128']

                with tf.summary.create_file_writer(hp_log_dir+run_name):
                    hp.hparams(hparams)
                    loss = run(modified_config)
                    tf.summary.scalar('loss', loss, step=modified_config['train_param']['epochs'])

                sess_num += 1

def run(config):
    # --- dataset
    if config['dataset_param']['datatype'] == 'satellite':
        train_gen, test_gen = genDatasetSatellite(**config['dataset_param'])    
    else :
        train_gen, test_gen = genDatasetCelebA(**config['dataset_param'])

    # --- model
    model = vae_models[config['model_params']['name']](**config['model_params'])

    # --- train
    loss = trainer(model,
                train_gen, 
                test_gen, 
                tfk.optimizers.Adam(config['opt_param']['LR']),
                epochs=config['train_param']['epochs'],
                save_path=config['train_param']['save_path'],
                save_iter=config['train_param']['save_iter'],
                scale=config['dataset_param']['scale'],
                batch_size=config['dataset_param']['batch_size'],
                check_point_iter=config['train_param']['check_point_iter'],
                check_point_path=config['train_param']['check_point_path'],
                log_dir=config['train_param']['log_dir'],
                check_loss_cnt=config['train_param']['check_loss_cnt'])

    # --- save model
    if args.Is_save_model == 'true':
        Path(config['train_param']['save_model_path']).mkdir(parents=True, exist_ok=True)
        path = config['train_param']['save_model_path'] + model.model_name +'.h5' 
        model.save_weights(path)

    return loss