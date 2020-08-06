import tensorflow as tf
from .types_ import *
from .decorder import make_ResNet50v2_Decoder
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers

def makeLayers(layer_spec: dict) -> Layer:
    if layer_spec['name'] == 'ResNet50v2':
        layer_spec.pop('name')
        return tfk.applications.ResNet50V2(**layer_spec)
    elif layer_spec['name'] == 'ResNet50v2_Decoder':
        layer_spec.pop('name')
        return make_ResNet50v2_Decoder
    elif layer_spec['name'] == 'Conv2D':
        layer_spec.pop('name')
        return tfkl.Conv2D(**layer_spec)
    elif layer_spec['name'] == 'Conv2DTranspose':
        layer_spec.pop('name')
        return tfkl.Conv2DTranspose(**layer_spec)
    elif layer_spec['name'] == 'BN':
        return tfkl.BatchNormalization()
    elif layer_spec['name'] == 'LeakyReLu':
        return tfkl.LeakyReLU()
    elif layer_spec['name'] == 'Flatten':
        return tfkl.Flatten()
    elif layer_spec['name'] == 'tanh':
        return tfk.activations.tanh
    elif layer_spec['name'] == 'Dense':
        return tfkl.Dense(layer_spec['units'])
    elif layer_spec['name'] == 'Reshape':
        layer_spec.pop('name')
        return tfkl.Reshape(**layer_spec)
    elif layer_spec['name'] == 'UpSampling2D':
        layer_spec.pop('name')
        return tfkl.UpSampling2D(**layer_spec)
    else :
        print('there is no layer : {}'.format(layer_spec['name']) )
        return False

        
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5*((sample-mean)**2.*tf.exp(-logvar)+logvar+log2pi), axis=raxis
    )