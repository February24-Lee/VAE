import tensorflow as tf
from .types_ import *

tfk = tf.keras
tfkl = tf.keras.layers

def makeLayers(layer_spec: dict) -> Layer:
    if layer_spec['name'] == 'Conv2D':
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
    else :
        print('there is no layer : {}'.format(layer_spec['name']) )
        return False