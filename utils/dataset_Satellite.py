import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.types_ import *

tfk = tf.keras
tfkl = tf.keras.layers

def genDatasetSatellite(path : str = None,
                    input_shape : list = (128, 128),
                    test_split : float = 0.2,
                    batch_szie : int = 32,
                    shuffle : bool = True,
                    color_mode: str = 'gray',
                    is_reverse: str = 'false',
                    **kwargs):
    # you should input shpae except channel
    if len(input_shape) == 3 :
        print('you should input shape except channel size,')
        input_shape = input_shape[0:2]

    if is_reverse == 'true':
        datagen = tfk.preprocessing.image.ImageDataGenerator(validation_split=test_split,
                                                            preprocessing_function=reverse_pixel)
    else :
        datagen = tfk.preprocessing.image.ImageDataGenerator(validation_split=test_split)

    
    train_gen = datagen.flow_from_directory(path,
                                            target_size=input_shape,
                                            color_mode= color_mode,
                                            class_mode=None,
                                            batch_size=batch_szie,
                                            shuffle=shuffle,
                                            subset='training')
    test_gen = datagen.flow_from_directory(path,
                                            target_size=input_shape,
                                            color_mode=color_mode,
                                            class_mode=None,
                                            batch_size=batch_szie,
                                            shuffle=shuffle,
                                            subset='validation')
    return train_gen, test_gen

def reverse_pixel(img):
    '''
    reverse pixcel value:
    0 ~ 255 => 255 ~ 0
    '''
    return -1 * (img-255.)
