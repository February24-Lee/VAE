import tensorflow as tf
import numpy as np

tfk = tf.keras

def genDatasetSatellite(path : str = None,
                        input_shape : list = (128, 128),
                        test_split : float = 0.2, 
                        batch_size : int = 32,
                        shuffle : bool = True,
                        **kwargs):
    # you should input shpae except channel
    if len(input_shape) == 3 :
        print('you should input shape except channel size,')
        input_shape = input_shape[0:2]

    datagen = tfk.preprocessing.image.ImageDataGenerator(validation_split=test_split)
    train_gen = datagen.flow_from_directory(path,
                                            target_size=input_shape,
                                            class_mode=None,
                                            batch_size=batch_szie,
                                            shuffle=shuffle,
                                            subset='training')
    test_gen = datagen.flow_from_directory(path,
                                            target_size=input_shape,
                                            class_mode=None,
                                            batch_size=batch_szie,
                                            shuffle=shuffle,
                                            subset='validation')
    return train_gen, test_gen