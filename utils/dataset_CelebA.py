import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tfk = tf.keras
tfkl = tf.keras.layers

def genDatasetCelebA(path : str = None,
                    input_shape : list = (128, 128),
                    test_split : float = 0.2,
                    batch_szie : int = 32,
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

def genDatasetCelebA_test(**kwargs):
    train_gen, test_gen = genDatasetCelebA(**kwargs)
    train_ds = next(train_gen)
    test_ds = next(test_gen)
    if kwargs['is_show'] :
        plt.figure()
        for i in range(6):
            plt.subplot(2,6,i+1)
            plt.axis('off')
            plt.imshow(train_ds[i]/255.)
            plt.title('train_{}'.format(i))

            plt.subplot(2,6,i+7)
            plt.axis('off')
            plt.imshow(test_ds[i]/255.)
            plt.title('test_{}'.format(i))
        plt.show()


