import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from .types_ import *

tfk = tf.keras
tfkl = tf.keras.layers

def trainer(model,
            train_x: DirectoryIterator = None,
            test_x: DirectoryIterator =None,
            opt=tfk.optimizers.Adam(1e-4),
            epochs=10,
            save_path: str=None):
    for epoch in range(1, epochs+1):
        start_t = time.time()
        for x in train_x:
            model.train_step(x, opt=opt)
        end_time = time.time()

        loss = tfk.metrics.Mean()
        for x in test_x:
            loss(model.compute_loss(x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_t))
        path = save_images + model.model_name + '_epoch_' + str(epoch) + '.png'
        save_images(model, img_num=32, x=x, path=path)
    return 

def save_images(model,
                img_num=32,
                x: List[np.array]=None,
                path: str=None):

    plt.figure(figsize=(10,10))
    for i in range(img_num):
        plt.subplot(8,4,i+1)
        plt.imshow(x[i])
    plt.savefig(path)
    return



