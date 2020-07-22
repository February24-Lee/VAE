import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .types_ import *

tfk = tf.keras
tfkl = tf.keras.layers

def trainer(model,
            train_x: DirectoryIterator = None,
            test_x: DirectoryIterator =None,
            opt=tfk.optimizers.Adam(1e-4),
            epochs=10,
            save_path: str=None,
            scale='sigmoid',
            batch_size:int =32):

    train_iter = train_x.n // batch_size
    test_iter = test_x.n // batch_size

    for epoch in range(1, epochs+1):
        start_t = time.time()
        print('Epoch : {} training..'.format(epoch))
        for index, x in enumerate(tqdm(train_x)):
            if index > train_iter:
                break
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.
            model.train_step(x, opt=opt)
        end_time = time.time()

        print('Calculating testset...')
        loss = tfk.metrics.Mean()
        for index, x in enumerate(test_x):
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.

            if index > test_iter:
                break
            loss(model.compute_loss(x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_t))

        if len(x) is not batch_size:
            x = next(test_x)

        path = save_path + model.model_name + '_epoch_' + str(epoch) + '.png'
        save_images(model, img_num=batch_size, x=x, path=path, scale=scale)
    return 

def save_images(model,
                img_num=32,
                x: List[np.array]=None,
                path: str=None,
                scale: str="sigmoid"):
    plt.figure(figsize=(10,10))
    if scale == 'tanh':
        x = (x+1.)/2. 
    for i in range(img_num):
        plt.subplot(8,4,i+1)
        plt.imshow(x[i])
        plt.axis('off')
    plt.savefig(path)
    return



