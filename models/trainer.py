import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from tqdm import tqdm
from .types_ import *
from pathlib import Path

tfk = tf.keras
tfkl = tf.keras.layers

def trainer(model,
            train_x: DirectoryIterator = None,
            test_x: DirectoryIterator =None,
            opt=tfk.optimizers.Adam(1e-4),
            epochs=10,
            save_path: str=None,
            save_iter: int=10,
            scale='sigmoid',
            batch_size:int =32,
            check_point_path:str = 'checkpoint/',
            check_point_iter:int = 5,
            log_dir:str = 'logs/',
            check_loss_cnt:int = 1):

    train_iter = train_x.n // train_x.batch_size
    test_iter = test_x.n // test_x.batch_size

    # --- for log save
    current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir + current_time + '_' + model.model_name + '/train'
    test_log_dir = log_dir + current_time + '_' + model.model_name + '/test'
    img_log_dir = log_dir + current_time + '_' + model.model_name + '/img'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    img_summary_writer = tf.summary.create_file_writer(img_log_dir)

    # --- make logs for loss functions
    loss_list = [tfk.metrics.Mean() for _ in range(check_loss_cnt)]

    for epoch in range(1, epochs+1):
        # --- Train
        print('Epoch : {} training..'.format(epoch))
        train_x.reset()
        for index, x in enumerate(tqdm(train_x)):
            if index > train_iter:
                break
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.
            model.train_step(x, opt=opt)

        train_x.reset()
        #  --- Calculate Trainset Loss
        for index, x in enumerate(train_x):
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.

            if index > train_iter:
                break
            loss_dic = model.compute_loss(x)
            for i, (_, value) in enumerate(loss_dic.items()):
                loss_list[i](value)
        loss = loss_list[0].result()
        print('Epoch: {}, train set loss: {}'.format(epoch, loss))
        with train_summary_writer.as_default():
            for index, loss_name in enumerate(loss_dic):
                tf.summary.scalar(loss_name, loss_list[index].result(), step=epoch)
                loss_list[index].reset_states()

        test_x.reset()
        # --- Calculate Testset Loss
        for index, x in enumerate(test_x):
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.

            if index > test_iter:
                break
            loss_dic = model.compute_loss(x)
            for i, (_, value) in enumerate(loss_dic.items()):
                loss_list[i](value)
        loss = loss_list[0].result()
        print('Epoch: {}, Test set loss: {}'.format(epoch, loss))
        FINAL_LOSS = loss_list[0].result()
        with test_summary_writer.as_default():
            for index, loss_name in enumerate(loss_dic):
                tf.summary.scalar(loss_name, loss_list[index].result(), step=epoch)
                loss_list[index].reset_states()

        test_x.reset()
        # --- save image
        if epoch % save_iter == 0 :
            if len(x) is not batch_size:
                x = next(test_x)
            reconstruct_x = model.forward(x)
            color_type = 'rgb' # default
            
            
            # --- TENSORBOARD
            with img_summary_writer.as_default():
                tf.summary.image('Reconstruct IMG', reconstruct_x, step=epoch, max_outputs=len(reconstruct_x))

            # --- for gray_scale case
            if tf.shape(reconstruct_x)[-1] ==1:
                reconstruct_x = tf.reshape(reconstruct_x, tf.shape(reconstruct_x)[:-1])
                color_type = 'gray'
            
                
            Path(save_path).mkdir(parents=True, exist_ok=True)
            path = save_path + model.model_name + '_epoch_' + str(epoch) + '.png'
            save_images(model, img_num=batch_size, x=reconstruct_x, path=path, scale=scale, color_type=color_type)


        # --- check point sace
        if epoch % check_point_iter == 0 :
            path = check_point_path + model.model_name +'_checkpoint_{}'.format(epoch)
            model.save_weights(path)

    return FINAL_LOSS

def save_images(model,
                img_num=32,
                x: List[np.array]=None,
                path: str=None,
                scale: str="sigmoid",
                color_type: str = 'rgb'):
    plt.figure(figsize=(15,15))
    if scale == 'tanh':
        x = (x+1.)/2. 
    for i in range(img_num):
        if img_num == 32 :
            plt.subplot(8,4,i+1)
        elif img_num == 64 :
            plt.subplot(8,8,i+1)

        if color_type is 'gray':
            plt.imshow(x[i], cmap='gray')
        else :
            plt.imshow(x[i])
        plt.axis('off')
    plt.savefig(path)
    return



