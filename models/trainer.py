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
            check_loss_cnt:int = 1,
            result_path : str = None,
            **kwargs):

    train_iter = train_x.n // train_x.batch_size
    test_iter = test_x.n // test_x.batch_size

    # --- for log save
    #current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- TOTAL SAVE PATH
    RESULT_PATH = result_path

    train_log_dir = RESULT_PATH + log_dir + 'train'
    test_log_dir = RESULT_PATH + log_dir + 'test'
    img_log_dir = RESULT_PATH + log_dir + 'img'


    # --- original version
    #train_log_dir = log_dir + current_time + '_' + model.model_name + '/train'
    #test_log_dir = log_dir + current_time + '_' + model.model_name + '/test'
    #img_log_dir = log_dir + current_time + '_' + model.model_name + '/img'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    #img_summary_writer = tf.summary.create_file_writer(img_log_dir)

    # --- make logs for loss functions
    loss_list = [tfk.metrics.Mean() for _ in range(check_loss_cnt)]
    
    total_iter = 0
    for epoch in range(1, epochs+1):
        # --- Train
        print('Epoch : {} training..'.format(epoch))
        train_x.reset()
        for index, x in enumerate(tqdm(train_x)):
            total_iter += 1
            # Dump last batch because batch size is diffenent
            if index > train_iter or len(x) != batch_size:
                break
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.
            train_loss = model.train_step(x, opt=opt)

            # 2020. 10. 17 Update
            if train_loss is not None:
                with train_summary_writer.as_default():
                    for key, val in train_loss.items():
                        tf.summary.scalar(key, val, step=total_iter)
                    tf.summary.scalar('learning_rate', opt._decayed_lr(tf.float32).numpy(), step=total_iter)
                    tf.summary.scalar('epoch', epoch, step=total_iter)



        train_x.reset()
        #  --- Calculate Trainset Loss
        for index, x in enumerate(train_x):
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.
            # Dump last batch because batch size is diffenent
            if index > train_iter or len(x) != batch_size:
                break
            loss_dic = model.compute_loss(x, is_training=True)
            for i, (_, value) in enumerate(loss_dic.items()):
                loss_list[i](value)
        loss = loss_list[0].result()
        print('Epoch: {}, train set loss: {}'.format(epoch, loss))
        #with train_summary_writer.as_default():
        #    for index, loss_name in enumerate(loss_dic):
        #        tf.summary.scalar(loss_name, loss_list[index].result(), step=epoch)
        #        loss_list[index].reset_states()

        test_x.reset()
        # --- Calculate Testset Loss
        for index, x in enumerate(test_x):
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.

            # Dump last batch because batch size is diffenent
            if index > test_iter or len(x) != batch_size:
                break
            loss_dic = model.compute_loss(x, is_training=False)
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
            #with img_summary_writer.as_default():
            #    tf.summary.image('Reconstruct IMG', reconstruct_x, step=epoch, max_outputs=len(reconstruct_x))

            # --- for gray_scale case
            if tf.shape(reconstruct_x)[-1] ==1:
                reconstruct_x = tf.reshape(reconstruct_x, tf.shape(reconstruct_x)[:-1])
                color_type = 'gray'
            
            Path(RESULT_PATH + save_path).mkdir(parents=True, exist_ok=True)
            path = RESULT_PATH + save_path + 'epoch_' + str(epoch) + '.png'
            save_images(model, img_num=min(batch_size, 64), x=reconstruct_x, path=path, scale=scale, color_type=color_type)

        # --- check point sace
        if epoch % check_point_iter == 0 :
            path = RESULT_PATH + check_point_path + 'checkpoint_{}'.format(epoch)
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
        else :
            plt.subplot(8,8,i+1)

        if color_type is 'gray':
            plt.imshow(x[i], cmap='gray')
        else :
            plt.imshow(x[i])
        plt.axis('off')
    plt.savefig(path)
    return



