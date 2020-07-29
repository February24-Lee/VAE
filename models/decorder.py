import tensorflow as tf
from .types_ import *
tfk = tf.keras
tfkl = tfk.layers

def make_ResNet50v2_Decoder(input_x:Tensor = None):
    '''
    Model
    '''

    # ---- input 4 x 4 x 2048
    f_list = [[1, 512], [3, 512], [1, 2048]]
    x = Residual_Block_A(input_x=input_x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    f_list = [[1, 512], [3, 512], [1, 1024]]
    x = Residual_Block_A(input_x=x, input_channel_same=False, filter_kernel_list=f_list)
    # ---- output 4 x 4 x 1024

    # --- 4 x 4 x 1024
    f_list = [[1, 256], [3, 256], [1, 1024]]
    x = Residual_Block_B(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    # --- 8 x 8 x 1024
    f_list = [[1, 256], [3, 256], [1, 512]]
    x = Residual_Block_A(input_x=x, input_channel_same=False, filter_kernel_list=f_list)
    # --- 8 x 8 x 512

    # --- 8 x 8 x 512
    f_list = [[1, 128], [3, 128], [1, 512]]
    x = Residual_Block_B(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    # --- 16 x 16 x 512
    f_list = [[1, 128], [3, 128], [1, 256]]
    x = Residual_Block_A(input_x=x, input_channel_same=False, filter_kernel_list=f_list)
    

    # --- 16 x 16 x 256
    f_list = [[1, 64], [3, 64], [1, 256]]
    x = Residual_Block_B(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    x = Residual_Block_A(input_x=x, input_channel_same=True, filter_kernel_list=f_list)
    # --- 32 x 32 x 256
    return x

def Residual_Block_A(input_x = None,
                    input_channel_same:bool = True,
                    filter_kernel_list:list = [[1, 64], [3, 64], [1,256]]):
    # pre
    pre_x = tfkl.BatchNormalization()(input_x)
    pre_x = tfkl.ReLU()(pre_x)

    for index, l in enumerate(filter_kernel_list):
        if index == 0:
            x = tfkl.Conv2D(filters=l[1], 
                        kernel_size=l[0],
                        strides=1,
                        padding='same')(pre_x)    
        else:
            x = tfkl.Conv2D(filters=l[1], 
                            kernel_size=l[0],
                            strides=1,
                            padding='same')(x)
        if index == len(filter_kernel_list)-1 :
            break
        x = tfkl.BatchNormalization()(x)
        x = tfkl.ReLU()(x)

    if not input_channel_same:
        pre_x = tfkl.Conv2D(filters=filter_kernel_list[-1][1],
                            kernel_size=1,
                            strides=1,
                            padding='same')(pre_x)
    x = tfkl.add([x, pre_x])
    return x
    
def Residual_Block_B(input_x = None,
                    input_channel_same:bool = True,
                    filter_kernel_list:list = [[1, 64], [3, 64], [1,256]]):
    # pre
    pre_x = tfkl.BatchNormalization()(input_x)
    pre_x = tfkl.ReLU()(pre_x)

    for index, l in enumerate(filter_kernel_list):
        if index == 0:
            x = tfkl.Conv2D(filters=l[1], 
                        kernel_size=l[0],
                        strides=1,
                        padding='same')(pre_x)
        elif index == 1:
            x = tfkl.Conv2DTranspose(filters=l[1], 
                        kernel_size=l[0],
                        strides=2,
                        padding='same')(x)
        else:
            x = tfkl.Conv2D(filters=l[1], 
                            kernel_size=l[0],
                            strides=1,
                            padding='same')(x)
        if index == len(filter_kernel_list)-1 :
            break
        x = tfkl.BatchNormalization()(x)
        x = tfkl.ReLU()(x)

    if not input_channel_same:
        pre_x = tfkl.Conv2D(filters=filter_kernel_list[-1][1],
                            kernel_size=1,
                            strides=1,
                            padding='same')(pre_x)
    pre_x = tfkl.UpSampling2D(size=2)(pre_x)
    x = tfkl.add([x, pre_x])
    return x
