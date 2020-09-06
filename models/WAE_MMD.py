import tensorflow as tf
import numpy as np

from .BaseVAE import BaseVAE
from .utils import makeLayers
from .types_ import *

tfk = tf.keras
tfkl = tfk.layers

class WAE_MMD(BaseVAE):
    def __init__(self,
                latent_dim : int = None,
                input_shape : list = None,
                encoder_layers: list = None,
                decoder_layers: list = None,
                regular_weight: int = None,
                kernel_type: str = 'RBF',
                kernel_var: float = 2.0,
                kernel_capacity: float = 0.0,
                loss_function_type: str = 'MSE',
                **kwargs) -> None:
        super(WAE_MMD, self).__init__()

        self.latent_dim = latent_dim
        self.regluar_weight = regular_weight
        self.model_name = 'WAE_MMD'
        self.kernel_type = kernel_type
        self.kernel_var = kernel_var
        self.kernel_capacity = kernel_capacity
        self.loss_function_type = loss_function_type

        # --- prior dist [p(z)]
        self.gen_random = tf.random_normal_initializer()

        # --- Encoder
        encoder_input = tfk.Input(shape = input_shape)
        for index, layer_spec in enumerate(encoder_layers):
            if index == 0 :
                x = makeLayers(layer_spec = layer_spec)(encoder_input)
            else :
                x = makeLayers(layer_spec = layer_spec)(x)
        self.encoder = tfk.Model(inputs=encoder_input, outputs=x)

        # --- Decoder
        decoder_input = tfk.Input(shape=(latent_dim,))
        for index, layer_spec in enumerate(decoder_layers):
            if index ==0 :
                x = makeLayers(layer_spec=layer_spec)(decoder_input)
            else:
                x = makeLayers(layer_spec=layer_spec)(x)
        self.decoder = tfk.Model(inputs=decoder_input, outputs=x)


    def encode(self, x:Tensor)->Tensor:
        return self.encoder(x)


    def decode(self, z:Tensor, apply_sigmoid=False) -> Tensor:
        if self.loss_function_type == 'BCE':
            if apply_sigmoid :
                return tf.nn.sigmoid(self.decoder(z))
            return self.decoder(z)
        elif self.loss_function_type == 'MSE':
            return tf.nn.tanh(self.decoder(z))
        else:
            return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x), apply_sigmoid=True)


    def RBFKernel(self, x1:Tensor=None, x2:Tensor=None) -> Tensor:
        '''
        x1 and x2 shape should be same [ B x D ]
        x1 change -> [ B x 1 x D ]
        x2 change -> [ B x D x 1 ]
        '''
        B_size = x1.shape[0]
        D_size = x1.shape[1]

        x1 = tf.reshape(x1, (B_size, 1, D_size))
        x2 = tf.reshape(x2, (B_size, D_size, 1)) 

        # x1-x2's shape [ B x D x D ] 
        dist = tf.reduce_sum(tf.math.pow(x1-x2, 2), axis=2) # B x D
        
        return tf.math.exp(-dist/(2.0 * self.kernel_var * self.latent_dim))


    def compute_loss(self, x:Tensor, **kwargs) -> dict:
        z_data = self.encode(x)
        z_prior = self.gen_random(shape=tf.shape(z_data))
        x_recons = self.decode(z_data, apply_sigmoid=False)

        # number of shaple
        n = z_prior.shape[0]

        # reconstruct loss [B x 1]
        if self.loss_function_type == 'MSE':
            # MSE loss
            recon_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x, x_recons), axis=[1,2])
        elif self.loss_function_type == 'BCE':
            # BCE loss
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recons, labels=x)
            recon_loss = tf.reduce_mean(cross_ent, axis=[1,2,3])

        # --- MMD_loss
        # Kernel(z_data,z_data) [B x 1]
        mmd_loss_z_data = tf.reduce_sum(self.RBFKernel(z_data, z_data), axis=[-1]) / (n*(n-1))
        mmd_loss_z_prior = tf.reduce_sum(self.RBFKernel(z_prior, z_prior), axis=[-1]) / (n*(n-1))
        mmd_loss_z_data_prior = tf.reduce_sum(self.RBFKernel(z_prior, z_data), axis=[-1]) / (n*n)

        mmd_loss = mmd_loss_z_data + mmd_loss_z_prior - 2 * mmd_loss_z_data_prior

        return {'total_loss' : tf.reduce_mean(recon_loss + self.regluar_weight * tf.abs(mmd_loss-self.kernel_capacity)),
                'mmd_loss' : tf.reduce_mean(mmd_loss),
                'recons_loss' : tf.reduce_mean(recon_loss)}

    @tf.function
    def train_step(self, x, opt=tfk.optimizers.Adam()) -> Tensor:
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)['total_loss']
        grad = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(grad, self.trainable_variables))
        return 
        

    @tf.function
    def sample(self, sample_num: int=100):
        eps = self.gen_random(shape=[sample_num, self.latent_dim])
        return self.decode(eps, apply_sigmoid=True)