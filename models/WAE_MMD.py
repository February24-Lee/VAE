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
                **kwargs) -> None:
        super(WAE_MMD, self).__init__()

        self.latent_dim = latent_dim
        self.regluar_weight = regular_weight
        self.model_name = 'WAE_MMD'
        self.kernel_type = kernel_type
        self.kernel_var = kernel_var

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
        if apply_sigmoid :
            return tf.nn.sigmoid(self.decoder(z))
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


    def compute_loss(self, x:Tensor) -> dict:
        z_data = self.encode(x)
        z_prior = self.gen_random(shape=tf.shape(z_data))
        x_recons = self.decode(z_data, apply_sigmoid=True)

        # reconstruct loss [B x 1]
        recon_loss = tf.reduce_mean(tfk.losses.mean_squared_error(x, x_recons), axis=[1,2])

        # --- MMD_loss
        # Kernel(z_data,z_data) [B x 1]
        mmd_loss_z_data = tf.reduce_mean(self.RBFKernel(z_data, z_data), axis=[-1])



        return {'total_loss' : tf.reduce_mean(recon_loss + self.regluar_weight * tfk.losses.binary_crossentropy(tf.ones_like(y_data), y_data)),
                'disc_loss' : disc_loss,
                'recons_loss' : tf.reduce_mean(recon_loss)}

    @tf.function
    def train_step(self, x, opt=tfk.optimizers.Adam()) -> Tensor:
        with tf.GradientTape(persistent=True) as tape:
            loss_dic = self.compute_loss(x)
            disc_loss = self.regluar_weight * loss_dic['disc_loss']
            total_loss = loss_dic['total_loss']
        disc_grad = tape.gradient(disc_loss, self.latent_disc.trainable_variables)
        opt.apply_gradients(zip(disc_grad, self.latent_disc.trainable_variables))

        layer_list = [*self.encoder.trainable_variables, *self.decoder.trainable_variables]
        total_grad = tape.gradient(total_loss, layer_list)
        opt.apply_gradients(zip(total_grad, layer_list))
        del tape

    @tf.function
    def sample(self, sample_num: int=100):
        eps = self.gen_random(shape=[sample_num, self.latent_dim])
        return self.decode(eps, apply_sigmoid=True)