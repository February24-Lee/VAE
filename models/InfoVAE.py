import tensorflow as tf
import numpy as np

from .BaseVAE import BaseVAE
from .utils import makeLayers
from .types_ import *

tfk = tf.keras
tfkl = tfk.layers

class INFOVAE(BaseVAE):
    def __init__(self,
                latent_dim : int = None,
                input_shape : list = None,
                encoder_layers: list = None,
                decoder_layers: list = None,
                regular_weight: float = None,
                recons_weight: float = None,
                kernel_type: str = 'RBF',
                kernel_var: float = 2.0,
                alpha: float = 0.5,
                **kwargs) -> None:

        super(INFOVAE, self).__init__()

        assert alpha <= 1.0

        self.model_name = 'INFOVAE'
        self.latent_dim = latent_dim
        self.regluar_weight = regular_weight
        self.recons_weight = recons_weight
        self.kernel_type = kernel_type
        self.kernel_var = kernel_var
        self.alpha = alpha

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
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar


    def decode(self, z:Tensor, apply_sigmoid=False) -> Tensor:
        if apply_sigmoid :
            return tf.nn.sigmoid(self.decoder(z))
        return self.decoder(z)


    def forward(self, x: Tensor) -> Tensor:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, apply_sigmoid=True)


    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        eps = tf.random.normal(shape=mean.shape)
        return mean + eps*tf.exp(logvar * 0.5)


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
        mean, logvar = self.encode(x)
        z_data = self.reparameterize(mean, logvar)
        z_prior = self.gen_random(shape=tf.shape(z_data))
        x_recons = self.decode(z_data, apply_sigmoid=True)

        # reconstruct loss [B x 1]
        recon_loss = tf.reduce_mean(tfk.losses.mean_squared_error(x, x_recons), axis=[1,2])

        # KL loss [B x 1]
        kl_loss = -0.5 * tf.reduce_sum((1 + logvar - mean**2 - tf.math.exp(logvar)), axis=1)

        # --- MMD_loss
        # Kernel(z_data,z_data) [B x 1]
        mmd_loss_z_data = tf.reduce_mean(self.RBFKernel(z_data, z_data), axis=[-1]) 
        mmd_loss_z_prior = tf.reduce_mean(self.RBFKernel(z_prior, z_prior), axis=[-1])
        mmd_loss_z_data_prior = tf.reduce_mean(self.RBFKernel(z_prior, z_data), axis=[-1]) 

        mmd_loss = mmd_loss_z_data + mmd_loss_z_prior - 2 * mmd_loss_z_data_prior

        return {'total_loss' : tf.reduce_mean(self.recons_weight * recon_loss + 
                                            (1-self.alpha) * kl_loss +
                                            (self.alpha + self.regluar_weight  -1.) * mmd_loss),
                'kl_loss': tf.reduce_mean(kl_loss),
                'mmd_loss': tf.reduce_mean(mmd_loss),
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