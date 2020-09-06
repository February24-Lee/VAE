import tensorflow as tf
import numpy as np

from .BaseVAE import BaseVAE
from .utils import makeLayers
from .types_ import *

tfk = tf.keras
tfkl = tfk.layers

class WAE_GAN(BaseVAE):
    def __init__(self,
                latent_dim : int = None,
                input_shape : list = None,
                encoder_layers: list = None,
                decoder_layers: list = None,
                latent_discriminator_layers: list = None,
                regular_weight: int = None,
                loss_function_type: str = 'MSE',
                **kwargs) -> None:
        super(WAE_GAN, self).__init__()

        self.latent_dim = latent_dim
        self.regluar_weight = regular_weight
        self.model_name = 'WAE_GAN'
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

        # --- latent discriminator
        disc_input = tfk.Input(shape=(latent_dim, ))
        for index, layer_spec in enumerate(latent_discriminator_layers):
            if index == 0 :
                x = makeLayers(layer_spec=layer_spec)(disc_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)
        self.latent_disc = tfk.Model(inputs=disc_input, outputs=x)

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
        
    def compute_loss(self, x:Tensor) -> dict:
        z_data = self.encode(x)
        z_prior = self.gen_random(shape=tf.shape(z_data))
        x_recons = self.decode(z_data, apply_sigmoid=True)

        # latent discriminator
        y_data = self.latent_disc(z_data)
        y_prior = self.latent_disc(z_prior)

        y_prior_loss = tfk.losses.binary_crossentropy(tf.ones_like(y_prior), y_prior)
        y_data_loss = tfk.losses.binary_crossentropy(tf.zeros_like(y_data), y_data)
        disc_loss = tf.reduce_mean(0.5 * (y_data_loss + y_prior_loss))

        # reconstruct loss
        if self.loss_function_type == 'MSE':
            # MSE loss
            recon_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x, x_recons), axis=[1,2])
        elif self.loss_function_type == 'BCE':
            # BCE loss
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recons, labels=x)
            recon_loss = tf.reduce_mean(cross_ent, axis=[1,2,3])

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
