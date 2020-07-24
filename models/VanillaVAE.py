import tensorflow as tf
import numpy as np

from .BaseVAE import BaseVAE
from .utils import makeLayers, log_normal_pdf
from .types_ import *

tfk = tf.keras
tfkl = tf.keras.layers


class VanillaVAE(BaseVAE):
    def __init__(self, 
                latent_dim : int = None,
                input_shape : list = None,
                encoder_layers: list = None,
                decoder_layers: list = None,
                **kwargs):
        
        super(VanillaVAE, self).__init__()

        self.model_name = 'VanillaVAE'
        self.latent_dim = latent_dim

        # --- Encoder
        encoder_input = tfkl.Input(shape=input_shape)

        for index, layer_spec in enumerate(encoder_layers):
            if index is 0 :
                x = makeLayers(layer_spec=layer_spec)(encoder_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)

        self.encoder = tfk.Model(inputs=encoder_input, outputs=x)

        # --- Decoder
        decoder_input = tfkl.Input(shape=(latent_dim,))

        for index, layer_spec in enumerate(decoder_layers):
            if index is 0 :
                x = makeLayers(layer_spec=layer_spec)(decoder_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)

        self.decoder = tfk.Model(inputs=decoder_input, outputs=x)


    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar


    def decode(self, z):
        x = self.decoder(z)
        return x


    @tf.function
    def sample(self, sample_num: int =100, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(sample_num, self.latent_dim ))
        return self.decode(eps)


    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.exp(logvar * .5) + mean


    def compute_loss(self, x) -> dict:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruct_x = self.decode(z)
        
        # BCE loss
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruct_x, labels=x)
        rec_loss = tf.reduce_sum(cross_ent, axis=[1,2,3])

        # KL loss
        kl_loss = -0.5 * tf.reduce_sum((1 + logvar - mean**2 - tf.math.exp(logvar)), axis=1)

        return {'total_loss': tf.reduce_mean(rec_loss + kl_loss), 'rec_loss' : rec_loss, 'kl_loss' : kl_loss}


    def forward(self, x) -> List[Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def generate(self, x):
        return self.forward(x)

    @tf.function
    def train_step(self, x, opt=tfk.optimizers.Adam(1e-4)):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)['total_loss']
        gradients = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(gradients, self.trainable_variables))
        return 
