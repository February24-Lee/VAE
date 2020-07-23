import tensorflow as tf
import numpy as np

from .BaseVAE import BaseVAE
from .utils import makeLayers, log_normal_pdf
from .types_ import *

tfk = tf.keras
tfkl = tf.keras.layers



class IWAE(BaseVAE):
    def __init__(self, 
                latent_dim : int = None,
                input_shape : list = None,
                encoder_layers: list = None,
                decoder_layers: list = None,
                sample_num: int = 5,
                **kwargs):
        
        super(IWAE, self).__init__()

        self.model_name = 'IWAE'
        self.latent_dim = latent_dim
        self.sample_num = sample_num

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
        logits = self.decoder(z)
        return logits


    @tf.function
    def sample(self, sample_num: int =1, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(sample_num, self.latent_dim ))
        return tf.nn.sigmoid(self.decode(eps))


    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.exp(logvar * .5) + mean


    def compute_loss(self, x):
        # B x D
        mean, logvar = self.encode(x)

        # BS X D
        mu = tf.repeat(mu, [self.sample_num]*len(mu), axis=0) 
        logvar = tf.repeat(logvar, [self.sample_num]*len(mu), axis=0)
        z = self.reparameterize(mean, logvar)
        x_logits = self.decode(z)

        # BS x H x W x C
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x)

        # BS
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1,2,3]) 
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


    def forward(self, x) -> List[Tensor]:
        # B x D
        mu, logvar = self.encode(x) 

        # BS X D
        mu = tf.repeat(mu, [self.sample_num]*len(mu), axis=0) 
        logvar = tf.repeat(logvar, [self.sample_num]*len(mu), axis=0)

        # BS x D
        z = self.reparameterize(mu, logvar)

        return [tf.nn.sigmoid(self.decoder(z)), mu, logvar]

    def generate(self, x):
        return self.forward(x)

    @tf.function
    def train_step(self, x, opt=tfk.optimizers.Adam(1e-4)):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(gradients, self.trainable_variables))
        return 
