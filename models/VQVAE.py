import tensorflow as tf
import numpy as np

from .BaseVAE import BaseVAE
from .utils import makeLayers
from .types_ import *

tfk = tf.keras
tfkl = tf.keras.layers

class VectorQuantizer(tf.Module):
    def __init__(self,
                num_embeddings:int = 512,
                latent_dim:int = 128,
                beta:float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = tfk.Sequential([tfkl.Embedding(input_dim=num_embeddings,
                                        output_dim=latent_dim)])

    def __call__(self, z:Tensor) -> Tensor:
        z_shape = tf.shape(z) # B x H x W x D
        flat_z = tf.reshape(z, [-1, self.latent_dim]) # BHW x D

        # distance |z-e|^2
        _z2 = tf.reduce_sum(flat_z**2, axis=1, keepdims=True)
        _ze = tf.matmul(flat_z, tf.squeeze(tf.transpose(self.embedding.weights)))
        _e2 = tf.reduce_sum(tf.squeeze(tf.square(self.embedding.weights)))

        dist = _z2 - 2*_ze + _e2

        argmin = tf.reshape(tf.argmin(dist, axis=1), [-1,1]) # BHW x 1

        one_hot = tf.squeeze(tf.one_hot(argmin, self.num_embeddings, dtype=tf.float32)) # BHW x K

        quantized_z = tf.matmul(one_hot, self.embedding.weights) # BHW x D
        quantized_z = tf.reshape(quantized_z, z_shape) # B x H x W x D

        return quantized_z
        


class VQVAE(BaseVAE):
    def __init__(self,
                input_shape:list = [128, 128, 3],
                latent_dim:int = 128,
                num_embeddings:int = 512,
                beta:float = 0.25,
                encoder_layers: list = None,
                decoder_layers: list = None,
                **kwargs) -> None :
        super(VQVAE, self).__init__()
        self.model_name = 'VQVAE'
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        # --- Encoder
        encoder_input = tfkl.Input(shape=input_shape)
        for index, layer_spec in enumerate(encoder_layers):
            if index is 0 :
                x = makeLayers(layer_spec=layer_spec)(encoder_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)
        self.encoder = tfk.Model(inputs=encoder_input, outputs=x)
        latent_shape = self.encoder.output.shape[1]

        # --- VQ
        self.vq = VectorQuantizer(num_embeddings=num_embeddings,
                                latent_dim=latent_dim)

        # --- Decoder
        decoder_input = tfkl.Input(shape=(latent_shape, latent_shape, 128))
        for index, layer_spec in enumerate(decoder_layers):
            if index is 0 :
                x = makeLayers(layer_spec=layer_spec)(decoder_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)
        self.decoder = tfk.Model(inputs=decoder_input, outputs=x)

        return 

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return z

    def decode(self, z:Tensor, apply_sigmoid=False) -> Tensor:
        x = self.decoder(z)
        if apply_sigmoid:
            return tf.nn.sigmoid(x)
        return x

    @tf.function
    def compute_loss(self, x, **kwargs):
        z = self.encoder(x)
        quantized_z = self.vq(z)

        #Embedding_loss
        embedding_loss = tf.nn.l2_loss(tf.stop_gradient(z)-quantized_z)

        #Commitment_loss
        commitment_loss = tf.nn.l2_loss(z-tf.stop_gradient(quantized_z))

        # Residue back for update
        quantized_z = z + tf.stop_gradient(quantized_z - z)

        recon_x = self.decode(quantized_z, apply_sigmoid=False)

        #Reconstruct loss
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=recon_x, labels=x)
        recon_loss = tf.reduce_mean(tf.reduce_mean(cross_ent, axis=[1,2,3]))

        #Total_loss
        total_loss = recon_loss + embedding_loss + self.beta * commitment_loss
        return {'total_loss':total_loss,
                'rec_loss':recon_loss,
                'embedding_loss':embedding_loss,
                'commitment_loss':commitment_loss}


    def forward(self, x:Tensor) -> Tensor:
        z = self.encode(x)
        quantized_z = self.vq(z)
        return self.decode(quantized_z, apply_sigmoid=True)

    @tf.function
    def train_step(self, x, opt=tfk.optimizers.Adam(1e-4)):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)['total_loss']
        
        layer_list = [*self.trainable_variables, *self.vq.trainable_variables]
        gradients = tape.gradient(loss, layer_list)
        opt.apply_gradients(zip(gradients, layer_list))
        return 