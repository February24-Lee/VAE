import tensorflow as tf
import numpy as np

from .BaseVAE import BaseVAE
from .utils import makeLayers
from .types_ import *

tfk = tf.keras
tfkl = tf.keras.layers

def log_normal_pdf(sample, mean, logvar):
    log2pi = tf.math.log(2. * np.pi)
    return -.5*((sample-mean)**2.*tf.exp(-logvar)+logvar+log2pi)
    

class BetaTCVAE(BaseVAE):
    def __init__(self,
                latent_dim : int = None,
                input_shape : list = None,
                encoder_layers: list = None,
                decoder_layers: list = None,
                alpha : float = None,
                beta : float = None,
                gamma : float = None,
                loss_function_type : str = 'MSE',
                data_size : int = None,
                **kwargs) -> None:
        super(BetaTCVAE, self).__init__()

        self.model_name = 'BetaTCVAE'
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_function_type = loss_function_type
        self.data_size = data_size

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

        
    
    def encode(self, x: Tensor ) -> List[Tensor]:
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z: Tensor, apply_sigmoid=False):
        if self.loss_function_type == 'BCE':
            x = self.decoder(z)
            if apply_sigmoid :
                x = tf.nn.sigmoid(x)
            return x
        elif self.loss_function_type == 'MSE':
            return tf.nn.tanh(self.decoder(z))
        else :
            return self.decoder(z)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.exp(logvar * .5) + mean

    @tf.function
    def sample(self, sample_num: int = 100, eps: Tensor =None) ->Tensor:
        if eps is None:
            eps = tf.random.normal(shape=(sample_num, self.latent_dim ))
        return self.decode(eps, apply_sigmoid=True)


    def forward(self, x: Tensor) -> Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, apply_sigmoid=True)

    @tf.function
    def compute_loss(self, x: Tensor) -> dict:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recons_x = self.decode(z) # not apply activate function

        batch_size =  z.shape[0]
        latent_dim = self.latent_dim
        dataset_size = self.data_size

        # --- reconstruct loss
        if self.loss_function_type == 'MSE':
            # MSE loss
            rec_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x, recons_x), axis=[1,2])
        elif self.loss_function_type == 'BCE':
            # BCE loss
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=recons_x, labels=x)
            rec_loss = tf.reduce_mean(cross_ent, axis=[1,2,3])

        # --- KL tern
        log_q_zx = tf.reduce_sum(log_normal_pdf(z, mean, logvar), axis=1)
        log_p_z = tf.reduce_sum(log_normal_pdf(z, tf.zeros_like(mean), tf.zeros_like(logvar)), axis=1)

        mat_log_q_z = log_normal_pdf(tf.reshape(z, [batch_size, 1, latent_dim]),
                                    tf.reshape(mean, [1, batch_size, latent_dim]),
                                    tf.reshape(logvar, [1, batch_size, latent_dim]))

        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))

        #importance_weight = tf.fill([batch_size, batch_size], 1/(batch_size-1))
        #tf.reshape(importance_weight, [-1])[::batch_size+1] = 1 / dataset_size
        #tf.reshape(importance_weight, [-1])[1::batch_size+1] = strat_weight
        #importance_weight[-1, 0] = strat_weight

        importance_weight_np = np.ones([batch_size, batch_size], dtype=np.float32) * (1/(batch_size-1))
        importance_weight_np.reshape(-1)[::batch_size+1] = 1 / dataset_size
        importance_weight_np.reshape(-1)[1::batch_size+1] = strat_weight
        importance_weight_np[-1,0] = strat_weight

        log_importance_weight = tf.math.log(importance_weight_np)

        mat_log_q_z += tf.reshape(log_importance_weight, [batch_size, batch_size, 1])

        log_q_z = tf.reduce_logsumexp(tf.reduce_sum(mat_log_q_z, axis=2), axis=1, keepdims=False)
        log_prod_q_z = tf.reduce_sum(tf.reduce_logsumexp(mat_log_q_z, axis=1, keepdims=False), axis=1)

        MI_loss = tf.reduce_mean(log_q_zx - log_q_z)
        TC_loss = tf.reduce_mean(log_q_z - log_prod_q_z)
        KLD_loss = tf.reduce_mean(log_prod_q_z - log_p_z)
        rec_loss = tf.reduce_mean(rec_loss)
        
        total_loss = rec_loss + self.alpha*MI_loss + self.beta * TC_loss + self.gamma * KLD_loss 

        return {'total_loss' : total_loss,
                'rec_loss' : rec_loss,
                'MI_loss' : MI_loss,
                'TC_loss': TC_loss,
                'KLD_loss': KLD_loss}
                
    @tf.function
    def train_step(self, x, opt=tfk.optimizers.Adam()) -> Tensor:
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)['total_loss']
        grad = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(grad, self.trainable_variables))
        return 
