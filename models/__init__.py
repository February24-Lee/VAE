from .BaseVAE import *
from .trainer import *
from .types_ import *
from utils import *
from .VanillaVAE import *
from .trainer import *
from .IWAE import *
from .VQVAE import *
from .WAE_GAN import *
from .WAE_MMD import *
from .InfoVAE import *
from .BetaVAE import *
from .BetaTCVAE import *


VAE = VanillaVAE

vae_models = {
    'VanillaVAE' : VanillaVAE,
    'IWAE' : IWAE,
    'VQVAE' : VQVAE,
    'WAE_GAN' : WAE_GAN,
    'WAE_MMD' : WAE_MMD,
    'INFOVAE' : INFOVAE,
    'BetaVAE' : BetaVAE,
    'BetaTCVAE' : BetaTCVAE
}