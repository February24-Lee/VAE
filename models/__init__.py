from .BaseVAE import *
from .trainer import *
from .types_ import *
from utils import *
from .VanillaVAE import *
from .trainer import *

VAE = VanillaVAE

vae_models = {
    'VanillaVAE' : VanillaVAE
}