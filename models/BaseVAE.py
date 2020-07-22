typefrom .types_ import *
from abc import abstractclassmethod
import tensorflow as tf

tfk = tf.keras

class BaseVAE(tfk.Model):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
    
    def encode(self, input: Tensor ) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    @tf.function
    def sample(self, eps: Tensor =None) ->Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractclassmethod
    def forward(self, input: Tensor) -> Tensor:
        pass

    @abstractclassmethod
    def compute_loss(self, input: Tensor, **kwargs) -> Tensor:
        pass

    @abstractclassmethod
    def train_step(self, x, opt) -> Tensor:
        pass
