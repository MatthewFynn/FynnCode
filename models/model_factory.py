"""
    model_factory.py
    Author: Matthew Fynn

    Purpose: To create models
"""

from typing import Optional, Dict
from torch.optim import lr_scheduler
import os
import torch.optim as optim

#models and configs
from models.cnns.wav2vec2_cnn import (
    Wav2VecCNNConfig,
    Wav2VecCNN,
)

from models.cnns.encodec_cnn import (
    EnCodecConfig,
    EnCodecCNN
)

from models.cnns.mfcc_cnn import (
    mfccCNNConfig,
    mfccCNN
)

from models.cnns.mfcc_2D_cnn import (
    mfcc2DCNNConfig,
    mfcc2DCNN
)

from models.cnns.UNET_SSL_cnn import (
    UNET_SSL_Config,
    UNET_SSL_CNN
)

from models.cnns.UNET_SSL_cnn_2D import(
    UNET_SSL_Config2D,
    UNET_SSL_CNN2D
)


from models.cnns.opera_ce import(
    OperaCEConfig,
    OperaCE
)

from models.cnns.hubert_cnn import(
    HuBERTCNNConfig,
    HuBERTCNN
)

from models.cnns.hybrid import (
    HybridConfig,
    HybridModel
)

from models.multi_audio import (
    MultiInputAudioConfig,
    MultiInputAudioModel
)


#^add new models here^

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)


#define function to return optimizer and scheduler
def get_optimizer_and_scheduler(
        params, 
        optimizer_type: str, 
        lr: Optional[float] = None,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 1e-5,
        momentum: float = 0.176,
        step_size: int = 2,
        gamma: float = 0.24,
        **kwargs
    ):
    """
    Returns the required optimiser.
    """
    fun = lambda epoch: gamma ** (epoch // step_size) 

    if lr is None:
        if optimizer_type == 'sgd':
            lr = 0.001
            momentum = 0.9
            step_size = 3
            gamma = 0.1
        elif optimizer_type == 'adam':
            # lr = 1e-4
            lr = 1e-3
        elif optimizer_type == 'adamw':
            lr = 1e-4
        elif optimizer_type == "rmsprop":
            lr = 1e-4
            weight_decay = 6.1148e-5
        elif optimizer_type == 'adamhy':
            lr = 1e-4
        else:
            lr = 1e-5
    
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer, lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_epsilon, weight_decay=weight_decay)
        return optimizer, lr_scheduler.LambdaLR(optimizer, lr_lambda=fun)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return optimizer, lr_scheduler.LambdaLR(optimizer, lr_lambda=fun)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        return optimizer, lr_scheduler.LambdaLR(optimizer, lr_lambda=fun)
    else:
        raise NotImplementedError(f"Invalid Optimiser: {optimizer_type}")

def get_Wav2VecCNN(device, config: PretrainedConfig):
    """Creates a Wav2Vec CNN model for training"""
    model = Wav2VecCNN(config)
    # model = model.to(device)

    return model

def get_EnCodecCNN(device, config: PretrainedConfig):
    """Creates a Encodec CNN model for training"""
    model = EnCodecCNN(config)
    # model = model.to(device)

    return model

def get_mfccCNN(device, config: PretrainedConfig):
    """Creates a Wav2Vec CNN model for training"""
    model = mfccCNN(config)
    # model = model.to(device)

    return model

def get_mfcc2DCNN(device, config: PretrainedConfig):
    """Creates a Wav2Vec CNN model for training"""
    model = mfcc2DCNN(config)
    # model = model.to(device)

    return model

def get_UnetSSLCNN(device, config: PretrainedConfig):
    model = UNET_SSL_CNN(config)

    return model

def get_UnetSSLCNN2D(device, config: PretrainedConfig):
    model = UNET_SSL_CNN2D(config)

    return model

def get_UnetSSLCNN2D_L(device, config: PretrainedConfig):
    model = UNET_SSL_CNN2D_La(config)

    return model

# def get_Wav2Vec(device, config: PretrainedConfig):
#     """Creates a Wav2Vec with transformer model for training"""
#     model = Wav2Vec(config)
#     # model = model.to(device)

#     return model

def get_OperaCE(device, config: PretrainedConfig):
    model = OperaCE(config)

    return model

def get_HuBERTCNN(device, config: PretrainedConfig):
    model = HuBERTCNN(config)

    return model


def get_Hybrid(device, config: PretrainedConfig):
    model = HybridModel(config)
    return model

def get_multi_input_audio(device, config: PretrainedConfig, models: list[PreTrainedModel], aux_model_code: str):
    """
    Returns the MultiInputAudio Model
    """
    config.aux_model_type = aux_model_code
    model = MultiInputAudioModel(config, models)

    for param in model.parameters():
        param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model

#^ add more models to return here^


def get_audio_models():
    return (
        "wav2vec_cnn",
        "encodec_cnn",
        "wav2vec",
        "mfcc_cnn",
        "mfcc2d_cnn",
        "unetssl_cnn",
        "unetssl2d_cnn",
        "unetssl2d_cnn_large",
        "opera_ce",
        "hubert_cnn",
        "hybrid"
    )

def get_multi_models():
    return (
        "stack",
    )

class ModelFactory():
    """
    Model factory to make it easier to change between models and stuff
    """

    def __init__(self, device):
        self.device = device

        self.model_configs = self._get_model_configs()
        self.model_classes = self._get_models()
        self.audio_model = get_audio_models()
        self.multi_models = get_multi_models()


    def _get_model_configs(self):
        return {
            "wav2vec_cnn": Wav2VecCNNConfig,
            "encodec_cnn": EnCodecConfig,
            # "wav2vec": Wav2VecConfig,
            "mfcc_cnn": mfccCNNConfig,
            "mfcc2d_cnn": mfcc2DCNNConfig,
            "unetssl_cnn": UNET_SSL_Config,
            "unetssl2d_cnn": UNET_SSL_Config2D,
            "unetssl2d_cnn_large": UNET_SSL_Config2D_La,
            "opera_ce": OperaCEConfig,
            "hubert_cnn": HuBERTCNNConfig,
            "hybrid": HybridConfig,
            "stack": MultiInputAudioConfig
        }

    def _get_models(self):
        return {
            "wav2vec_cnn": Wav2VecCNN,
            "encodec_cnn": EnCodecCNN,
            # "wav2vec": Wav2Vec,
            "mfcc_cnn": mfccCNN,
            "mfcc2d_cnn": mfcc2DCNN,
            "unet_ssl": UNET_SSL_CNN,
            "unetssl2d_cnn": UNET_SSL_CNN2D,
            "unetssl2d_cnn_large": UNET_SSL_CNN2D_La,
            "opera_ce": OperaCE,
            "hubert_cnn": HuBERTCNN,
            "hybrid": HybridModel,
            "stack": MultiInputAudioModel
        }

    def get_config(self, 
            model_code: str, 
            config: dict, 
            models:Optional[list[PreTrainedModel]] = None,
            aux_model_code:Optional[str] = None
        ) -> PretrainedConfig:
        
        if config == {}:
            return self._get_model_configs()[model_code]()
        else:
            return self._get_model_configs()[model_code](**config)


    def get_class(self,
            model_code: str, 
    ):
        try:
            return self.model_classes[model_code]
        except KeyError:
            raise ValueError(f"Invalid model code: {model_code}")

    def create_model(self, 
            model_code: str, 
            config: dict, 
            models: Optional[list[PreTrainedModel]] = None, 
            aux_model_code: Optional[str] = None
        ) -> PreTrainedModel:
        """
        Creates the model specified by the model_code
        """
        model_config = self.get_config(model_code, config)
        
        if model_code == "wav2vec_cnn":
            return get_Wav2VecCNN(self.device, model_config) 
        elif model_code == "encodec_cnn":
            return get_EnCodecCNN(self.device, model_config)
        elif model_code == "wav2vec":
            return get_Wav2Vec(self.device, model_config)
        elif model_code == "mfcc_cnn":
            return get_mfccCNN(self.device, model_config)
        elif model_code == "mfcc2d_cnn":
            return get_mfcc2DCNN(self.device, model_config)
        elif model_code == "unetssl_cnn":
            return get_UnetSSLCNN(self.device, model_config)
        elif model_code == "unetssl2d_cnn":
            return get_UnetSSLCNN2D(self.device, model_config)
        elif model_code == "unetssl2d_cnn_large":
            return get_UnetSSLCNN2D_L(self.device, model_config)
        elif model_code == "opera_ce":
            return get_OperaCE(self.device, model_config)
        elif model_code == "hubert_cnn":
            return get_HuBERTCNN(self.device, model_config)
        elif model_code == "hybrid":
            return get_Hybrid(self.device, model_config)
        elif model_code == "stack":
            if models is None or aux_model_code is None:
                raise Exception("Must provide models to ensemble")

            return get_multi_input_audio(self.device, model_config, models, aux_model_code)

        else:
            raise ValueError(f"Invalid CNN model: {model_code=}")
    
    def load_model(self, model_code: str, dir):
        if model_code == "wav2vec_cnn":
            return Wav2VecCNN.from_pretrained(dir)
        elif model_code == "encodec_cnn":
            return EnCodecCNN.from_pretrained(dir)
        elif model_code == "wav2vec":
            return Wav2Vec.from_pretrained(dir)
        elif model_code == "mfcc_cnn":
            # x = mfccCNN.from_pretrained(dir)
            return mfccCNN.from_pretrained(dir)
        elif model_code == "mfcc2d_cnn":
            return mfcc2DCNN.from_pretrained(dir)
        elif model_code == "unetssl_cnn":
            return UNET_SSL_CNN.from_pretrained(dir)
        elif model_code == "unetssl2d_cnn":
            return UNET_SSL_CNN2D.from_pretrained(dir)
        elif model_code == "unetssl2d_cnn_large":
            return UNET_SSL_CNN2D_La.from_pretrained(dir)
        elif model_code == "opera_ce":
            return OperaCE.from_pretrained(dir)
        elif model_code == "hubert_cnn":
            return HuBERTCNN.from_pretarained(dir)
        elif model_code == "hybrid":
            return HybridModel.from_pretrained(dir)
        elif model_code == "stack":
            return MultiInputAudioModel.from_pretrained(dir)
        
    def get_fs(self,model_code: str):
        if model_code == "wav2vec_cnn":
            # return 4125
            return 2000
        elif model_code == "wav2vec":
            # return 4125
            return 4125
        elif model_code == "encodec_cnn":
            # return 24000
            return 2000
        elif model_code == "mfcc_cnn":
            return 2000
        elif model_code == "mfcc2d_cnn":
            return 2000
        elif model_code == "unetssl_cnn":
            return 2000
        elif model_code == "unetssl2d_cnn":
            return 2000
        elif model_code == "unetssl2d_cnn_large":
            return 2000
        elif model_code == "opera_ce":
            return 16000
        elif model_code == "hubert_cnn":
            return 16000
