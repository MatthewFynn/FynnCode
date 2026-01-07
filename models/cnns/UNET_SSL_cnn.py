"""
    wav2vec_cnn.py
    Author: Milan Marocchi

    Purpose: Contains custom SSL model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import os
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)

import sys
# Add the parent directory to sys.path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)

from _SSL_Model.unet_model_4layers_skip_ft import UNet1d, UNETConfig, UNetSSLFeatureExtractor_2L, UNetSSLFeatureExtractor_3L, UNetSSLFeatureExtractor_4L

class UNET_SSL_Config(PretrainedConfig):
    """
    This is the config class for the custom Wav2Vec Based Audio classifier model 

    Args:
        num_classes (int) : Number of classes to classify
        hidden_size (int) : Size of the hidden layer
    """

    model_type = "unet_ssl_classifier"

    def __init__(
        self,
        num_classes=2,
        hidden_size: int = 512,
        fs: int = 2000,
        signal_len_t: int = 2, #seconds
        dropout: float = 0,
        num_cnn_layers = 0,
        checkpoint = 'ep1',
        trained_path = 'trained_SSL_model',
        **kwargs
    ):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.signal_len_t = signal_len_t
        self.fs = fs
        self.dropout = dropout
        self.num_cnn_layers = num_cnn_layers
        self.checkpoint = checkpoint
        self.trained_path = trained_path

class UNET_SSL_CNN(PreTrainedModel):

    config_class = UNET_SSL_Config

    def __init__(self, config: PretrainedConfig, **kwargs): 
        super(UNET_SSL_CNN, self).__init__(config)

        self.criterion = nn.CrossEntropyLoss()
        self.path = os.path.join('_SSL_Model',self.config.trained_path)
        ssl_config = UNETConfig.from_pretrained(os.path.join(self.path,self.config.checkpoint))
        model = UNet1d.from_pretrained(os.path.join(self.path,self.config.checkpoint), config = ssl_config)

        if self.config.num_cnn_layers == 2:
            self.ft_extractor = UNetSSLFeatureExtractor_2L(model.feature_extractor)
        elif self.config.num_cnn_layers == 3:
            self.ft_extractor = UNetSSLFeatureExtractor_3L(model.feature_extractor)
        elif self.config.num_cnn_layers == 4:
            self.ft_extractor = UNetSSLFeatureExtractor_4L(model.feature_extractor)
        
        self.cnn_out_size = self._get_cnn_size()

        # if type(self.config.hidden_size) == int:
        #     self.classifier = nn.Sequential(
        #         nn.Linear(self.cnn_out_size, self.config.hidden_size),
        #         nn.GELU(),
        #         nn.Linear(self.config.hidden_size, self.config.num_classes)
        #     )
        # else:
        #     self.classifier = self.build_classifier()
        if isinstance(self.config.hidden_size, int):
            layers = [
                nn.Linear(self.cnn_out_size, self.config.hidden_size),
                nn.GELU(),
            ]

            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))

            layers.append(nn.Linear(self.config.hidden_size, self.config.num_classes))

            self.classifier = nn.Sequential(*layers)

        else:
            self.classifier = self.build_classifier()

        self.num_classes = config.num_classes

    def _get_cnn_size(self):
        input_tensor = torch.zeros(1, 1, int(self.config.signal_len_t*self.config.fs))
        with torch.no_grad():
            features = self.ft_extractor(input_tensor)
            # print(features.shape)
            pooled_features = features.mean(dim=2)

        # print(pooled_features.shape)
        # input('hi')
        return pooled_features.shape[-1]
    
    def build_classifier(self):
        layers = list()
        in_features = self._get_cnn_size()

        for layer_size in self.config.hidden_size:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.GELU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            in_features = layer_size

        layers.append(nn.Linear(in_features, self.config.num_classes))
        return nn.Sequential(*layers)
 

    def forward(self, input_vals, **kwargs):

        input_vals = input_vals.unsqueeze(1)

        # Pass input through the encoder (feature extractor)
        features = self.ft_extractor(input_vals)
        pooled_features = features.mean(dim=2)  # Global average pooling over time
        # print(features.shape)
        # print(pooled_features.shape)
        # input('hi')
        out = self.classifier(pooled_features)
        return out


