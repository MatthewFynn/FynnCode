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

from _SSL_Model_MFCC.unet_model_4layers_skip_ft_2D import UNet2D, UNETConfig2D, UNetSSLFeatureExtractor2D, UNetSSLFeatureExtractor2D_3L, UNetSSLFeatureExtractor2D_4L


class UNET_SSL_Config2D(PretrainedConfig):
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

class UNET_SSL_CNN2D(PreTrainedModel):

    config_class = UNET_SSL_Config2D

    def __init__(self, config: PretrainedConfig, **kwargs): 
        super(UNET_SSL_CNN2D, self).__init__(config)

        self.criterion = nn.CrossEntropyLoss()
        self.path = os.path.join('_SSL_Model_MFCC',self.config.trained_path)
        ssl_config = UNETConfig2D.from_pretrained(os.path.join(self.path,self.config.checkpoint))
        model = UNet2D.from_pretrained(os.path.join(self.path,self.config.checkpoint), config = ssl_config)

        #SELECT APPROPRIATE CNN LAYER
        # print(self.config.num_cnn_layers)
        # input('hi')
        if self.config.num_cnn_layers == 2:
            self.ft_extractor = UNetSSLFeatureExtractor2D(model.feature_extractor)
        elif self.config.num_cnn_layers == 3:
            self.ft_extractor = UNetSSLFeatureExtractor2D_3L(model.feature_extractor)
        elif self.config.num_cnn_layers == 4:
            self.ft_extractor = UNetSSLFeatureExtractor2D_4L(model.feature_extractor)
        
        # self.ft_extractor = UNetSSLFeatureExtractor2D(model.feature_extractor)


        self.time_to_mfcc = model.time_to_mfcc
        self.num_mfccs = model.config.num_mfccs

        self.cnn_out_size = self._get_cnn_size()

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

        # self.head = TemporalAttentionHead(in_channels=32, embed_dim=512,
        #                                   attn_hidden=64, dropout=0.1)
        # self.classifier2 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.GELU(),
        #     # nn.Dropout(0.3),
        #     nn.Linear(512, self.num_classes)
        # )



    def _get_cnn_size(self):
        input_tensor = self.time_to_mfcc(torch.rand(64, int(self.config.signal_len_t*self.config.fs)), self.num_mfccs).unsqueeze(1) #will have to change to incorporate mfccs
        output_tensor = self.ft_extractor(input_tensor).flatten(start_dim=1)
        # output_tensor = self.ft_extractor(input_tensor).mean(dim=-1).flatten(start_dim=1)
        return output_tensor.shape[-1]

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
        # print(input_vals.shape)

        # print(input_vals_mfcc.shape)
        # Pass input through the encoder (feature extractor)
        input_vals_mfcc = self.time_to_mfcc(input_vals=input_vals, n_mfcc=self.num_mfccs).unsqueeze(1)
        features = self.ft_extractor(input_vals_mfcc)

        features = features.flatten(start_dim=1)
        
        out = self.classifier(features)
        return out

        ###########################NEW
        # 1) Time -> MFCC (do this ONCE)
        # input_vals_mfcc = self.time_to_mfcc(input_vals=input_vals,
        #                                     n_mfcc=self.num_mfccs).unsqueeze(1)
        # # 2) CNN feature extractor -> (B, C, F, T)
        # features = self.ft_extractor(input_vals_mfcc)

        # # 3) Temporal attention pooling -> (B, D)
        # emb = self.head(features)

        # # 4) Classifier
        # out = self.classifier2(emb)
        # return out

# class TemporalAttentionHead(nn.Module):
#     """
#     Input:  x (B, C, F, T)
#     Output: z (B, D)  -- attention-weighted temporal summary
#     """
#     def __init__(self, in_channels: int, embed_dim: int = 512,
#                  attn_hidden: int = 64, dropout: float = 0.1):
#         super().__init__()
#         # Collapse frequency with average pooling -> (B, C, 1, T) -> (B, C, T)
#         self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
#         self.attn = nn.Sequential(
#             nn.Conv1d(in_channels, attn_hidden, kernel_size=1),
#             nn.Tanh(),
#             nn.Conv1d(attn_hidden, 1, kernel_size=1)  # (B,1,T)
#         )
#         self.proj = nn.Sequential(
#             nn.Linear(in_channels, embed_dim),
#             nn.LayerNorm(embed_dim)
#         )
#         self.do = nn.Dropout(dropout)

#     def forward(self, x):                 # x: (B,C,F,T)
#         x = self.freq_pool(x).squeeze(2)  # (B,C,T)
#         scores = self.attn(x)             # (B,1,T)
#         alpha = torch.softmax(scores, dim=-1)
#         x_att = (x * alpha).sum(dim=-1)   # (B,C)
#         z = self.proj(self.do(x_att))     # (B,D)
#         return z

