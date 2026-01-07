"""
    encodec_cnn.py
    Author: Matthew Fynn

    Purpose: Contains Encodec based classifier and model.
"""

import torch
import torch.nn as nn

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    EncodecModel
)

class EnCodecConfig(PretrainedConfig):
    """
    This is the config class for the custom EnCodec Based Audio Feature Extracter

    Args:
        num_classes (int) : Number of classes to classify
        hidden_size (int) : Size of the hidden layer
    """
    model_type = "EncodecCNN_classifier"
    def __init__(
        self,
        num_classes=2,
        hidden_size: int = 256,
        fs: int = 24000,
        signal_len_t: int = 2, #seconds
        num_cnn_layers = 0,
        dropout: float = 0,
        **kwargs
    ):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.signal_len = int(signal_len_t*fs)
        self.fs = fs
        self.num_cnn_layers = num_cnn_layers
        self.dropout = dropout

class EnCodecCNN(PreTrainedModel):

    config_class = EnCodecConfig

    def __init__(self, config: PretrainedConfig, **kwargs): 
        super(EnCodecCNN, self).__init__(config)

        self.criterion = nn.CrossEntropyLoss()
        self.ft_extractor = EncodecModel.from_pretrained("facebook/encodec_24khz").encoder
        
        # print(self.ft_extractor)
        # input('hey')
        # Limit the number of layers in the encoder, if specified
        # if self.config.num_cnn_layers != 0:
        self.ft_extractor.layers = nn.ModuleList(self.ft_extractor.layers[0:self.config.num_cnn_layers])
        # print(self.config.num_cnn_layers)
        # input('here')
        
        # print(self.ft_extractor)
        # input('here')


        self.cnn_out_size = self._get_cnn_size()

        if type(self.config.hidden_size) == int:
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
        input_tensor = torch.zeros(64, 1, self.config.signal_len)
        output_tensor = self.ft_extractor(input_tensor).mean(dim=2)
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
        # Without altering the feature extraction.
        # with torch.no_grad():)
        # print(input_vals.shape)
        if input_vals.dim() == 2:
            input_vals = input_vals.unsqueeze(1)
        # print(input_vals.shape)
        # input('hey')

        out = self.ft_extractor(input_vals)
        # print(out.shape)
        # input('hi')
        out = out.mean(dim=2)
        # print(out.shape)
        # input('hi')
        out = self.classifier(out)

        return out