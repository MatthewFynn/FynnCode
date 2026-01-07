"""
    wav2vec_cnn.py
    Author: Milan Marocchi

    Purpose: Contains Wav2Vec 2.0 based classifier and model.
"""

import torch
import torch.nn as nn
import gc

from transformers import (
    HubertModel,
    PretrainedConfig,
    PreTrainedModel
)


class HuBERTCNNConfig(PretrainedConfig):
    """
    This is the config class for the custom Wav2Vec Based Audio classifier model 

    Args:
        num_classes (int) : Number of classes to classify
        hidden_size (int) : Size of the hidden layer
    """

    model_type = "hubert_classifier"

    def __init__(
        self,
        num_classes=2,
        hidden_size = 512,
        fs: int = 2000,
        signal_len_t: int = 2, #seconds
        dropout: float = 0,
        num_cnn_layers = 2,
        **kwargs
    ):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.signal_len_t = signal_len_t
        self.fs = fs
        self.dropout = dropout
        self.num_cnn_layers = num_cnn_layers

class HuBERTCNN(PreTrainedModel):

    config_class = HuBERTCNNConfig

    def __init__(self, config: PretrainedConfig, **kwargs): 
        super(HuBERTCNN, self).__init__(config)

        self.criterion = nn.CrossEntropyLoss()
        # Load full model
        full_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        # print(full_model)
        # all_params = torch.cat([p.data.flatten() for p in full_model.parameters()])
        # print("Global mean:", all_params.mean().item())
        # input('here')

        # Keep only the feature extractor
        self.ft_extractor = full_model.feature_extractor  # CNN part
        
        # Delete the transformer encoder to free up VRAM
        del full_model.encoder
        torch.cuda.empty_cache()  # Clear unused memory
        gc.collect()  # Force garbage collection
        # self.ft_extractor = full_model.feature_extractor # type: ignore
         # Use only the specified number of convolutional layers
        # print(self.ft_extractor)
        if self.config.num_cnn_layers > 0:
            self.ft_extractor.conv_layers = nn.ModuleList(
                self.ft_extractor.conv_layers[:self.config.num_cnn_layers]
            )
        # input('hi')
        # print(self.ft_extractor)
        # input('hi')

        # if self.config.dropout > 0:
        #     for layer in self.ft_extractor.conv_layers:
        #         layer.conv = nn.Sequential(layer.conv, nn.Dropout(self.config.dropout))

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

    def _get_cnn_size(self):

        input_tensor1 = torch.zeros(64, int(self.config.signal_len_t*self.config.fs))
        output_tensor = self.ft_extractor(input_tensor1).mean(dim=2)
        # output_tensor = self.ft_extractor(input_tensor1).flatten(start_dim=1)
        # print(output_tensor.shape)
        # input()
        return output_tensor.shape[-1]

    def extract_features(self, input_vals):
        if input_vals.dim() != 2:
            raise ValueError(f"Expected 'inputs' tensor to have shape [batch_size, sequence_len, num_models/channels]. {input_vals.dim()=}")
        return self.ft_extractor(input_vals).last_hidden_state.mean(dim=2) # type: ignore

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
        if input_vals.dim() != 2:
            raise ValueError(f"Expected 'inputs' tensor to have shape [batch_size, sequence_len, num_models/channels]. {input_vals.dim()=}")
        # print(input_vals)
        # print(self.ft_extractor(input_vals))
        # print(self.ft_extractor(input_vals).mean(dim=2))
        # print(input_vals.shape)
        # input('lol')
        out = self.ft_extractor(input_vals)
        # print(out.shape)
        out = out.mean(dim=2)
        # out = out.flatten(start_dim=1)
        # print(out.shape)
        # input()
        out = self.classifier(out)

        return out
