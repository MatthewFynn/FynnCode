"""
    wav2vec_cnn.py
    Author: Milan Marocchi

    Purpose: Contains Wav2Vec 2.0 based classifier and model.
"""

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import librosa


from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)


class mfcc2DCNNConfig(PretrainedConfig):
    """
    This is the config class for the custom Wav2Vec Based Audio classifier model 

    Args:
        num_classes (int) : Number of classes to classify
        hidden_size (int) : Size of the hidden layer
    """

    model_type = "mfcc_classifier"

    def __init__(
        self,
        num_classes=2,
        hidden_size: int = 512,
        fs: int = 2000,
        signal_len_t: int = 2, #seconds
        dropout: float = 0.2,
        frame_length: float = 0.04,
        overlap: float = 0.5,
        num_mfccs: int = 13,
        mlp_flag:int =0,
        ks: int = 3,
        ks2:int = 3,
        num_cnn_layers:int = 2,
        **kwargs
    ):
        self.num_classes = num_classes
        self.mlp_flag = mlp_flag
        self.hidden_size = [hidden_size,256] if self.mlp_flag ==1 else hidden_size
        self.signal_len_t = signal_len_t
        # self.signal_len = int(signal_len_t*fs)
        self.fs = fs
        self.dropout = dropout
        self.frame_length = frame_length
        self.n_fft = int(self.frame_length*self.fs)
        self.hop_length = int(int(self.frame_length*self.fs) * (1 - overlap))
        self.num_mfccs = num_mfccs
        self.ks=ks
        self.ks2 = ks2
        self.num_cnn_layers = num_cnn_layers
        self.pruned_heads = {}

class mfcc2DCNN(PreTrainedModel):

    config_class = mfcc2DCNNConfig

    def __init__(self, config: PretrainedConfig, **kwargs): 
        super(mfcc2DCNN, self).__init__(config)

        self.criterion = nn.CrossEntropyLoss()
        if self.config.num_cnn_layers == 2:
            self.ft_extractor = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(self.config.ks, self.config.ks2), stride=(1, 1), padding=1),
                nn.Dropout2d(p=self.config.dropout),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(self.config.ks, self.config.ks2), stride=(1, 1), padding=1),
                nn.Dropout2d(p=self.config.dropout),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
        elif self.config.num_cnn_layers == 3:
                self.ft_extractor = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(self.config.ks, self.config.ks2), stride=(1, 1), padding=1),
                nn.Dropout2d(p=self.config.dropout),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(self.config.ks, self.config.ks2), stride=(1, 1), padding=1),
                nn.Dropout2d(p=self.config.dropout),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(self.config.ks, self.config.ks2), stride=(1, 1), padding=1),
                nn.Dropout2d(p=self.config.dropout),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),

            )

        
        self.fs = self.config.fs
        self.num_classes = config.num_classes

        object.__setattr__(self, "_mfcc_transform", torchaudio.transforms.MFCC(
            sample_rate=config.fs,
            n_mfcc=config.num_mfccs,
            melkwargs={
                'n_fft': config.n_fft,
                'n_mels': 21,
                'hop_length': config.hop_length,
                'f_max': config.fs / 2,
                'f_min': 0.0,
                'window_fn': torch.hann_window,
                'power': 2.0,
                'mel_scale': 'slaney',
                'norm': 'slaney',
            },
            dct_type=2,
            norm='ortho',
            log_mels=True,
        ))
        # Hugging Face style weight init
        self.post_init()

        self.cnn_out_size = self._get_cnn_size()
        if type(self.config.hidden_size) == int:
            self.classifier = nn.Sequential(
                nn.Linear(self.cnn_out_size, self.config.hidden_size), 
                nn.GELU(),
                nn.Linear(self.config.hidden_size, self.config.num_classes)
            )
        else:
            self.classifier = self.build_classifier()

    def _get_cnn_size(self):
        # print(self.config.fs)
        # print(self.config.signal_len_t)
        # input()
        input_tensor = self.time_to_mfcc(torch.rand(64, int(self.config.signal_len_t*self.config.fs)), self.config.num_mfccs).unsqueeze(1) #will have to change to incorporate mfccs
        # print(input_tensor.shape)
        # input('hi')
        output_tensor = self.ft_extractor(input_tensor).mean(dim=-1)
        output_tensor = output_tensor.flatten(start_dim=1)
        # print(output_tensor.shape)
        # input('hello friends')
        return output_tensor.shape[-1]

    def build_classifier(self):
        layers = list()
        in_features = self._get_cnn_size()
        for layer_size in self.config.hidden_size:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.GELU())
            in_features = layer_size

        layers.append(nn.Linear(in_features, self.config.num_classes))
        return nn.Sequential(*layers)
    
    
    def time_to_mfcc(self, input_vals, n_mfcc, normalize=True):
        # Ensure input is 2D (batch, time)
        if input_vals.dim() == 1:
            input_vals = input_vals.unsqueeze(0)

        self._mfcc_transform.to(input_vals.device)
        # Apply the transform
        mfcc = self._mfcc_transform(input_vals)
        mfcc = mfcc * (10.0 / np.log(10))

        # Normalize per time frame if needed (axis=2 = time)
        if normalize:
            mean = mfcc.mean(dim=2, keepdim=True)
            std = mfcc.std(dim=2, keepdim=True) + 1e-8
            mfcc = (mfcc - mean) / std

        return mfcc

    def forward(self, input_vals, **kwargs):
        # Without altering the feature extraction.
        # with torch.no_grad():)
        # print('oooo friends')
        # print(self.time_to_mfcc(input_vals=input_vals, n_mfcc=self.config.num_mfccs).shape)
        input_vals_mfcc = self.time_to_mfcc(input_vals=input_vals, n_mfcc=self.config.num_mfccs).unsqueeze(1)
        # print(input_vals_mfcc.shape)
        # input('hi')

        out = self.ft_extractor(input_vals_mfcc)
        # print("")
        # print(out.shape)
        out=out.mean(dim=-1) 
        # print(out.shape)
        out=out.flatten(start_dim=1)
        # print(out.shape)
        # input('here')
        out = self.classifier(out)

        return out
