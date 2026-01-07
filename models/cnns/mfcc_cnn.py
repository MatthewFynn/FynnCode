"""
    wav2vec_cnn.py
    Author: Milan Marocchi

    Purpose: Contains Wav2Vec 2.0 based classifier and model.
"""

import numpy as np
import torch
import torch.nn as nn
import librosa


from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)


class mfccCNNConfig(PretrainedConfig):
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
        hidden_size_len: int = 512,
        fs: int = 2000,
        signal_len_t: int = 2, #seconds
        dropout: float = 0.2,
        frame_length: float = 0.04,
        overlap: float = 0.5,
        num_mfccs: int = 13,
        mlp_flag:int =0,
        ks: int = 16,
        **kwargs
    ):
        self.num_classes = num_classes
        self.mlp_flag = mlp_flag
        self.hidden_size = [hidden_size_len,256] if self.mlp_flag ==1 else hidden_size_len
        self.signal_len_t = signal_len_t
        # self.signal_len = int(signal_len_t*fs)
        self.fs = fs
        self.dropout = dropout
        self.frame_length = frame_length
        self.n_fft = int(self.frame_length*self.fs)
        self.hop_length = int(int(self.frame_length*self.fs) * (1 - overlap))
        self.num_mfccs = num_mfccs
        self.ks=ks

class mfccCNN(PreTrainedModel):

    config_class = mfccCNNConfig

    def __init__(self, config: PretrainedConfig, **kwargs): 
        super(mfccCNN, self).__init__(config)

        self.criterion = nn.CrossEntropyLoss()
        # self.ft_extractor = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, stride=1),
        #     nn.Dropout1d(p = self.config.dropout),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=4, stride=2),
        #     nn.Conv1d(in_channels=16, out_channels=8, kernel_size=16, stride=1),
        #     nn.Dropout1d(p = self.config.dropout),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=4, stride=2),
        #     nn.Conv1d(in_channels=8, out_channels=8, kernel_size=16, stride=1),
        #     nn.Dropout1d(p = self.config.dropout),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=4, stride=2),
        # )
        self.ft_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=self.config.ks, stride=1),
            nn.Dropout1d(p = self.config.dropout),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=self.config.ks, stride=1),
            nn.Dropout1d(p = self.config.dropout),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=self.config.ks, stride=1),
            nn.Dropout1d(p = self.config.dropout),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=self.config.ks, stride=1),
            nn.Dropout1d(p = self.config.dropout),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
        )
        # self.ft_extractor = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=16, kernel_size=self.config.ks, stride=1),
        #     nn.Dropout1d(p = self.config.dropout),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=4, stride=2),
        #     nn.Conv1d(in_channels=16, out_channels=16, kernel_size=self.config.ks, stride=1),
        #     nn.Dropout1d(p = self.config.dropout),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=4, stride=2),
        #     nn.Conv1d(in_channels=16, out_channels=16, kernel_size=self.config.ks, stride=1),
        #     nn.Dropout1d(p = self.config.dropout),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=4, stride=2),
        #     nn.Conv1d(in_channels=16, out_channels=8, kernel_size=self.config.ks, stride=1),
        #     nn.Dropout1d(p = self.config.dropout),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=4, stride=2),
        #     nn.Conv1d(in_channels=8, out_channels=8, kernel_size=self.config.ks, stride=1),
        #     nn.Dropout1d(p = self.config.dropout),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=4, stride=2),
        # )

        self.cnn_out_size = self._get_cnn_size()
        self.fs = self.config.fs
        if type(self.config.hidden_size) == int:
            self.classifier = nn.Sequential(
                nn.Linear(self.cnn_out_size, self.config.hidden_size), 
                nn.GELU(),
                nn.Linear(self.config.hidden_size, self.config.num_classes)
            )
        else:
            self.classifier = self.build_classifier()

        self.num_classes = config.num_classes

    def _get_cnn_size(self):
        # print(self.config.fs)
        # print(self.config.signal_len_t)
        # input()
        input_tensor = self.time_to_mfcc(torch.rand(64, int(self.config.signal_len_t*self.config.fs)), self.config.num_mfccs).unsqueeze(1) #will have to change to incorporate mfccs
        output_tensor = self.ft_extractor(input_tensor).flatten(start_dim=1)

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
    
    
    def time_to_mfcc(self, input_vals, n_mfcc):
        processed_batch = []
        for signal in input_vals:
            signal = signal.cpu().numpy()
            mfcc = librosa.feature.mfcc(
                    y=signal, 
                    sr = self.config.fs, 
                    fmax = self.config.fs/2, #ORIGINAL
                    # fmax = 1000, 
                    n_mfcc=n_mfcc, 
                    # n_mels = 21, #ORIGINAL - 73% acc
                    n_mels = 21,
                    # n_mels = 15,
                    n_fft = self.config.n_fft,
                    hop_length = self.config.hop_length)
            flattened_mfcc = mfcc.T.flatten()
            processed_batch.append(flattened_mfcc)
        return torch.tensor(np.array(processed_batch)).to(input_vals.device)

    def forward(self, input_vals, **kwargs):
        # Without altering the feature extraction.
        # with torch.no_grad():)
        
        input_vals_mfcc = self.time_to_mfcc(input_vals=input_vals, n_mfcc=self.config.num_mfccs).unsqueeze(1)
        out = self.ft_extractor(input_vals_mfcc).flatten(start_dim=1)
        out = self.classifier(out)

        return out
