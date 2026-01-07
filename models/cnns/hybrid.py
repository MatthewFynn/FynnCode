"""
    wav2vec_cnn.py
    Author: Matthew Fynn

    Purpose: Combined a LF and HF hybrid model
"""

import torch
import torch.nn as nn
import os

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)

from models.cnns.wav2vec2_cnn import Wav2VecCNNConfig,Wav2VecCNN
from models.cnns.encodec_cnn import EnCodecConfig, EnCodecCNN
from models.cnns.UNET_SSL_cnn import UNET_SSL_Config, UNET_SSL_CNN
from models.cnns.UNET_SSL_cnn_2D import UNET_SSL_Config2D, UNET_SSL_CNN2D
from models.cnns.hubert_cnn import HuBERTCNNConfig, HuBERTCNN
from models.cnns.opera_ce import OperaCEConfig, OperaCE


class HybridConfig(PretrainedConfig):

    model_type = "hybrid_classfier"

    def __init__(
            self,
            num_classes=2,
            hidden_size = 512,
            LF_dir: str = 'LFmod/verx',
            HF_dir: str = 'HFmod/verx',
            channel = 8,
            fold = 7,
            dropout=0,
            **kwargs
    ):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.channel = channel
        self.fold = fold
        self.LF_dir = LF_dir
        self.HF_dir = HF_dir
        self.dropout = dropout
        
        
class HybridModel(PreTrainedModel):

    config_class = HybridConfig

    def __init__(self, config: PretrainedConfig, **kwargs):
        super(HybridModel, self).__init__(config)

        self.criterion = nn.CrossEntropyLoss()

        self.LF_dir = os.path.join('/home/tickingheart/dev/code',self.config.LF_dir,f'ch{self.config.channel}',f'fold{self.config.fold}')
        self.HF_dir = os.path.join('/home/tickingheart/dev/code',self.config.HF_dir,f'ch{self.config.channel}',f'fold{self.config.fold}')
        print("")
        print(self.LF_dir)
        print(self.HF_dir)
        
        self.ssl_code = self.LF_dir.split('/')[-5] 
        if self.ssl_code == "unetssl2d_cnn":
            self.LF_mod = UNET_SSL_CNN2D.from_pretrained(self.LF_dir)
        elif self.ssl_code == "unetssl_cnn":
            self.LF_mod = UNET_SSL_CNN.from_pretrained(self.LF_dir)
        
        self.mod_code = self.HF_dir.split('/')[-5] 
        if self.mod_code == "encodec_cnn":
            self.HF_mod = EnCodecCNN.from_pretrained(self.HF_dir)
        elif self.mod_code == "wav2vec_cnn":
            self.HF_mod = Wav2VecCNN.from_pretrained(self.HF_dir)
        elif self.mod_code == "hubert_cnn":
            self.HF_mod = HuBERTCNN.from_pretrained(self.HF_dir)
        elif self.mod_code == "opera_ce":
            self.HF_mod = OperaCE.from_pretrained(self.HF_dir)


        # print(self.HF_mod.config.num_cnn_layers)
        # print(self.HF_mod.config)
        # print(next(self.HF_mod.parameters()))
        # lstm_module = self.HF_mod.ft_extractor.layers[13].lstm
        # print(lstm_module.weight_ih_l0)

        # input('ok')
        self.out_size = self._get_cnn_size()

        if type(self.config.hidden_size) == int:
            self.classifier = nn.Sequential(
                nn.Linear(self.out_size, self.config.hidden_size), 
                nn.GELU(),
                nn.Linear(self.config.hidden_size, self.config.num_classes)
            )
        else:
            self.classifier = self.build_classifier()
        
        
    def _get_cnn_size(self):
        input_ten1 = torch.zeros(6,1,int(self.LF_mod.config.signal_len_t*self.LF_mod.config.fs),1)

        input_ten2 = torch.zeros(6,1,self.HF_mod.config.signal_len,1) if self.mod_code=="encodec_cnn" else torch.zeros(6,1,int(self.HF_mod.config.signal_len_t*self.HF_mod.config.fs),1)
        out_ten = self.ft_extracter_plus_Layer(input_ten1, input_ten2)
        return out_ten.shape[-1]
    
    def build_classifier(self):
        layers = list()
        in_features = self._get_cnn_size()

        for layer_size in self.config.hidden_size:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.GELU())
            if self.config.dropout != 0:
                layers.append(nn.Dropout(self.config.dropout))
            in_features = layer_size

        layers.append(nn.Linear(in_features, self.config.num_classes))
        return nn.Sequential(*layers)


    def ft_extracter_plus_Layer(self, LF_vals, HF_vals):
        # (Batch, N, T, C)
        # Averaging across N dimension after feature extraction
        #FIRST WE EXTRACT AND AVERAGE (IF >1 SEG) THE LF FEATURES
        # print(LF_vals.shape, HF_vals.shape)
        # input('hi')
        LF_feat_all = []
        for i in range(LF_vals.shape[1]):
            vals = LF_vals[:,i,:,0] #FIXME need to make sure this works for the multichannel case (extract_features function)
            # print(vals.shape)
            if self.ssl_code == "unetssl2d_cnn":
                vals_mfcc = self.LF_mod.time_to_mfcc(input_vals=vals, n_mfcc=self.LF_mod.num_mfccs).unsqueeze(1)
                with torch.no_grad():
                    x = self.LF_mod.ft_extractor(vals_mfcc).flatten(start_dim=1)
                # print(x.shape)
                x = self.LF_mod.classifier[0](x)   # Linear
                # print(x.shape)

                LF_feat = self.LF_mod.classifier[1](x)  # GELU output
                # print(LF_feat.shape)
                # input('here')

            elif self.ssl_code == "unetssl_cnn":
                with torch.no_grad():
                    x = self.LF_mod.ft_extractor(vals.unsqueeze(1)).mean(dim=2)
                # print(x.shape)
                x = self.LF_mod.classifier[0](x)   # Linear
                # print(x.shape)
                
                LF_feat = self.LF_mod.classifier[1](x)  # GELU output
                # print(LF_feat.shape)
                # input('hi')

            # print(LF_feat)
            LF_feat_all.append(LF_feat)
        LF_feat_avg = torch.mean(torch.stack(LF_feat_all, dim=0),dim=0)
        # print(LF_feat_avg)

        HF_feat_all = []
        for i in range(HF_vals.shape[1]):
            vals = HF_vals[:,i,:,0]
            # print(vals.shape)
            if self.mod_code == "encodec_cnn":
                with torch.no_grad():
                    x = self.HF_mod.ft_extractor(vals.unsqueeze(1)).mean(dim=2)
                # print(x.shape)
                x = self.HF_mod.classifier[0](x)   # Linear
                # print(x.shape)
                HF_feat = self.HF_mod.classifier[1](x)  # GELU output
                # print(HF_feat.shape)
                # input('ok') 
                
            elif self.mod_code == "wav2vec_cnn":
                with torch.no_grad():
                    x = self.HF_mod.ft_extractor(vals).mean(dim=2)
                
                x = self.HF_mod.classifier[0](x)   # Linear
                HF_feat = self.HF_mod.classifier[1](x)  # GELU output
            
            elif self.mod_code == "hubert_cnn":
                with torch.no_grad():
                    x = self.HF_mod.ft_extractor(vals).mean(dim=2)
                
                x = self.HF_mod.classifier[0](x) #Linear
                HF_feat=self.HF_mod.classifier[1](x) #GELU output

            elif self.mod_code == "opera_ce":
                with torch.no_grad():
                    x = self.HF_mod.extract_features(vals)
                # print(x.shape)
                x = self.HF_mod.classifier[0](x) #Linear
                HF_feat=self.HF_mod.classifier[1](x) #GELU output
                # print(HF_feat.shape)

                # input('ok') 

            
            HF_feat_all.append(HF_feat)
        HF_feat_avg = torch.mean(torch.stack(HF_feat_all, dim=0),dim=0)

        # print(LF_feat_avg.shape)

        concat_feat = torch.cat([LF_feat_avg, HF_feat_avg], dim=1)
        return concat_feat

    def forward_hybrid(self, LF_vals, HF_vals):

        concat_feat = self.ft_extracter_plus_Layer(LF_vals, HF_vals)
        # print(concat_feat.shape)
        # print(self.out_size)
        
        out = self.classifier(concat_feat)
        # print(out.shape)
        # input('fwd')


        return out
    
    def forward(self, LF_vals, HF_vals):
        return self.forward_hybrid(LF_vals, HF_vals)