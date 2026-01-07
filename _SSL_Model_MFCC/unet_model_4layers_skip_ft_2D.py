import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np
import librosa
import torchaudio
import torchaudio.functional as AF
import math

class UNETConfig2D(PretrainedConfig):
    model_type = "unet"

    def __init__(
        self, 
        fs = 2000,
        frame_length: float = 0.04,
        overlap: float = 0.5,
        num_mfccs: int = 20,
        dropout = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fs = fs
        self.frame_length = frame_length
        self.n_fft = int(self.frame_length*self.fs)
        self.hop_length = int(int(self.frame_length*self.fs) * (1 - overlap))
        self.num_mfccs = num_mfccs
        self.dropout = dropout


class UNet2D(PreTrainedModel):
    config_class = UNETConfig2D

    def __init__(self, config):
        super().__init__(config)

        self.feature_extractor = nn.ModuleDict({
            'enc1': nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 8), stride=(1, 1), padding=1),
                nn.Dropout2d(p=config.dropout),
                nn.GELU(),
            ),
            'pool1': nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),

            'enc2': nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 8), stride=(1, 1), padding=1),
                nn.Dropout2d(p=config.dropout),
                nn.GELU(),
            ),
            'pool2': nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            'enc3': nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.GELU(),
            ),
            'pool3': nn.MaxPool2d(kernel_size=(2, 2), stride = (1,1)),

            'enc4': nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=2, padding=1),
                nn.GELU(),
            ),
        })

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(4,3), padding=1),
            nn.GELU(), #FIXME - look into this as was originally ReLU
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 64, kernel_size=(2,2), stride=(1,1), output_padding=0)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=(2,6), stride=2, output_padding=(0, 1))
        self.dec2 = nn.Sequential(
            nn.Conv2d(32+32, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 6), stride=(1, 2), output_padding=0)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.conv_last = nn.Conv2d(16, 1, kernel_size=1)

        object.__setattr__(self, "_mfcc_transform", torchaudio.transforms.MFCC(
            sample_rate=config.fs,
            n_mfcc=config.num_mfccs,
            melkwargs={
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "n_mels": 21,
                "f_min": 0.0,
                "f_max": config.fs / 2,
                "power": 2.0,
                "mel_scale": "slaney",
                "norm": "slaney",
                "center": True,          # match librosa
                "pad_mode": "reflect",   # match librosa
                "window_fn": torch.hann_window,
            },
            dct_type=2,
            norm="ortho",
            log_mels=True,              # log before DCT (like librosa)
        ))
    


    def forward(self, x):
        # Encoder
        enc1 = self.feature_extractor['enc1'](x)
        enc1p = self.feature_extractor['pool1'](enc1)

        enc2 = self.feature_extractor['enc2'](enc1p)
        enc2p = self.feature_extractor['pool2'](enc2)

        enc3 = self.feature_extractor['enc3'](enc2p)
        enc3p = self.feature_extractor['pool3'](enc3)

        enc4 = self.feature_extractor['enc4'](enc3p)
        bottleneck = self.bottleneck(enc4)

        # Decoder with concatenation
        up3_out = self.up3(bottleneck)
        enc3_interp = F.interpolate(enc3, size=up3_out.shape[2:], mode='nearest')
        dec3 = torch.cat([up3_out, enc3_interp], dim=1)
        dec3 = self.dec3(dec3)

        up2_out = self.up2(dec3)
        enc2_interp = F.interpolate(enc2, size=up2_out.shape[2:], mode='nearest')
        dec2 = torch.cat([up2_out, enc2_interp], dim=1)
        dec2 = self.dec2(dec2)

        up1_out = self.up1(dec2)
        # Uncomment if you want to concat enc1 as well
        # enc1_interp = F.interpolate(enc1, size=up1_out.shape[2:], mode='nearest')
        # up1_out = torch.cat([up1_out, enc1_interp], dim=1)
        dec1 = self.dec1(up1_out)

        out = self.conv_last(dec1)
        out = out[:, :, :20, :201]

        return out, enc4
    
    
    def time_to_mfcc(self, input_vals, n_mfcc, normalize=True):
        # ensure [B, T]
        if input_vals.dim() == 1:
            input_vals = input_vals.unsqueeze(0)
        x = input_vals.to(dtype=torch.float32)

        # move transform to the right device (no reassignment!)
        self._mfcc_transform.to(x.device)

        # MFCC: [B, n_mfcc, frames]
        mfcc = self._mfcc_transform(x)

        # Convert ln -> 10*log10 to match librosa's power_to_db
        mfcc = mfcc * (10.0 / math.log(10.0))

        if normalize:
            # PER-FRAME normalization (match your librosa path)
            # mean/std over coefficients (dim=1), for each time frame
            mean = mfcc.mean(dim=1, keepdim=True)
            std  = mfcc.std(dim=1, keepdim=True).clamp_min(1e-8)
            mfcc = (mfcc - mean) / std

        return mfcc.contiguous()
    
    def time_to_mfcc_cpu(self, input_vals, n_mfcc, normalize = True):
        processed_batch = []
        for signal in input_vals:
            signal = signal.cpu().numpy()
            mfcc = librosa.feature.mfcc(
                    y=signal, 
                    sr = self.config.fs, 
                    fmax = self.config.fs/2, #ORIGINAL
                    n_mfcc=n_mfcc, 
                    n_mels = 21,
                    n_fft = self.config.n_fft,
                    hop_length = self.config.hop_length)
            if normalize == True:
                mfcc = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)
                
            processed_batch.append(mfcc)
        return torch.tensor(np.array(processed_batch)).to(input_vals.device)



class UNetSSLFeatureExtractor2D(nn.Module):
    def __init__(self, feature_extractor):
        super(UNetSSLFeatureExtractor2D, self).__init__()
        self.feature_extractor = nn.ModuleDict({
            'enc1': feature_extractor['enc1'],
            'pool1': feature_extractor['pool1'],
            'enc2': feature_extractor['enc2'],
            'pool2': feature_extractor['pool2'],
        })

    def forward(self, x):
        output = x
        for name, layer in self.feature_extractor.items():
            output = layer(output)
        return output
    
class UNetSSLFeatureExtractor2D_3L(nn.Module):
    def __init__(self, feature_extractor):
        super(UNetSSLFeatureExtractor2D_3L, self).__init__()
        self.feature_extractor = nn.ModuleDict({
            'enc1': feature_extractor['enc1'],
            'pool1': feature_extractor['pool1'],
            'enc2': feature_extractor['enc2'],
            'pool2': feature_extractor['pool2'],
            'enc3': feature_extractor['enc3'],
            'pool3': feature_extractor['pool3'],
        })

    def forward(self, x):
        output = x
        for name, layer in self.feature_extractor.items():
            output = layer(output)
        return output
    
class UNetSSLFeatureExtractor2D_4L(nn.Module):
    def __init__(self, feature_extractor):
        super(UNetSSLFeatureExtractor2D_4L, self).__init__()
        self.feature_extractor = nn.ModuleDict({
            'enc1': feature_extractor['enc1'],
            'pool1': feature_extractor['pool1'],
            'enc2': feature_extractor['enc2'],
            'pool2': feature_extractor['pool2'],
            'enc3': feature_extractor['enc3'],
            'pool3': feature_extractor['pool3'],
            'enc4': feature_extractor['enc4'],
        })

    def forward(self, x):
        output = x
        for name, layer in self.feature_extractor.items():
            output = layer(output)
        return output



