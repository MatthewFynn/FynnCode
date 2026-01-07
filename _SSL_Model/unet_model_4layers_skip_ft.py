import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class UNETConfig(PretrainedConfig):
    model_type = "unet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # You can add model configuration parameters here if needed

class UNet1d(PreTrainedModel):
    config_class = UNETConfig

    def __init__(self, config):
        super().__init__(config)

        # Feature extractor (encoder)
        self.feature_extractor = nn.ModuleDict({
            'enc1': nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=99, padding=49, bias = False),
                nn.GroupNorm(num_groups=64, num_channels=64, eps = 1e-05, affine=True), #LayerNorm as 1 group
                nn.GELU(),
            ),
            'pool1': nn.MaxPool1d(2),
            'enc2': nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=49, padding=24, bias = False),
                nn.GELU(),
            ),
            'pool2': nn.MaxPool1d(2),
            'enc3': nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=25, padding=12, bias = False),
                nn.GELU(),
            ),
            'pool3': nn.MaxPool1d(2),
            'enc4': nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=11, padding=5, bias = False),
                nn.GELU(),
            ),
        })

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=25, padding=12),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=49, padding=24),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=99, padding=49),
            nn.ReLU(),
        )

        self.conv_last = nn.Conv1d(64, 1, kernel_size=1)

        # Initialize weights
        self.post_init()

    def forward(self, x, **kwargs):
        # Encoder
        enc1 = self.feature_extractor['enc1'](x)
        enc1_pool = self.feature_extractor['pool1'](enc1)

        enc2 = self.feature_extractor['enc2'](enc1_pool)
        enc2_pool = self.feature_extractor['pool2'](enc2)

        enc3 = self.feature_extractor['enc3'](enc2_pool)
        enc3_pool = self.feature_extractor['pool3'](enc3)

        enc4 = self.feature_extractor['enc4'](enc3_pool)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder with skip connections
        dec3 = self.up3(bottleneck) + enc3
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3) + enc2
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = self.dec1(dec1)

        out = self.conv_last(dec1)

        return out, enc4


class UNetSSLFeatureExtractor_2L(nn.Module):
    def __init__(self, feature_extractor):
        super(UNetSSLFeatureExtractor_2L, self).__init__()
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
    
class UNetSSLFeatureExtractor_3L(nn.Module):
    def __init__(self, feature_extractor):
        super(UNetSSLFeatureExtractor_3L, self).__init__()
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
    
class UNetSSLFeatureExtractor_4L(nn.Module):
    def __init__(self, feature_extractor):
        super(UNetSSLFeatureExtractor_4L, self).__init__()
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
