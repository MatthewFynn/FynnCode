import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from transformers import PreTrainedModel, PretrainedConfig

import os, sys

# Dynamically add OPERA repo root to sys.path
# Adjust this path if OPERA is located elsewhere on your machine
opera_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "OPERA"))
if opera_root not in sys.path:
    sys.path.insert(0, opera_root)
# from the OPERA repo you showed:
from src.benchmark.model_util import initialize_pretrained_model, get_encoder_path, SR

class OperaCEConfig(PretrainedConfig):
    """
    OPERA-CE (EfficientNet CNN) classifier, Wav2VecCNN-style.

    Args:
        num_classes (int)
        hidden_size (int or list[int]) : MLP head size(s)
        fs (int) : sample rate (must match SR=16000 used in OPERA)
        signal_len_t (float) : seconds per sample
        dropout (float) : dropout in classifier head
        num_cnn_layers (int|None) : keep first N backbone blocks if available
        mel_bins (int) : number of mel bands
        n_fft, hop_length : STFT params for mel
    """
    model_type = "operaCE_classifier"
    def __init__(
        self,
        num_classes=2,
        hidden_size=512,
        fs: int = 16000,
        signal_len_t: float = 8.0,
        dropout: float = 0.0,
        num_cnn_layers: int | None = None,
        mel_bins: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        **kwargs
    ):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.fs = fs
        self.signal_len_t = signal_len_t
        self.dropout = dropout
        self.num_cnn_layers = num_cnn_layers
        self.mel_bins = mel_bins
        self.n_fft = n_fft
        self.hop_length = hop_length
        # super().__init__(**kwargs)

class OperaCE(PreTrainedModel):
    config_class = OperaCEConfig

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config)
        self.criterion = nn.CrossEntropyLoss()

        # --- build & load OPERA-CE (EfficientNet) ---
        self.backbone = initialize_pretrained_model("operaCE")         # Cola(encoder="efficientnet")
        ckpt = torch.load(get_encoder_path("operaCE"), map_location="cpu")
        self.backbone.load_state_dict(ckpt["state_dict"], strict=False)
        self.backbone.eval()  # we’ll still fine-tune; eval avoids batchnorm updates in probes

        # print(self.backbone)
        # optional: truncate early blocks if the container is discoverable
        if self.config.num_cnn_layers > 0:
            out_ch = self._truncate_efficientnet_for_features(self.backbone, self.config.num_cnn_layers, gap=True)
        # input('ok')

        # head (dimension set after probing)
        self.dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity()
        enc_dim = self._get_cnn_size()
        self.classifier = self._build_classifier(enc_dim, self.config.hidden_size, self.config.num_classes)
        # print(enc_dim)
        # input('hi')
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- simple mel front-end (log-mel like OPERA utils) ----
    def _wave_to_mel(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) waveform @ fs==16000
        return: (B, 256, mel_bins) approx for ~8s if hop_length=512
        """
        # if self.config.fs != SR:
        #     raise ValueError(f"OPERA expects fs={SR}, got {self.config.fs}")
        B, T = x.shape
        # STFT
        win = torch.hann_window(self.config.n_fft, device=x.device)
        spec = torch.stft(x, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
                          window=win, return_complex=True)
        mag = (spec.real**2 + spec.imag**2).clamp_min(1e-10).sqrt()     # (B, F, frames)
        # mel filterbank
        fbank = torch.tensor(
            self._mel_filter(self.config.n_fft//2 + 1, self.config.mel_bins, SR),
            device=x.device, dtype=mag.dtype
        )  # (mel_bins, F)
        mel = torch.matmul(fbank, mag)                                  # (B, mel_bins, frames)
        mel = torch.log1p(mel)                                          # log-mel
        mel = mel.transpose(1, 2)                                       # (B, frames, mel_bins)
        return mel

    # minimal mel filter (triangular filters); fine for training starts
    def _mel_filter(self, n_freq: int, n_mels: int, sr: int):
        import numpy as np
        def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0**(m / 2595.0) - 1.0)
        f_min, f_max = 0.0, sr / 2
        mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
        hz = mel_to_hz(mels)
        bins = np.floor((self.config.n_fft + 1) * hz / sr).astype(int)
        fb = np.zeros((n_mels, n_freq), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
            if f_m_minus == f_m or f_m == f_m_plus: continue
            for k in range(f_m_minus, f_m):
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return fb

    # ---- backbone truncation (best-effort) ----
    def _truncate_backbone(self, model: nn.Module, n: int):
        # Try common containers used by EfficientNet wrappers
        for attr in ["features", "blocks", "layers", "stages"]:
            cont = getattr(model, attr, None)
            if isinstance(cont, (nn.Sequential, nn.ModuleList)):
                children = list(cont.children())
                if n < len(children):
                    new_cont = nn.Sequential(*children[:n]) if isinstance(cont, nn.Sequential) else nn.ModuleList(children[:n])
                    setattr(model, attr, new_cont)
                return
    def _truncate_efficientnet_for_features(self, backbone: nn.Module, n_blocks: int, *, gap: bool = True) -> int:
        """
        Truncate EfficientNet's MBConv blocks even when wrapped (e.g., backbone.efficientnet).
        Also monkey-patch backbone.extract_feature to stop after truncated blocks.
        Returns the output channels after the last kept block.
        """
        # 1) Find the EfficientNet module that actually holds _blocks
        eff = getattr(backbone, "efficientnet", None)
        if eff is None:
            # Fallback: search for a submodule that has `_blocks` as a ModuleList
            eff = None
            for _, m in backbone.named_modules():
                if hasattr(m, "_blocks") and isinstance(getattr(m, "_blocks"), nn.ModuleList):
                    eff = m
                    break
        if eff is None or not hasattr(eff, "_blocks") or not isinstance(eff._blocks, nn.ModuleList):
            raise AttributeError("Could not locate EfficientNet._blocks ModuleList on this backbone.")

        # 2) Truncate the blocks
        total = len(eff._blocks)
        if not (1 <= n_blocks <= total):
            raise ValueError(f"n_blocks={n_blocks} out of range (1..{total})")
        eff._blocks = nn.ModuleList(list(eff._blocks[:n_blocks]))

        last_out_ch = eff._blocks[-1]._project_conv.out_channels

        # 3) Build a safe extract_feature on the *wrapper* (backbone),
        #    honoring its front conv (cnn1) if present and skipping the head.
        def _extract_feature_override(x, dim=None, *args, **kwargs):
            # x is (B, frames, mel) from _wave_to_mel
            if x.dim() != 3:
                raise ValueError(f"Expected (B, frames, mel), got {tuple(x.shape)}")
            # -> (B, 1, mel, frames)
            x = x.permute(0, 2, 1).unsqueeze(1)

            # --- ensure channel count matches EfficientNet stem ---
            need_c = eff._conv_stem.in_channels  # typically 3
            if x.size(1) != need_c:
                used = False
                # Prefer a real 1->3 conv if present
                if hasattr(backbone, "cnn1") and isinstance(backbone.cnn1, nn.Conv2d):
                    x = backbone.cnn1(x); used = True
                elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "cnn1") \
                    and isinstance(backbone.encoder.cnn1, nn.Conv2d):
                    x = backbone.encoder.cnn1(x); used = True
                # Fallback: replicate channels
                if not used:
                    if x.size(1) == 1 and need_c == 3:
                        x = x.repeat(1, 3, 1, 1)
                    else:
                        raise RuntimeError(
                            f"Channel mismatch: got C={x.size(1)}, stem needs C={need_c} "
                            f"and no suitable 1->{need_c} conv was found."
                        )

            # --- EfficientNet stem + truncated blocks ---
            x = eff._conv_stem(x)
            x = eff._bn0(x)
            x = eff._swish(x)

            for b in eff._blocks:
                x = b(x)

            if gap:
                x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, C)
            return x

        backbone.extract_feature = _extract_feature_override  # monkey-patch on wrapper
        return last_out_ch

    # ---- probed dim & head ----
    def _build_classifier(self, in_dim, hidden, num_classes):
        if isinstance(hidden, int):
            return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), self.dropout, nn.Linear(hidden, num_classes))
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), self.dropout]
            d = h
        layers.append(nn.Linear(d, num_classes))
        return nn.Sequential(*layers)

    def _get_cnn_size(self):
        T = int(self.config.signal_len_t * self.config.fs)
        with torch.no_grad():
            z = self._encode(torch.zeros(2, T))
            if z.dim() == 3: z = z.mean(dim=1)
        return z.shape[-1]

    # ---- encode via OPERA-CE's interface ----
    def _encode(self, wav: torch.Tensor):
        mel = self._wave_to_mel(wav)                    # (B, frames, mel)
        # OPERA extract_feature expects (B, 256, 64) style input; ensure float
        feats = self.backbone.extract_feature(mel.float(), dim=1280)  # dim arg is used inside OPERA
        if isinstance(feats, dict):
            for k in ("last_hidden_state", "embeddings", "feat", "x"):
                if k in feats: feats = feats[k]; break
        return feats  # (B, D) or (B, T', D)

    # ---- public API ----
    def extract_features(self, input_vals: torch.Tensor):
        if input_vals.dim() != 2:
            raise ValueError(f"Expected [batch, time]; got {tuple(input_vals.shape)}")
        z = self._encode(input_vals)
        return z.mean(dim=1) if z.dim() == 3 else z

    def forward(self, input_vals: torch.Tensor, **kwargs):
        if input_vals.dim() != 2:
            raise ValueError(f"Expected [batch, time]; got {tuple(input_vals.shape)}")
        # print(input_vals.shape)
        # input('ok')
        z = self.extract_features(input_vals)
        return self.classifier(z)
