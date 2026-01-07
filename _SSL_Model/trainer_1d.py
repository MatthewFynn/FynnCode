#!/usr/bin/env python3
import os
import random
import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet_model_4layers_skip_ft import UNet1d, UNETConfig
from dataset import FeatureVectorsDataset
from preprocess_training_data import get_wav_files_to_dataframe
from safetensors.torch import load_file

import torchaudio
from torchaudio.transforms import MelSpectrogram
import pandas as pd


# ------------------------- helpers -------------------------

def set_seed(seed: int = 101):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True   # good for 1D convs
    torch.backends.cudnn.deterministic = False


def power_to_db(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # librosa-like 10*log10 for power spectrograms
    return 10.0 * torch.log10(x.clamp_min(eps))


def build_mel_transform(fs: int, n_fft: int, hop: int, n_mels: int) -> MelSpectrogram:
    return MelSpectrogram(
        sample_rate=fs,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        f_min=0.0,
        f_max=fs / 2,
        power=2.0,
        center=True,
        pad_mode="reflect",
        window_fn=torch.hann_window,
        mel_scale="slaney",
        norm="slaney",
    )


def masked_mse(pred: torch.Tensor, target: torch.Tensor, time_mask_keep: torch.Tensor, eps: float = 1e-8):
    """
    pred/target/mask: [B,1,T], mask=1 keep, 0 masked.
    Return MSE over masked region only (where mask == 0).
    """
    inv = (1.0 - time_mask_keep)
    num = (inv * (pred - target) ** 2).sum()
    den = inv.sum().clamp_min(eps)
    return num / den


def mel_spectral_l1_masked(pred_wav: torch.Tensor,
                           target_wav: torch.Tensor,
                           time_mask_keep: torch.Tensor,
                           mel_tfm: MelSpectrogram,
                           n_fft: int,
                           hop: int,
                           eps: float = 1e-8):
    """
    pred_wav/target_wav/time_mask_keep: [B,1,T]
    Convert to log-Mel and compute L1 ONLY over Mel frames that correspond to masked time windows.
    """
    B, _, T = pred_wav.shape
    pred_mel   = mel_tfm(pred_wav.squeeze(1))    # [B,F,TMel]
    target_mel = mel_tfm(target_wav.squeeze(1))  # [B,F,TMel]
    pred_mel   = power_to_db(pred_mel).unsqueeze(1)     # [B,1,F,TMel]
    target_mel = power_to_db(target_mel).unsqueeze(1)

    # map time mask -> Mel frame mask
    pad = n_fft // 2
    Tmel = pred_mel.shape[-1]
    mel_mask_keep = torch.ones_like(pred_mel)  # 1 keep, 0 masked (in Mel domain)
    tm = time_mask_keep[:, 0]                  # [B,T]

    for i in range(B):
        tvec = tm[i]
        for j in range(Tmel):
            center = int(round(j * hop))
            st = max(0, center - pad)
            en = min(T, center + pad)
            if en > st and tvec[st:en].float().mean() < 0.25:
                mel_mask_keep[i, 0, :, j] = 0.0

    inv = (1.0 - mel_mask_keep)
    num = (inv * (pred_mel - target_mel).abs()).sum()
    den = inv.sum().clamp_min(eps)
    return num / den


def cosine_encoder_loss_1d(
    emb_masked: torch.Tensor,       # [B,C,Te]
    emb_unmasked: torch.Tensor,     # [B,C,Te]
    time_mask_keep: torch.Tensor,   # [B,1,T]  (1=keep, 0=masked)
    T_recon: int,                   # 8000
    Te: int,                        # 1000
    enc_unmasked_weight: float = 0.05,
):
    """
    Downsample the time mask to Te using stride-based pooling that matches the
    encoder's effective stride (~T/Te). Then compute mean(1-cos) over masked Te
    positions + enc_unmasked_weight * mean(1-cos) over unmasked Te positions.
    """
    # Expected stride (round to nearest int)
    S = max(1, int(round(T_recon / Te)))   # ~8 for 8000->1000

    # Convert keep->masked for masked selection
    masked_time = 1.0 - time_mask_keep.float()        # [B,1,T]
    keep_time   = time_mask_keep.float()              # [B,1,T]

    # Use pooling with stride S to align with embedding frames.
    # ceil_mode handles tails when T is not an exact multiple of S.
    masked_down = F.max_pool1d(masked_time, kernel_size=S, stride=S, ceil_mode=True)  # [B,1,~Te]
    keep_down   = F.max_pool1d(keep_time,   kernel_size=S, stride=S, ceil_mode=True)  # [B,1,~Te]

    # Crop/pad to exactly Te (safe for small off-by-ones)
    if masked_down.shape[-1] < Te:
        # pad right if needed
        pad_amt = Te - masked_down.shape[-1]
        masked_down = F.pad(masked_down, (0, pad_amt))
        keep_down   = F.pad(keep_down,   (0, pad_amt))
    elif masked_down.shape[-1] > Te:
        masked_down = masked_down[..., :Te]
        keep_down   = keep_down[..., :Te]

    sel_masked = masked_down.squeeze(1) > 0.5   # [B,Te]
    sel_keep   = keep_down.squeeze(1)   > 0.5   # [B,Te]

    # Cosine(emb_masked vs emb_unmasked) along channels
    em1 = F.normalize(emb_masked,   dim=1, eps=1e-8)
    em2 = F.normalize(emb_unmasked, dim=1, eps=1e-8)
    cos_t = torch.sum(em1 * em2, dim=1)         # [B,Te]
    one_minus_cos = 1.0 - cos_t

    has_m = sel_masked.any()
    has_u = sel_keep.any()

    enc_m = one_minus_cos[sel_masked].mean() if has_m else None
    enc_u = one_minus_cos[sel_keep].mean()   if has_u else None

    if has_m:
        enc_loss = enc_m
        if has_u and enc_unmasked_weight > 0:
            enc_loss = enc_loss + enc_unmasked_weight * enc_u
    else:
        enc_loss = torch.tensor(0.0, device=emb_masked.device)

    return enc_loss, (enc_m if has_m else None), (enc_u if has_u else None)


# ------------------------- CLI -------------------------

@click.group(context_settings={'show_default': True})
def cli(**kwargs):
    pass


@cli.command()
@click.option('--batch_size', '-b',  type=int, required=True, help="batch size")
@click.option('--learning_rate', '-lr', type=float, required=True, help="learning rate")
@click.option('--epochs', '-e', type=int, default=200, show_default=True)
@click.option('--pcg', '-p', type = int, required=True, help="percentage of pcg")
@click.option('--save_dir', '-s', required=True, help="save dir (checkpoints + tb logs)")

# mel loss config
@click.option('--fs', type=int, default=2000, show_default=True)
@click.option('--n_fft', type=int, default=400, show_default=True)
@click.option('--hop', type=int, default=160, show_default=True)
@click.option('--n_mels', type=int, default=64, show_default=True)
# weights
@click.option('--w_mel', type=float, default=2.0, show_default=True, help="weight for Mel L1 loss")
@click.option('--w_unmasked', type=float, default=0.05, show_default=True, help="MSE weight on unmasked region")
@click.option('--alpha', type=float, default=10.0, show_default=True, help="encoder (cosine) loss weight")
@click.option('--enc_unmasked_weight', type=float, default=0.05, show_default=True,
              help="weight on encoder-loss for unmasked selection")
@click.option('--beta', type=float, default=2.0, show_default=True, help="weight for decoder MSE loss")

def cli(batch_size, learning_rate, epochs, save_dir, pcg, 
        fs, n_fft, hop, n_mels,
        w_mel, w_unmasked, alpha, beta, enc_unmasked_weight):

    # set_seed(101)
    TRAIN_DIR = f'/home/{os.getlogin()}/Desktop/SSL_data/Ker2018_4s'
    TRAIN_DIR_PCG = f'/home/{os.getlogin()}/Desktop/SSL_data/pcg_ssl_training_data'
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu') ##########################
    print(device)
    # Load data
    wav_files_train_o = get_wav_files_to_dataframe(TRAIN_DIR)['File_Path'].sort_values()
    wav_files_train_pcg = get_wav_files_to_dataframe(TRAIN_DIR_PCG)['File_Path']

    if pcg > 0:
        target_length = int(len(wav_files_train_pcg)*100/pcg-len(wav_files_train_pcg))
        # print(target_length)
        # input('a')
        indices = np.linspace(0, len(wav_files_train_o)-1, target_length).astype(int)
        # print(indices[0:50])
        # input()
        wav_files_train_o = wav_files_train_o.iloc[indices]
        wav_files_train = pd.concat([wav_files_train_o,wav_files_train_pcg],ignore_index=True)
    else:
        wav_files_train = wav_files_train_o

    print(len(wav_files_train))
    print(len(wav_files_train_o))
    print(len(wav_files_train_pcg))
    # input('hi')
    train_data = FeatureVectorsDataset(wav_files_train) #FIXME
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # ---------------- model & opt ----------------
    config = UNETConfig.from_pretrained('/home/matthew-fynn/Desktop/DL/_SSL_Model/trained_SSL_model_pcg2_1D_a10b2m2/ep0')
    model = UNet1d(config).to(device)
    state_dict = load_file('/home/matthew-fynn/Desktop/DL/_SSL_Model/trained_SSL_model_pcg2_1D_a10b2m2/ep0/model.safetensors')
    # Filter out keys with size mismatch
    model_keys = dict(model.named_parameters())
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_keys and v.shape == model_keys[k].shape
    }

    model = UNet1d(UNETConfig()).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---------------- mel transform ----------------
    mel_tfm = build_mel_transform(fs, n_fft, hop, n_mels).to(device)

    # ---------------- logging ----------------
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
    model.save_pretrained(os.path.join(save_dir, "ep0"))

    train_total, train_mse, train_mel, train_enc = [], [], [], []
    warned_no_emb = False

    # ---------------- loop ----------------
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        model.train()
        tot, tot_mse, tot_mel, tot_enc = 0.0, 0.0, 0.0, 0.0

        # number of masks and length
        num_masks, min_mask_len, max_mask_len = 4, 200, 500

        for batch in train_loader:
            wav = batch.unsqueeze(1).to(device)  # [B,1,T]
            B, _, T = wav.shape

            # initialize time_mask_keep each batch
            time_mask_keep = torch.ones((B, 1, T), device=device)

            # apply per-sample random masking
            for i in range(B):
                for _ in range(num_masks):
                    # avoid overlapping masks if possible
                    L = random.randint(min_mask_len, max_mask_len)
                    start = random.randint(0, max(1, T - L))
                    end = start + L
                    time_mask_keep[i, :, start:end] = 0

            wav_masked = wav * time_mask_keep

            # forward: expect recon + embedding
            out = model(wav_masked)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                recon, emb_masked = out
                # run unmasked pass for encoder loss
                recon_u, emb_unmasked = model(wav)  # lightweight extra fwd for embeddings alignment
            else:
                recon = out
                emb_masked = emb_unmasked = None
                if (not warned_no_emb) and alpha > 0.0:
                    print("WARNING: UNet1d did not return embeddings; encoder loss disabled (set --alpha 0 to silence).")
                    warned_no_emb = True

            #  reconstruction MSE
            loss_mse_masked = masked_mse(recon, wav, time_mask_keep)
            if w_unmasked > 0:
                keep = time_mask_keep
                num = (keep * (recon - wav) ** 2).sum()
                den = keep.sum().clamp_min(1e-8)
                loss_mse_unmasked = num / den
                loss_mse = loss_mse_masked + w_unmasked * loss_mse_unmasked
            else:
                loss_mse = loss_mse_masked

            # mel spectral L1 on masked mel frames 
            loss_mel = mel_spectral_l1_masked(recon, wav, time_mask_keep, mel_tfm, n_fft, hop)

            # encoder cosine loss (masked + unmasked)
            if (emb_masked is not None) and (emb_unmasked is not None) and alpha > 0.0:
                Te = emb_masked.shape[-1]   # 1000
                T  = recon.shape[-1]        # 8000
                enc_loss, enc_m, enc_u = cosine_encoder_loss_1d(
                    emb_masked, emb_unmasked, time_mask_keep,
                    T_recon=T, Te=Te, enc_unmasked_weight=enc_unmasked_weight
                )
            else:
                enc_loss = torch.tensor(0.0, device=wav.device)
            # total loss
            loss = beta * loss_mse + w_mel * loss_mel + alpha * enc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot     += float(loss.item())
            tot_mse += float(loss_mse.item())
            tot_mel += float(loss_mel.item())
            tot_enc += float(enc_loss.item())

        n = len(train_loader)
        avg_tot  = tot / n
        avg_mse  = tot_mse / n
        avg_mel  = tot_mel / n
        avg_enc  = tot_enc / n

        train_total.append(avg_tot)
        train_mse.append(avg_mse)
        train_mel.append(avg_mel)
        train_enc.append(avg_enc)

        writer.add_scalar("train/total",   avg_tot, epoch)
        writer.add_scalar("train/mse",     avg_mse, epoch)
        writer.add_scalar("train/melL1",   avg_mel, epoch)
        writer.add_scalar("train/encoder", avg_enc, epoch)

        print(f"Epoch {epoch}/{epochs} | total {avg_tot:.6f} | mse {avg_mse:.6f} | mel {avg_mel:.6f} | enc {avg_enc:.6f}")

        # periodic checkpoint + running losses
        if epoch % 5 == 0:
            print(f"Saving periodic checkpoint at epoch {epoch}...")

            ep = os.path.join(save_dir, f"ep{epoch}")
            model.save_pretrained(ep)
            np.save(os.path.join(save_dir, "train_total.npy"), np.array(train_total))
            np.save(os.path.join(save_dir, "train_mse.npy"),   np.array(train_mse))
            np.save(os.path.join(save_dir, "train_mel.npy"),   np.array(train_mel))
            np.save(os.path.join(save_dir, "train_enc.npy"),   np.array(train_enc))

    # final save
    model.save_pretrained(os.path.join(save_dir, f"ep{epochs}"))
    np.save(os.path.join(save_dir, "train_total.npy"), np.array(train_total))
    np.save(os.path.join(save_dir, "train_mse.npy"),   np.array(train_mse))
    np.save(os.path.join(save_dir, "train_mel.npy"),   np.array(train_mel))
    np.save(os.path.join(save_dir, "train_enc.npy"),   np.array(train_enc))
    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    cli()
