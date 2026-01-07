"""
    augmentation.py
    Author: Leigh Abbott

    Purpose: Data augmentation on pcg and ecg signals
"""
from processing.filtering import (
    standardise_signal,
    stretch_resample,
    random_parametric_eq,
    random_crop,
    band_stop,
    time_stretch_crop
)
from util.paths import (
    EPHNOGRAM,
    MIT,
)

import librosa
import random
import numpy as np
import scipy.signal as ssg
import wfdb
import glob
import os
import matplotlib.pyplot as plt
from typing import Optional


def randfloat(low: float, high: float) -> float:
    return low + random.random() * (high - low)


def get_record(path: str, max_sig_len_s: float = -1.0) -> wfdb.Record:

    header = wfdb.rdheader(path)
    sig_len = header.sig_len
    fs = header.fs

    if max_sig_len_s <= -1.0:
        target_sig_len = sig_len
    else:
        target_sig_len = round(max_sig_len_s * fs) # type: ignore

    if sig_len > target_sig_len:
        sampfrom = random.randint(0, sig_len - target_sig_len)
        sampto = sampfrom + target_sig_len
    else:
        sampfrom = 0
        sampto = sig_len

    rec = wfdb.rdrecord(path, sampfrom=sampfrom, sampto=sampto)
    return rec


def get_pcg_noise(target_sr: float, len_record: int, path: str = "") -> np.ndarray:

    if path == "":
        path = EPHNOGRAM
    valid_files = glob.glob(f"{path}/*.hea")

    num_tries = 0

    while num_tries < 50:

        try:

            num_tries += 1

            valid_file = random.choice(valid_files)

            record = get_record(valid_file.removesuffix('.hea'))
            pcg_noise_1 = record.p_signal[:, record.sig_name.index('AUX1')] # type: ignore
            pcg_noise_2 = record.p_signal[:, record.sig_name.index('AUX2')] # type: ignore
            pcg_noise_1 = ssg.resample_poly(pcg_noise_1, target_sr, record.fs)
            pcg_noise_2 = ssg.resample_poly(pcg_noise_2, target_sr, record.fs)
            pcg_noise_1 = standardise_signal(random_crop(pcg_noise_1, len_record))
            pcg_noise_2 = standardise_signal(random_crop(pcg_noise_2, len_record))
            pcg_noise_1 = random.choice([0, randfloat(0.0, 0.05)]) * pcg_noise_1
            pcg_noise_2 = random.choice([0, randfloat(0.0, 0.05)]) * pcg_noise_2

            pcg_comb_noise = pcg_noise_1 + pcg_noise_2
            # Try to avoid the divide by 0 in standardise_signal
            if np.max(np.abs(pcg_comb_noise)) > 0.0:
                pcg_comb_noise = standardise_signal(pcg_comb_noise)

            # # Reduce the noise of the signal
            if np.std(pcg_comb_noise) > 0.1:
                pcg_comb_noise *= 0.05
            # pcg_comb_noise*=0.05
            return pcg_comb_noise


        except ValueError:
            pass

    return np.zeros(len_record)


def get_ecg_noise(target_sr: float, len_record: int, path: str = "") -> np.ndarray:

    if path == "":
        path = MIT

    em_noise = get_record(os.path.join(path,'em'))
    bw_noise = get_record(os.path.join(path,'bw'))
    ma_noise = get_record(os.path.join(path,'ma'))

    em_noise = ssg.resample_poly(em_noise.p_signal[:, 0], target_sr, em_noise.fs) # type: ignore
    bw_noise = ssg.resample_poly(bw_noise.p_signal[:, 0], target_sr, bw_noise.fs) # type: ignore
    ma_noise = ssg.resample_poly(ma_noise.p_signal[:, 0], target_sr, ma_noise.fs) # type: ignore

    em_noise = random.choice([0, randfloat(0.0, 0.25)]) * standardise_signal(random_crop(em_noise, len_record))
    bw_noise = random.choice([0, randfloat(0.0, 0.5)]) * standardise_signal(random_crop(bw_noise, len_record))
    ma_noise = random.choice([0, randfloat(0.0, 0.25)]) * standardise_signal(random_crop(ma_noise, len_record))

    return em_noise + bw_noise + ma_noise



def augment_time_warp(signal: np.ndarray, sr: int, min_factor: float = 0.8, max_factor: float = 1.2, time_stretch_factor: Optional[float] = None) -> np.ndarray:
    if time_stretch_factor is None:
        time_stretch_factor = randfloat(min_factor, max_factor)
    signal = stretch_resample(signal, sr, time_stretch_factor)
    signal = standardise_signal(signal)
    return signal


def augment_multi_pcg(orig_multi_pcg_wav: list, sr: int,
                    prob_noise: float = 0.50, 
                    prob_wandering_volume: float = 0.75, prob_time_warp: float = 0.7,
                    prob_hpss: float = 0.65, prob_banding: float = 0.65,
                    prob_real_noise: float = 0.5,
                    EPHNOGRAM="") -> list[np.ndarray]:
    """
    For multichannel pcg recordings, to ensure the same augmentation occurs on all channels
    """
    pcg_multi_wav = list()

    for orig_pcg_wav in orig_multi_pcg_wav:
        pcg_wav = orig_pcg_wav.copy()
        pcg_wav = standardise_signal(pcg_wav)
        pcg_multi_wav.append(pcg_wav)

    if np.random.rand() < prob_hpss:

        n_fft_1 = random.choice([512, 1024, 2048])
        win_len_1 = n_fft_1
        hop_len_1 = random.choice([16, 32, 64, 128])
        margin_1 = (randfloat(1.0, 2.0), randfloat(1.0, 2.0))
        kernel_1 = (random.randint(5, 30), random.randint(5, 30))

        n_fft_2 = random.choice([512, 1024, 2048])
        win_len_2 = n_fft_2
        hop_len_2 = random.choice([16, 32, 64, 128])
        margin_2 = (randfloat(1.0, 4.0), randfloat(1.0, 4.0))
        kernel_2 = (random.randint(5, 30), random.randint(5, 30))

        for idx, pcg_wav in enumerate(pcg_multi_wav):

            decomp = librosa.stft(
                pcg_wav,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_1,
                kernel_size=kernel_1,
            )

            y_1 = librosa.istft(
                harmon,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            y_2 = librosa.istft(
                percus,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            decomp = librosa.stft(
                y_1,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_2,
                kernel_size=kernel_2,
            )

            y_11 = librosa.istft(
                harmon,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            y_12 = librosa.istft(
                percus,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            decomp = librosa.stft(
                y_2,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_2,
                kernel_size=kernel_2,
            )

            y_21 = librosa.istft(
                harmon,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            y_22 = librosa.istft(
                percus,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            min_len = min(len(y_i) for y_i in (y_11, y_12, y_21, y_22))

            pcg_wav_1 = standardise_signal(
                1 * randfloat(0.01, 10)*y_11[:min_len]
                + 1 * randfloat(0.01, 10)*y_12[:min_len]
                + 1 * randfloat(0.01, 10)*y_21[:min_len]
                + 1 * randfloat(0.01, 10)*y_22[:min_len]
            )

            pcg_wav_2 = standardise_signal(
                1 * randfloat(0.01, 10)*standardise_signal(y_11[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_12[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_21[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_22[:min_len])
            )

            pcg_multi_wav[idx] = standardise_signal(pcg_wav_1 + randfloat(0.01, 0.05)*pcg_wav_2)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(0.90, 1.1)

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav = time_stretch_crop(pcg_wav, sr, time_stretch_factor)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_wandering_volume:
        t = np.arange(pcg_multi_wav[0].size) / sr
        vol_mod_1 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        vol_mod_2 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_multi_wav[idx] *= (1 + vol_mod_1 + vol_mod_2)
            pcg_multi_wav[idx] = standardise_signal(pcg_multi_wav[idx])

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_banding:
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav = random_parametric_eq(pcg_wav, sr, low=2, high=500)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_real_noise:
        pcg_noise = get_pcg_noise(sr, len(pcg_multi_wav[0]), EPHNOGRAM)
        for idx in range(len(pcg_multi_wav)):
            pcg_multi_wav[idx] += pcg_noise

    return pcg_multi_wav 

def augment_multi_pcg_ecppg(orig_multi_wav: list, sr: int,
                    prob_noise: float = 0.40, 
                    prob_wandering_volume: float = 0.75, prob_time_warp: float = 0.7,
                    prob_hpss: float = 0.65, prob_banding: float = 0.6,
                    prob_real_noise: float = 0.5,
                    EPHNOGRAM="", MIT = "") -> list[np.ndarray]:
    """
    For multichannel pcg recordings, to ensure the same augmentation occurs on all channels
    """
    pcg_multi_wav = list()
    ecg_ppg_wav = list()

    for orig_pcg_wav in orig_multi_wav[0:7]:
        pcg_wav = orig_pcg_wav.copy()
        pcg_wav = standardise_signal(pcg_wav)
        pcg_multi_wav.append(pcg_wav)
    ecg_wav = orig_multi_wav[7].copy()
    ecg_wav = standardise_signal(ecg_wav)
    ecg_ppg_wav.append(ecg_wav)
    ppg_wav = orig_multi_wav[8].copy()
    ppg_wav = standardise_signal(ppg_wav)
    ecg_ppg_wav.append(ppg_wav)
    
    if np.random.rand() < prob_hpss:

        n_fft_1 = random.choice([512, 1024, 2048])
        win_len_1 = n_fft_1
        hop_len_1 = random.choice([16, 32, 64, 128])
        margin_1 = (randfloat(1.0, 2.0), randfloat(1.0, 2.0))
        kernel_1 = (random.randint(5, 30), random.randint(5, 30))

        n_fft_2 = random.choice([512, 1024, 2048])
        win_len_2 = n_fft_2
        hop_len_2 = random.choice([16, 32, 64, 128])
        margin_2 = (randfloat(1.0, 4.0), randfloat(1.0, 4.0))
        kernel_2 = (random.randint(5, 30), random.randint(5, 30))

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            orig_len = len(pcg_wav)
            decomp = librosa.stft(
                pcg_wav,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_1,
                kernel_size=kernel_1,
            )

            y_1 = librosa.istft(
                harmon,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            y_2 = librosa.istft(
                percus,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            decomp = librosa.stft(
                y_1,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_2,
                kernel_size=kernel_2,
            )

            y_11 = librosa.istft(
                harmon,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            y_12 = librosa.istft(
                percus,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            decomp = librosa.stft(
                y_2,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_2,
                kernel_size=kernel_2,
            )

            y_21 = librosa.istft(
                harmon,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            y_22 = librosa.istft(
                percus,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            min_len = min(len(y_i) for y_i in (y_11, y_12, y_21, y_22))

            pcg_wav_1 = standardise_signal(
                1 * randfloat(0.01, 10)*y_11[:min_len]
                + 1 * randfloat(0.01, 10)*y_12[:min_len]
                + 1 * randfloat(0.01, 10)*y_21[:min_len]
                + 1 * randfloat(0.01, 10)*y_22[:min_len]
            )

            pcg_wav_2 = standardise_signal(
                1 * randfloat(0.01, 10)*standardise_signal(y_11[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_12[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_21[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_22[:min_len])
            )

            pcg_multi_wav[idx] = normalise_array_length(standardise_signal(pcg_wav_1 + randfloat(0.01, 0.05)*pcg_wav_2), orig_len)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)
        for idx, bio_wav in enumerate(ecg_ppg_wav):
            bio_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, bio_wav.shape)
            ecg_ppg_wav[idx] = standardise_signal(bio_wav)

    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(0.90, 1.1)

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav = time_stretch_crop(pcg_wav, sr, time_stretch_factor)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)
        for idx, bio_wav in enumerate(ecg_ppg_wav):
            bio_wav = time_stretch_crop(bio_wav, sr, time_stretch_factor)
            ecg_ppg_wav[idx] = standardise_signal(bio_wav)

    if np.random.rand() < prob_wandering_volume:
        t = np.arange(pcg_multi_wav[0].size) / sr
        vol_mod_1 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        vol_mod_2 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_multi_wav[idx] *= (1 + vol_mod_1 + vol_mod_2)
            pcg_multi_wav[idx] = standardise_signal(pcg_multi_wav[idx])

        t = np.arange(ecg_ppg_wav[0].size) / sr
        baseline_wander = randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        baseline_wander += randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        for idx, bio_wav in enumerate(ecg_ppg_wav):
            bio_wav += baseline_wander
            ecg_ppg_wav[idx] = standardise_signal(bio_wav)

 
    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)
        for idx, bio_wav in enumerate(ecg_ppg_wav):
            bio_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, bio_wav.shape)
            ecg_ppg_wav[idx] = standardise_signal(bio_wav)

    if np.random.rand() < prob_banding:
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav = random_parametric_eq(pcg_wav, sr, low=2, high=500)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)
        for idx, bio_wav in enumerate(ecg_ppg_wav):
            bio_wav = random_parametric_eq(bio_wav, sr, low=0.25, high=40)
            ecg_ppg_wav[idx] = standardise_signal(bio_wav)

    if np.random.rand() < prob_real_noise:
        pcg_noise = get_pcg_noise(sr, len(pcg_multi_wav[0]), EPHNOGRAM)
        for idx in range(len(pcg_multi_wav)):
            pcg_multi_wav[idx] += pcg_noise
        ecg_noise = get_ecg_noise(sr, len(ecg_ppg_wav[0]), MIT)
        for idx in range(len(ecg_ppg_wav)):
            ecg_ppg_wav[idx] += ecg_noise

    pcg_multi_wav.append(ecg_ppg_wav[0])
    pcg_multi_wav.append(ecg_ppg_wav[1]) #this puts the ecg/ppg back to channels 8/9

    return pcg_multi_wav 

def normalise_array_length(array, normalised_length):
    """
    Pad or crop the array to have a shape of (2500, second_dim_size).

    :param array: The input array.
    :param normalised_length: Length to normalise array to.
    :return: Array with shape (2500, second_dim_size).
    """
    pad_amount = 0
    # Pad or crop the first dimension to 2500
    if len(array) < normalised_length:
        # Pad
        pad_amount = normalised_length - len(array)
        array = np.pad(array, (0, pad_amount), mode='constant')
    elif len(array) > normalised_length:
        # Crop
        array = array[:normalised_length]


    return array

def plotting_multi(multi_pcg_data: list):

    num_subplots = len(multi_pcg_data)

    # Create subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 15))

    # Plot each array on a separate subplot
    for i, array in enumerate(multi_pcg_data):
        axes[i].plot(standardise_signal(array))
        axes[i].set_title(f'Array {i+1}')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel('Value')

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

def augment_signals(orig_ecg_wav: np.ndarray, orig_pcg_wav: np.ndarray, sr: int,
                    prob_noise: float = 0.30, prob_baseline_wander: float = 0.30,
                    prob_wandering_volume: float = 0.75, prob_time_warp: float = 0.25,
                    prob_hpss: float = 0.75, prob_banding: float = 0.25,
                    prob_real_noise: float = 0.5,
                    MIT="", EPHNOGRAM="") -> tuple[np.ndarray, np.ndarray]:

    ecg_wav = orig_ecg_wav.copy()
    pcg_wav = orig_pcg_wav.copy()

    ecg_wav = standardise_signal(ecg_wav)
    pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        ecg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, ecg_wav.shape)
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_baseline_wander:
        t = np.arange(ecg_wav.size) / sr
        baseline_wander = randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        baseline_wander += randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        ecg_wav += baseline_wander
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(0.94, 1.006)
        ecg_wav = stretch_resample(ecg_wav, sr, time_stretch_factor)
        pcg_wav = stretch_resample(pcg_wav, sr, time_stretch_factor)
        pcg_wav = standardise_signal(pcg_wav)
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_wandering_volume:
        t = np.arange(pcg_wav.size) / sr
        vol_mod_1 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        vol_mod_2 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        pcg_wav *= (1 + vol_mod_1 + vol_mod_2)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        ecg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, ecg_wav.shape)
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_banding:
        pcg_wav = random_parametric_eq(pcg_wav, sr, low=2, high=500)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_banding:
        ecg_wav = random_parametric_eq(ecg_wav, sr, low=0.25, high=100)
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_real_noise:
        ecg_wav += get_ecg_noise(sr, len(ecg_wav), MIT)

    if np.random.rand() < prob_real_noise:
        pcg_wav += get_pcg_noise(sr, len(pcg_wav), EPHNOGRAM)

    return ecg_wav, pcg_wav

class RandomTimeFreqMask:
    """Applies a random line mask in the spectrogram of the audio"""
    def __init__(self, thickness: float, fs: int):
        self.thickness = thickness
        self.fs = fs

    def __call__(self, audio: np.ndarray):
        sig_len = len(audio) 
        assert sig_len > 1000, "Correct way to get sig len for multi-channel"

        time_thickness = int(self.thickness * sig_len)
        freq_thickness = int(self.thickness * self.fs)

        if random.random() > 0.8:
            if random.random() > 0.5:
                # Time masking
                time = random.randint(0, len(audio) - time_thickness - 1)

                # check multi channel
                if audio.ndim == 1:
                    audio[time:time + time_thickness] = 0
                else:
                    num_channels = audio.shape[1]
                    channel = random.randint(0, num_channels - 1)
                    audio[time:time + time_thickness, channel]
            else:
                # Frequency masking
                frequency = random.randint(1, self.fs - freq_thickness - 1)

                # check multi channel
                if audio.ndim == 1:
                    audio = band_stop(audio, self.fs, frequency, frequency + freq_thickness)
                else:
                    num_channels = audio.shape[1]
                    channel = random.randint(0, num_channels - 1) 
                    audio[:, channel] = band_stop(audio[:, channel], self.fs, frequency, frequency + freq_thickness)

        return audio

class RandomStretch:
    """Applies a random stretch to a signal"""

    def __init__(self, fs: int):
        self.fs = fs

    def __call__(self, audio: np.ndarray):
        if random.random() > 0.8:

            stretch_factor = 0.96 + random.random() * (1.04 - 0.96)
            audio = time_stretch_crop(audio, self.fs, stretch_factor)

        return audio
