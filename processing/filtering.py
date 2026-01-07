"""
    filtering.py
    Author : Milan Marocchi

    Various filters
"""

from typing import Any, Optional
import numpy as np
import os
import math
import scipy.signal as ssg
from scipy.io import loadmat 
import random
import librosa
# import pywt
import pyrubberband as pyrb
import torch
import scipy

import logging
logging.basicConfig(level=logging.INFO)

# Matlab engine
ENG = None

# def start_matlab(matlab_location: str):
#     print(matlab_location)
#     if matlab_location != '':
#         try:
#             import matlab.engine
#             global ENG
#             ENG = matlab.engine.start_matlab()
#             ENG.addpath(ENG.genpath(str(matlab_location)), nargout=0)  # type: ignore
#             logging.info('STARTED MATLAB')
#         except ImportError as e:
#             logging.error('Matlab engine not installed --- trying anyway')
#             logging.error(e)


def stop_matlab():
    if ENG is not None:
        ENG.exit()  # type: ignore
        logging.info('STOPPED MATLAB')


def stretch_resample(signal: np.ndarray, sample_rate: int, time_stretch_factor: float) -> np.ndarray:
    signal = librosa.effects.time_stretch(signal, rate=time_stretch_factor)
    signal = librosa.resample(signal, orig_sr=round(sample_rate * time_stretch_factor), target_sr=sample_rate)
    return signal


def time_stretch_crop(signal: np.ndarray, fs: int, time_stretch_factor: float) -> np.ndarray:
    """Time stretches the signal and crops it to it's original length"""
    sig_len = len(signal)
    signal = pyrb.time_stretch(signal, fs, time_stretch_factor)
    
    if len(signal) > sig_len:
        signal = signal[:sig_len]
    else:
        pad_length = sig_len - len(signal)
        signal = np.pad(signal, (0, pad_length), mode = 'constant')

    return signal


def random_crop(signal: np.ndarray, len_crop: int) -> np.ndarray:
    start = random.randint(0, len(signal) - len_crop)
    end = start + len_crop
    return signal[start:end]


def random_parametric_eq(signal: np.ndarray, sr: float, low: float, high: float, num_bands: int = 5) -> np.ndarray:
    equalised_signal = np.copy(signal)

    for _ in range(num_bands):

        b_low = np.random.uniform(low=low, high=0.95*high)
        b_high = random.choice([np.random.uniform(low=b_low+0.05*(high-low), high=high), b_low+(high-low)/num_bands])
        if b_high > high:
            b_high = high - np.random.uniform(1,2)
        # print([b_low / (sr / 2), b_high / (sr / 2)])
        # input()
        sos = ssg.iirfilter(N=1, Wn=[b_low / (sr / 2), b_high / (sr / 2)], btype='band',
                            analog=False, ftype='butter', output='sos')

        equalised_signal = np.asarray(ssg.sosfilt(sos, equalised_signal))

    return standardise_signal(standardise_signal(equalised_signal)/50 + standardise_signal(signal))


def band_stop(signal: np.ndarray, fs: int, fs_low: int, fs_high: int, order:int = 4):
   b, a = ssg.butter(order, [fs_low / fs, fs_high / fs], btype='bandstop') 
   signal = ssg.filtfilt(b, a, signal)

   return signal


def interpolate_nans(a: np.ndarray) -> np.ndarray:
    mask = np.isnan(a)
    a[mask] = np.interp(np.flatnonzero(mask),
                        np.flatnonzero(~mask),
                        a[~mask])
    return a


def znormalise_signal(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 1:
        signal -= np.mean(signal)
        signal /= np.std(signal)
    else:
        means = signal.mean(axis=0) # type: ignore
        stds = signal.std(axis=0) # type: ignore
        signal = (signal - means) / stds

    return signal


def normalise_signal(signal: np.ndarray) -> np.ndarray:

    signal = interpolate_nans(signal)
    signal -= np.mean(signal)
    signal /= np.max(np.abs(signal))
    signal = np.clip(signal, -1, 1)

    return signal


def standardise_signal(signal: np.ndarray) -> np.ndarray:
    return normalise_signal(signal)


def bandpass(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    low /= nyquist_freq
    high /= nyquist_freq

    sos = ssg.butter(1, [low, high], 'bandpass', analog=False, output='sos',)
    signal = ssg.sosfiltfilt(sos, signal)

    return signal


def notchfilter(signal: np.ndarray, fs: float, notch: float, Q: float) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    notch /= nyquist_freq

    b, a = ssg.iirnotch(notch, Q)
    signal = ssg.filtfilt(b, a, signal)

    return signal


def pre_filter_ecg(signal: np.ndarray, fs: float) -> np.ndarray:
    signal = notchfilter(signal, fs, 50, 55)
    # signal = notchfilter(signal, fs, 60, 55)
    # signal = notchfilter(signal, fs, 100, 55)
    # signal = notchfilter(signal, fs, 120, 55)
    # signal = bandpass(signal, fs, 0.25, 150)
    # signal = wavefilt(signal, 'sym4', 4)
    # signal = bandpass(signal, fs, 0.5, 70)
    return signal


def create_band_filters(fs: int) -> list[np.ndarray]:
    N = 61
    sr = fs
    wn = 45 * 2 / sr
    b1 = ssg.firwin(N, wn, window='hamming', pass_zero='lowpass') # type: ignore
    wn = [45 * 2 / sr, 80 * 2 / sr]
    b2 = ssg.firwin(N, wn, window='hamming', pass_zero='bandpass') # type: ignore
    wn = [80 * 2 / sr, 200 * 2 / sr]
    b3 = ssg.firwin(N, wn, window='hamming', pass_zero='bandpass') # type: ignore
    wn = 200 * 2 / sr
    b4 = ssg.firwin(N, wn, window='hamming', pass_zero='highpass') # type: ignore

    return [b1, b2, b3, b4]


def spike_removal_python(original_signal: np.ndarray, fs: float) -> np.ndarray:
    """Python implementation of schmidt spike removal"""
    # Find the window size (500 ms)
    windowsize = int(np.round(fs/2))

    # Find any samples outside of a integer number of windows:
    trailingsamples = len(original_signal) % windowsize

    # Reshape the signal into a number of windows:
    sampleframes = original_signal[:len(original_signal)-trailingsamples].reshape(-1, windowsize).T

    # Find the MAAs:
    MAAs = np.max(np.abs(sampleframes), axis=0)

    # While there are still samples greater than 3* the median value of the MAAs, then remove those spikes:
    while np.any(MAAs > np.median(MAAs)*3):
        # Find the window with the max MAA:
        window_num = np.argmax(MAAs)

        # Find the postion of the spike within that window:
        spike_position = np.argmax(np.abs(sampleframes[:, window_num]))

        # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative):
        zero_crossings = np.concatenate([(np.abs(np.diff(np.sign(sampleframes[:, window_num]))) > 1) , np.zeros(1)])

        # Find the start of the spike, finding the last zero crossing before spike position. If that is empty, take the start of the window:
        if len(np.where(zero_crossings[:spike_position] == True)[0]) == 0:
            spike_start = 1
        else:
            spike_start = np.where(zero_crossings[:spike_position] == True)[0][-1] + 1

        # Find the end of the spike, finding the first zero crossing after spike position. If that is empty, take the end of the window:
        zero_crossings[:spike_position] = 0
        if len(np.where(zero_crossings == True)[0]) == 0:
            spike_end = windowsize
        else:
            spike_end = np.where(zero_crossings == True)[0][0]

        # Set to Zero
        sampleframes[spike_start:spike_end, window_num] = 0.0001

        # Recalculate MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)

        if np.all(np.isnan(MAAs)) or np.max(MAAs) == np.max(np.abs(sampleframes), axis=0).max():
            break

    despiked_signal = sampleframes.T.flatten()

    # Add the trailing samples back to the signal:
    despiked_signal = np.append(despiked_signal, original_signal[len(despiked_signal):])

    return despiked_signal


def spike_removal(signal: np.ndarray, fs: float, matlab_location: str = "") -> np.ndarray:
    signal = np.array(signal).reshape(-1, 1)
    signal = ENG.schmidt_spike_removal(signal, float(fs))  # type: ignore
    signal = np.asarray(signal).flatten()
    return signal


def get_segment_time(pcg: np.ndarray, fs_old: float, fs_new: float, time: float = 1.25) -> list[list[int]]:
    """Gets the PCG segments based on time."""
    pcg_resampled = ssg.resample_poly(pcg, fs_new, fs_old)

    sample_increment = round(fs_new * time)
    seg_idxs = [[i, i, i, i] for i in range(sample_increment, len(pcg_resampled), sample_increment)]

    return seg_idxs


def get_hand_label_seg_pcg(path: str, filename: str) -> list[list[Any]]:

    segment_info = loadmat(os.path.join(path, f"{filename}"))
    segment_info = segment_info['state_ans']

    # Remember to adjust index for python instead of matlab
    breakpoint()

    return segment_info


def get_segment_pcg(pcg: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    """Gets the PCG segments using mixture of MATLAB and python."""
    pcg_resampled = ssg.resample_poly(pcg, fs_new, fs_old) # type: ignore

    pcg_resampled = ENG.butterworth_low_pass_filter(pcg_resampled, 2, 400, fs_new) # type: ignore
    pcg_resampled = ENG.butterworth_high_pass_filter(pcg_resampled, 2, 25, fs_new) # type: ignore
    pcg_resampled = np.array(pcg_resampled).reshape(-1, 1)
    pcg_resampled = ENG.schmidt_spike_removal(pcg_resampled, float(fs_new))  # type: ignore

    assigned_states = ENG.segmentation(pcg_resampled, fs_new) # type: ignore
    seg_idxs = np.asarray(ENG.get_states(assigned_states), dtype=int) - 1 # type: ignore

    return seg_idxs


def resample(signal: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    return ssg.resample_poly(signal, fs_new, fs_old)


def low_pass_butter(signal: np.ndarray, order: int, fc: float, fs: float) -> np.ndarray:
    wn = fc / fs
    b, a = ssg.butter(order, wn, btype="lowpass")

    return np.asarray(ssg.lfilter(b, a, signal))


def high_pass_butter(signal: np.ndarray, order: int, fc: float, fs: float) -> np.ndarray:
    wn = fc / fs
    b, a = ssg.butter(order, wn, btype="highpass")

    return np.asarray(ssg.lfilter(b, a, signal))


def delay_signal(signal: np.ndarray, delay: int) -> np.ndarray:
    """
       Delays a signal by the specified delay
    """
    hh = np.concatenate((
        np.zeros(delay),
        np.ones(1),
        np.zeros(delay)),
        dtype="float32"
    )

    delayed_signal = np.asarray(ssg.lfilter(hh.flatten(), 1, signal))

    return delayed_signal


def correlations(xdn: np.ndarray, ydn: np.ndarray, FL: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
        Calculates the correlation matrix and crosscorrelation vector
    """
    DL = max(np.shape(xdn))
    RXX: np.ndarray = np.zeros((FL, FL), dtype="float32")
    rxy: np.ndarray = np.zeros((FL, 1), dtype="float32")
    ryy: float = 0

    yp: np.ndarray = np.zeros(DL, dtype="float32")
    for ii in range(FL, DL, 1):
        xv = xdn[ii:ii-FL:-1].reshape(-1, 1)
        RXX = RXX + xv @ xv.T
        rxy = rxy + xv * ydn[ii]
        ryy = ryy + ydn[ii] ** 2
        yp[ii] = ydn[ii]

    return RXX, rxy, ryy, yp


def multi_correlations(vdn: np.ndarray, xdn:np.ndarray, FL: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    DL = max(np.shape(vdn))
    M: int = min(np.shape(vdn)) # Channels
    RVV: np.ndarray = np.zeros((M* FL, M* FL), dtype="float32")
    Rvx: np.ndarray = np.zeros((M* FL, M), dtype="float32")
    rxx: np.ndarray = np.zeros((M, M), dtype="float32")

    xp: np.ndarray = np.zeros((M, DL), dtype="float32")
    for ii in range(M*FL, DL, 1):
        RVV = RVV + vdn[ii:ii-M*FL:-1, :] @ vdn[ii:ii-M*FL:-1, :].T
        Rvx = Rvx + vdn[ii:ii-M*FL:-1, :] * xdn[ii:ii-M*FL:-1, :]
        rxx = rxx + xdn[ii] ** 2
        xp[:,ii] = xdn[ii, :]
    
    rxx = rxx * np.eye(M)

    return RVV, Rvx, rxx, xp


def optimal_weights(RXX: np.ndarray, rxy: np.ndarray, ryy: float | np.ndarray, FL: int, DL: float) -> np.ndarray:
    """
        Finds the optimal weights
    """
    err0 = 0.0005
    egv = np.linalg.eigvals(RXX)
    egv = egv[np.argmax(np.abs(egv))]

    # FIXME: Pre compute the inverse thing to save computations
    # Define calcs as lambdas
    err_fun = lambda w : float(((ryy - 2 * w.T @ rxy + w.T @ RXX @ w) / (DL - FL)))
    w_fun = lambda err0 : np.linalg.lstsq(RXX + err0 * (egv) * np.eye(RXX.shape[0]), rxy, rcond=-1)[0]
    # logging.info("Calculated w_fun")

    w = w_fun(err0)
    err = err_fun(w)

    total_passes = 0
    passes = 0
    err_prev = 0
    while abs(err0 - err) > 1e-4 or passes <= 2:
        if err == err_prev:  # To get the same result as matlab.
            passes += 1
        w = w_fun(err0)

        err_prev = err0
        err0 = err

        err = err_fun(w)

        total_passes += 1
        # logging.info(f"Weiner filter, {total_passes=}")
        if total_passes > 150:
            return w

    return w


def multi_optimal_weights(RVV: np.ndarray, rvx: np.ndarray, rxx: np.ndarray, FL: int, DL: float, M: int) -> np.ndarray:
    """
        Finds the optimal weights
    """
    err0 = 0.0005
    egv = np.linalg.eigvals(RVV)
    egv = float(egv[np.argmax(np.abs(egv))])
    I_ML = np.eye(RVV.shape[0])

    mse = lambda w, beta : np.trace((rxx - 2 * w.T @ rvx + w.T @ RVV @ w + beta * egv * w.T @ w) / (DL - FL))
    weiner_hopf = lambda beta : np.linalg.inv(RVV + beta * egv * I_ML) @ rvx

    w = weiner_hopf(err0)
    err = mse(w, err0)
    logging.info("Calculated w_fun")

    total_passes = 0
    passes = 0
    err_prev = 0
    while (abs(err0 - err) > 0.0001) or passes <= 2:
        if (abs(err - err_prev) < 0.0000001):  # To get the same result as matlab.
            passes += 1
        w = weiner_hopf(err0)
        
        err_prev = err0
        err0 = err
        err = mse(w, err0)

        total_passes += 1
        logging.info(f"Weiner filter, {total_passes=}")
        if total_passes > 150:
            return w

    return w


def weiner_filter(xdn: np.ndarray, ydn: np.ndarray, FL: int, DL: float) -> np.ndarray:
    """
        Runs the weiner filter algorithm
    """
    RXX, rxy, ryy, yp = correlations(xdn, ydn, FL)
    w = optimal_weights(RXX, rxy, ryy, FL, DL)

    # apply weiner filter
    yhat = ssg.lfilter(w.flatten(), 1, xdn) # w^T v
    #yhat2 = ssg.fftconvolve(w.flatten(), xdn, mode='valid')
    #print(yhat - yhat2) 
    e = yp.T - yhat # e = x - w^T v

    return e


def multi_weiner_filter(vdn: np.ndarray, xdn: np.ndarray, FL: int, DL: int, fs: int) -> np.ndarray:
    """
    Computes the weiner filters between all of the diff channels and builds the matrix. 
    """
    # Currently just a stub
    M = min(np.shape(vdn)) # Number of channels
    N = max(np.shape(vdn)) # Number of samples in the signal

    e = np.zeros((N, M), dtype="float32")
    W = np.zeros((M * FL, M), dtype="float32")

    RVV, Rvx, rxx, xp = multi_correlations(vdn, xdn, FL)
    W = multi_optimal_weights(RVV, Rvx, rxx, FL, DL, M)

    xhat = np.zeros_like(vdn)
    for channel in range(M):
        x_subchannel = np.zeros((max(xdn.shape)))
        for sub_channel in range(M):
            x_subchannel += ssg.lfilter((W[sub_channel*FL:(sub_channel+1)*FL, channel].flatten()), 1, vdn[:, sub_channel])
        xhat[:, channel] = (x_subchannel) 
        #xhat += ssg.lfilter(W[:, channel], 1, vdn)
    e = xp - xhat.T
    return e


def noise_canc(xdn: np.ndarray, ydn: np.ndarray, fc: float = 20, fs: float = 2000, FL:int = 128, hp: bool = True) -> np.ndarray:
    """
    Noise cancellation using weiner filter and hp

    xdn is background noise,
    ydn is the signal with background noise
    """
    DL = max(np.shape(xdn))
    ydn = delay_signal(ydn, math.floor(FL/2)) 

    # High pass if required due to small filter length
    if hp:
        xdn = high_pass_butter(xdn, 2, fc, fs)

    return weiner_filter(xdn, ydn, FL, DL)


def multi_noise_canc(xdn: np.ndarray, ydn: np.ndarray, fc: float = 25, fs: float = 1000, FL: int = 128, hp: bool = False) -> np.ndarray:
    """
    Noise cancelation of multichannel with weiner filter

    xdn is background noise as a matrix [x1, x2, x3, ..., xn].T
    ydn is wanted signal as a matrix [y1, y2, y3, ..., yn].T 
    """
    DL = max(np.shape(xdn))
    for channel in range(min(ydn.shape)):
        ydn[:, channel] = delay_signal(ydn[:, channel], math.floor(FL/2))

    return multi_weiner_filter(xdn, ydn, FL, DL, int(fs))


# def wavelet_denoise(signal: np.ndarray, wavelet: str, level: int) -> np.ndarray:
#     signal = standardise_signal(signal)
#     coeffs = pywt.wavedec(signal, wavelet, level=level)

#     sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6725
#     var = np.var(coeffs[-1])
#     threshold = sigma**2 / np.sqrt(max(var - sigma**2, 1e-30))

#     coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
#     return pywt.waverec(coeffs, wavelet)


def wdenoise(signal: np.ndarray, wavelet: str, level: int, method: Optional[str] = None):
    if method is not None:
        denoised_signal = ENG.wdenoise(signal, float(level), 'Wavelet', wavlet, 'DenoisingMethod', method) # type: ignore
    else:
        denoised_signal = ENG.wdenoise(signal, float(level), 'Wavelet', wavelet) # type: ignore

    return denoised_signal


def add_chirp(audio_signal, fs):
    t = np.arange(len(audio_signal)) / fs

    chirp_signal = scipy.signal.chirp(t, f0=0, f1=fs/2, t1=t[-1], method='linear')
    chirp_signal = (chirp_signal / np.max(np.abs(chirp_signal))) * max(0.5, np.max(np.abs(audio_signal)))

    return audio_signal + chirp_signal

def create_spectrogram(signal, transform):
    spectrogram = transform(signal)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    return spectrogram

