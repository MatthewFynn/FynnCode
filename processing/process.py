"""
    preprocessing.py
    Author: Fynn

    Purpose: Run any preprocessing that is required for signals before training
"""

from processing.filtering import (
    interpolate_nans,
    resample,
    low_pass_butter,
    high_pass_butter,
    normalise_signal,
    pre_filter_ecg,
    noise_canc,
    znormalise_signal
    # spike_removal_python,
    # create_band_filters,
    # start_matlab,
    # stop_matlab
)
from processing.augmentation import augment_multi_pcg, augment_multi_pcg_ecppg
from util.fileio import  save_ticking_signals, create_multi_wav
from util.paths import EPHNOGRAM, MIT
import numpy as np
import os


# def pre_process_pcg(pcg: np.ndarray, fs: int, fs_new: int) -> np.ndarray:
#     pcg = interpolate_nans(pcg)
#     pcg = resample(pcg, fs, fs_new)

#     pcg = low_pass_butter(pcg, 2, 950, fs)
#     pcg = high_pass_butter(pcg, 2, 25, fs)
#     pcg = normalise_signal(pcg)

#     return pcg
def pre_process_pcg(pcg: np.ndarray, fs: int, low: int, high: int) -> np.ndarray:

    # pcg = low_pass_butter(pcg, 5, int(fs/2)-100, fs)
    pcg = low_pass_butter(pcg, 5, high, fs)
    pcg = high_pass_butter(pcg, 5, low, fs)
    # pcg = normalise_signal(pcg)

    return pcg

def pre_process_pcg2(pcg: np.ndarray, fs: int) -> np.ndarray:

    pcg = low_pass_butter(pcg, 2, 450, fs)
    pcg = high_pass_butter(pcg, 2, 25, fs)
    pcg = normalise_signal(pcg)

    return pcg


def pre_process_ecg(ecg: np.ndarray, fs: int) -> np.ndarray:

    # ecg = low_pass_butter(ecg, 2, 150, fs)
    ecg = high_pass_butter(ecg, 2, 2, fs)
    # ecg = normalise_signal(ecg)

    # ecg = pre_filter_ecg(ecg, fs)

    return ecg

def adjust_length(arr, target_length):
    """Zero-pads or trims a 1D NumPy array to match the target length."""
    current_length = len(arr)
    
    if current_length < target_length:
        # Zero-pad
        arr = np.pad(arr, (0, target_length - current_length), mode='constant', constant_values=0)
    elif current_length > target_length:
        # Trim
        arr = arr[:target_length]
    
    return arr

def save_signal_segments(filename, multi_signal: np.ndarray, fs, segment_length, overlap, path):
    """
    Divides a signal into segments of specified length with overlap and saves each segment.

    Parameters:
    - filename (str): The original filename ... to be appended
    - signal (np.array): The audio signal to be segmented.
    - fs (int): The sampling frequency of the audio signal.
    - segment_length (int): The number of samples in each segment.
    - overlap (int): The number of samples that each segment should overlap with the previous one.
    - path (str): The directory where the segments will be saved.
    """

    #remove path from filename
    filename = filename.split('/')[-1]
    # Calculate the step size (how much to move the window each time)
    step_size = segment_length - overlap

    # Initialize the start index
    start = 0
    segment_count = 0

    # Loop through the signal to extract and save segments
    while start + segment_length <= multi_signal.shape[0]:
        # Extract the segment
        segment = multi_signal[start:start + segment_length,0::]

        # Save the segment
        filename_new = f"{filename[0:-4]}_f{segment_count:02d}.wav"
        full_path = os.path.join(path, filename_new)
        
        save_ticking_signals(segment, fs, full_path)

        # Move the start index by the step size
        start += step_size
        segment_count += 1

    # print(f"Total {segment_count} segments saved in {path}")

def segment_nc_aug(filename, multi_signal_pcg: list, multi_signal_noi: list, fs, segment_length, overlap, path, num_aug, nc = 0, FL = 256):
    """
    Divides a signal into segments of specified length with overlap and the applies noise cancellation
    and augmentation

    Parameters:
    - filename (str): The original filename ... to be appended
    - signal (np.array): The audio signal to be segmented.
    - fs (int): The sampling frequency of the audio signal.
    - segment_length (int): The number of samples in each segment.
    - overlap (int): The number of samples that each segment should overlap with the previous one.
    - path (str): The directory where the segments will be saved.
    """
    #remove path from filename
    filename = filename.split('/')[-1]
    # Calculate the step size (how much to move the window each time)
    step_size = segment_length - overlap
    # Initialize the start index
    start = 0
    segment_count = 0

    # Loop through the signal to extract and save segments
    while start + segment_length <= multi_signal_pcg[0].shape[0]-FL-1:
        # Extract the segment
        filename_frag = f"{filename[0:-4]}_f{segment_count:02d}.wav"
        multi_pcg_frag = list()
        for idx, pcg_wav in enumerate(multi_signal_pcg):
            pcg_seg = pcg_wav[start:start + segment_length +FL+1]

            if nc == 1 and idx < 6: #only pcg channels 1-6
                noi_seg = multi_signal_noi[idx][start:start + segment_length+FL]
                pcg_seg_nc = noise_canc(xdn=noi_seg, ydn=pcg_seg, fs = fs, FL = FL, hp = True)
                multi_pcg_frag.append(pcg_seg_nc[FL::])
                del pcg_seg_nc
                
            else:
                multi_pcg_frag.append(pcg_seg[:len(pcg_seg) - (FL+1)])
                del pcg_seg


        multi_pcg_frag_pre = []
        for idx, pcg in enumerate(multi_pcg_frag):
            multi_pcg_frag_pre.append(pre_process_pcg(pcg, fs))

        pcg_wav = create_multi_wav(multi_pcg_frag_pre, 7)
        #save_original_fragment
        full_path = os.path.join(path, filename_frag)
        
        save_ticking_signals(pcg_wav, fs, full_path)
        
        #augment the fragment
        for a in range(num_aug-1):
            multi_pcg_frag_aug = augment_multi_pcg(orig_multi_pcg_wav=multi_pcg_frag, sr=fs, EPHNOGRAM=EPHNOGRAM)
            multi_pcg_frag_aug_pre = []
            for idx, pcg in enumerate(multi_pcg_frag_aug):
                multi_pcg_frag_aug_pre.append(pre_process_pcg(pcg, fs))

            filename_frag_aug = filename_frag[0:-4] + f'_aug{a:02d}.wav'

            pcg_wav_aug = create_multi_wav(multi_pcg_frag_aug_pre,7)
            full_path = os.path.join(path, filename_frag_aug)
            save_ticking_signals(pcg_wav_aug, fs, full_path)
            
        # Move the start index by the step size
        start += step_size
        segment_count += 1

def segment_nc_aug_ecppg(filename, multi_signal: list, multi_signal_noi: list, fs, segment_length, overlap, path, num_aug, nc = 0, FL = 256):
    """
    Divides a signal into segments of specified length with overlap and the applies noise cancellation
    and augmentation

    Parameters:
    - filename (str): The original filename ... to be appended
    - signal (np.array): The audio signal to be segmented.
    - fs (int): The sampling frequency of the audio signal.
    - segment_length (int): The number of samples in each segment.
    - overlap (int): The number of samples that each segment should overlap with the previous one.
    - path (str): The directory where the segments will be saved.
    """
    #remove path from filename
    filename = filename.split('/')[-1]
    # Calculate the step size (how much to move the window each time)
    step_size = segment_length - overlap
    # Initialize the start index
    start = 0
    segment_count = 0
    multi_signal[8] = normalise_signal(multi_signal[8])
    multi_signal[7] = normalise_signal(multi_signal[7])
    # multi_signal[2] = normalise_signal(multi_signal[2])
    # Loop through the signal to extract and save segments
    while start + segment_length <= multi_signal[0].shape[0]-FL-1:
        # Extract the segment
        filename_frag = f"{filename[0:-4]}_f{segment_count:02d}.wav"
        multi_frag = list()
        for idx, sig in enumerate(multi_signal):
            seg = sig[start:start + segment_length +FL+1]

            if nc == 1 and idx < 6: #only pcg channels 1-6
                noi_seg = multi_signal_noi[idx][start:start + segment_length+FL]
                seg_nc = noise_canc(xdn=noi_seg, ydn=seg, fs = fs, FL = FL, hp = True)
                multi_frag.append(seg_nc[FL::])
                del seg_nc
                
            else:
                multi_frag.append(seg[:len(seg) - (FL+1)])
                del seg
            
        
        multi_frag_pre = []
        for idx, sig in enumerate(multi_frag):
            if idx < 7:
                multi_frag_pre.append(pre_process_pcg(sig, fs)) #pcg signals
                # multi_frag_pre.append(sig) #pcg signals
            else:
                multi_frag_pre.append(pre_process_ecg(sig, fs)) #ecg and ppg
                # multi_frag_pre.append(sig) #ecg and ppg

        sig_wav = create_multi_wav(multi_frag_pre, 9)
        #save_original_fragment
        full_path = os.path.join(path, filename_frag)
        
        save_ticking_signals(sig_wav, fs, full_path)
        
        #augment the fragment
        for a in range(num_aug-1):
            multi_frag_aug = augment_multi_pcg_ecppg(orig_multi_wav=multi_frag, sr=fs, EPHNOGRAM=EPHNOGRAM, MIT = MIT)
            multi_frag_aug_pre = []
            for idx, sig in enumerate(multi_frag_aug):
                if idx < 7:
                    multi_frag_aug_pre.append(pre_process_pcg(sig, fs))
                else:
                    multi_frag_aug_pre.append(pre_process_ecg(sig, fs))
                    # multi_frag_pre.append(sig)

            filename_frag_aug = filename_frag[0:-4] + f'_aug{a:02d}.wav'

            sig_wav_aug = create_multi_wav(multi_frag_aug_pre,9)
            full_path = os.path.join(path, filename_frag_aug)
            save_ticking_signals(sig_wav_aug, fs, full_path)
            
        # Move the start index by the step size
        start += step_size
        segment_count += 1

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

    pad_idx = len(array) - pad_amount

    return array, pad_idx


def normalise_2d_array_length(array, normalised_length):
    """
    Pad or crop the array to have a shape of (2500, second_dim_size).

    :param array: The input array.
    :param normalised_length: Length to normalise array to.
    :return: Array with shape (2500, second_dim_size).
    """
    pad_amount = 0
    # Pad or crop the first dimension to 2500
    if array.shape[0] < normalised_length:
        # Pad
        pad_amount = normalised_length - array.shape[0]
        array = np.pad(array, ((0, pad_amount), (0, 0)), mode='constant')
    elif array.shape[0] > normalised_length:
        # Crop
        array = array[:normalised_length, :]

    pad_idx = len(array) - pad_amount

    return array, pad_idx
