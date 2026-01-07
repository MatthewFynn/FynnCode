import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import wavfile
import scipy.signal as ssg
def resample(signal: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    return ssg.resample_poly(signal, fs_new, fs_old)


def segment_signal_noise_multi_MedPow(
    multi_signal,
    frame_len_sec,
    fs_new,
    threshold,
    total_sig,
    nm,
    fig=0
):
    """
    Python version of MATLAB:
    segment_signal_noise_multi_MedPow.m

    Parameters
    ----------
    multi_signal : np.ndarray
        Shape (N,) or (N, C)
    frame_len_sec : float
        Frame length in seconds
    fs_new : int
        Sampling frequency
    threshold : float
        Energy threshold multiplier
    total_sig : int
        Number of signals (channels)
    nm : int
        1 = noise mic, 0 = PCG
    fig : int
        Ignored (MATLAB plotting flag)

    Returns
    -------
    ind_flag_m : np.ndarray or list of np.ndarray
        Each array is (K,2) [start, end] indices (0-based)
    """

    # Ensure 2D: (samples, channels)
    if multi_signal.ndim == 1:
        multi_signal = multi_signal[:, None]

    frame_len_sample = int(round(frame_len_sec * fs_new))
    ind_flag_m = [] if total_sig > 1 else None

    for c in range(total_sig):
        signal = multi_signal[:, c]
        len_signal = len(signal)

        no_frames = int(np.floor(len_signal / frame_len_sample))
        En = np.zeros(no_frames)

        # ---- Frame energy ----
        for i in range(no_frames):
            start = i * frame_len_sample
            end   = start + frame_len_sample-1
            sig = signal[start:end+1]
            En[i] = np.sum(sig ** 2)

        # ---- Median energy (exclude edges) ----
        if len(En) > 2:
            med_val = float(np.median(En[1:-1].astype(np.float64)))
        else:
            med_val = np.median(En)

        ind_flag = []

        # ---- Thresholding ----
        
        j = 0  # Python uses 0-based indexing

        for i in range(len(En)):  # i = 0, 1, ..., len(En)-1
            if En[i] > threshold * med_val:
                start = i * frame_len_sample 
                end = (i + 1) * frame_len_sample-1
                ind_flag.append([start, end])
                j += 1

        #ind_flag = np.array(ind_flag, dtype=int)

        # ---- NM-specific rule: add max-energy frame ----
        if nm == 1 and len(En) > 0:
            k = np.argmax(En)
            start = k * frame_len_sample
            end   = (k + 1) * frame_len_sample
            ind_flag.append([start, end])
        
        ind_flag = np.array(ind_flag, dtype=int)

        #ind_flag_m.append(ind_flag)

    # Store results
        if total_sig > 1:
            ind_flag_m.append(ind_flag)
        else:
            ind_flag_m = ind_flag

    return ind_flag_m




def schmidt_spike_removal(signal, fs):
    """
    Schmidt spike removal (to be implemented).
    """
    """
    Python version of schmidt_spike_removal.m
    Faithfully reproduces MATLAB logic from:
    Schmidt et al., Physiol. Meas., 2010

    Parameters
    ----------
    signal : 1D numpy array
        Original PCG signal
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    despiked_signal : 1D numpy array
        Signal with spikes removed
    """

    signal = np.asarray(signal).flatten()
    original_len = len(signal)

    # --- Window size (500 ms) ---
    windowsize = int(round(fs / 2))

    # --- Trailing samples ---
    trailingsamples = original_len % windowsize

    # --- Reshape into windows (MATLAB column-wise) ---
    sampleframes = signal[:original_len - trailingsamples] \
        .reshape((windowsize, -1), order='F')

    # --- Maximum Absolute Amplitudes ---
    MAAs = np.max(np.abs(sampleframes), axis=0)

    # --- Main spike removal loop (MATLAB-style) ---
    while np.any(MAAs > np.median(MAAs) * 3):

        # Window with maximum MAA
        window_num = np.where(MAAs == np.max(MAAs))[0][0]

        # Spike position inside window
        spike_position = np.where(
            np.abs(sampleframes[:, window_num]) ==
            np.max(np.abs(sampleframes[:, window_num]))
        )[0][0]

        # Zero crossings
        zero_crossings = np.concatenate([
            np.abs(np.diff(np.sign(sampleframes[:, window_num]))) > 1,
            [0]
        ])

        # Spike start
        before = np.where(zero_crossings[:spike_position])[0]
        if before.size == 0:
            spike_start = 0
        else:
            spike_start = before[-1] + 1

        # Spike end
        zero_crossings[:spike_position] = 0
        after = np.where(zero_crossings)[0]
        if after.size == 0:
            spike_end = windowsize - 1
        else:
            spike_end = after[0]

        # Replace spike with small value
        sampleframes[spike_start:spike_end + 1, window_num] = 0.0001

        # Recalculate MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)

    # --- Reshape back to 1D ---
    despiked_signal = sampleframes.flatten(order='F')

    # --- Append trailing samples ---
    despiked_signal = np.concatenate([
        despiked_signal,
        signal[len(despiked_signal):]
    ])

    return despiked_signal
    

save_ind_table = True
fs_new = 2000
folder_path = "/home/sparc/Desktop/TH_alldata_fil"
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#os.chdir(base_path)

# Channel ordering (MATLAB → Python: convert to 0-based)
order_1  = [0, 1, 3, 4, 5, 6, 7, 10]
order_23 = [1, 3, 4, 5, 7, 6, 8, 10]

#files = os.listdir(base_path)

# Read subject list
data = pd.read_csv('REFERENCE_ALLROUNDS_ExclusionCriteria.csv',names = ['patient', 'abnormality','rounds'])
subs = data.iloc[:, 0].tolist()

# for ccc in range(6):
for ccc in range(0,1):
    ind_cell = []

    for s, sub in enumerate(subs):
        print(s,sub)
        # Get all files for current subject
        sub_files = [f for f in file_names if sub in f]
        sub_files.sort()
        
        data_concat = []
        len_array = [0]
        bad_ind = []

        for f, file in enumerate(sub_files):
            file_path = os.path.join(folder_path, file)
            print(file_path)
            # fs, data_wav1 = wavfile.read(file_path)
            data_wav1, fs = sf.read(file_path, dtype="float64")
            
            # Resample
            num_samples = int(len(data_wav1) * fs_new / fs)
            data_wav = resample(data_wav1, fs, fs_new)

            # Determine channel order
            if file[1] in ['c', 'v']:
                order = order_23
            else:
                order = order_1

            # Segment noisy intervals
            ind_medPow_HM_chan = segment_signal_noise_multi_MedPow(data_wav[:, order[ccc]], 1.5, fs_new, 2.5, 1, 0, 0)
            ind_medPow_NM = segment_signal_noise_multi_MedPow(data_wav[:, order[3]+8], 0.25, fs_new, 2.5, 1, 1, 0)

            # Spike removal
            # for chan in range(7):
            #     col = order[chan]
            #     temp_chan = data_wav[:, col]
            #     temp_chan[fs_new-200 : len(temp_chan)-1000] = schmidt_spike_removal(temp_chan[fs_new-200 : len(temp_chan)-1000], fs_new)
            #     data_wav[:, col] = temp_chan

            # Channel-wise concatenate
            data_concat.append(data_wav)

            # Update bad indices with previous lengths
            prev_len = sum(len_array)
            if ind_medPow_HM_chan.size > 0:
                bad_ind_HM_chan = ind_medPow_HM_chan + prev_len
                bad_ind.append(bad_ind_HM_chan)
            if ind_medPow_NM.size > 0:
                bad_ind_NM = ind_medPow_NM + prev_len
                bad_ind.append(bad_ind_NM)

            # Update lengths
            len_array.append(len(data_wav))

        # Flatten concatenated data
        data_concat = np.vstack(data_concat)

        # Calculate good indices
        chop = int(1.5 * fs_new) + 1
        good_ind = []
        x = 0
        for i in range(len(len_array)-1):
            good_ind.append(chop + sum(len_array[:i+1]))
            good_ind.append(sum(len_array[:i+2]) - chop)

        # Merge bad intervals
        if bad_ind:
            bad_ind_s = np.vstack(bad_ind)
            bad_ind_s = bad_ind_s[bad_ind_s[:,0].argsort()]  # sort by start
            merged_bad_ind = []
            current_range = bad_ind_s[0]

            for i in range(1, len(bad_ind_s)):
                this_range = bad_ind_s[i]
                if this_range[0] <= current_range[1] + 1:
                    current_range[1] = max(current_range[1], this_range[1])
                else:
                    merged_bad_ind.append(current_range)
                    current_range = this_range
            merged_bad_ind.append(current_range)
            merged_bad_ind = np.array(merged_bad_ind)
        else:
            merged_bad_ind = np.empty((0,2), dtype=int)

        # Adjust good indices by removing bad intervals
        new_good_ind = []
        for i in range(0, len(good_ind), 2):
            good_start = good_ind[i]
            good_end = good_ind[i+1]
            overlapping = merged_bad_ind[(merged_bad_ind[:,1] >= good_start) & (merged_bad_ind[:,0] <= good_end)]

            if len(overlapping) == 0:
                new_good_ind.extend([good_start, good_end])
            else:
                current_start = good_start
                for bad in overlapping:
                    bad_start = max(bad[0], good_start)
                    bad_end = min(bad[1], good_end)
                    if current_start < bad_start:
                        new_good_ind.extend([current_start, bad_start - 1])
                    current_start = bad_end + 1
                if current_start <= good_end:
                    new_good_ind.extend([current_start, good_end])

        # Store subject indices
        ind_cell.append([sub, np.array(new_good_ind, dtype=int)])

    # Save CSV
    ind_table = pd.DataFrame(ind_cell, columns=['subject', 'good_indices'])
    if save_ind_table:
        ind_table.to_csv(f'New_Ind_Table_chan{ccc+1}_NM4.csv', index=False)

