"""
    data_factory.py
    Author: Matthew Fynn

    Purpose: Builds the dataset for the dataloader
"""


import torch
import os
from torch.utils.data import  Dataset
from processing.filtering import znormalise_signal, resample
from util.fileio import read_ticking_PCG
import numpy as np
from processing.filtering import  normalise_signal
from processing.augmentation import RandomStretch, RandomTimeFreqMask
from processing.process import normalise_array_length, normalise_2d_array_length
from typing import Optional


class FeatureVectorsDataset_noWav(Dataset):
    def __init__(self, df, channels, test_flag: Optional[int] = 0, train_flag: Optional[int] = 0):
        self.df = df #this will now have cols 'frag':[list of channel frags], 'label':[int label]: and 'sub':[subject number]
        self.test_flag = test_flag
        self.train_flag = train_flag
        self.fragments = df['frag']
        self.labels = df['label']
        self.sub = df['sub']
        self.channels = channels

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        channel_frag = self.fragments.iloc[idx] #This will have the multichannel (or single) fragments
        label = self.labels.iloc[idx]
        
        if len(self.channels)>1:
            # Create an empty array for multi-channel signals
            multi_sig = np.zeros((len(channel_frag[0]), len(self.channels)))
            for i, c in enumerate(self.channels):
                sig = channel_frag[c-1] #minus 1 as pyhton index starts at 0
                multi_sig[:,i] = normalise_signal(sig)
        else:
            multi_sig = normalise_signal(channel_frag[self.channels[0]-1])
            # multi_sig = normalise_signal(channel_frag[0]) #FIXME THIS WILL WORK FOR SINGLE CHANNEL DATA

        if self.train_flag:
            multi_sig = self.augment(multi_sig, self.fs_new)
            multi_sig = normalise_2d_array_length(multi_sig, self.fs_new*2)[0]
            #normalise again?

        if self.test_flag:
            return torch.tensor(multi_sig.copy()).float(), torch.tensor(label), self.sub.iloc[idx]
        else:
            return torch.tensor(multi_sig.copy()).float(), torch.tensor(label)

    def augment(self, sig, fs):
        sig = RandomStretch(fs=fs).__call__(sig)
        sig = RandomTimeFreqMask(thickness=0.09, fs=fs).__call__(sig)
        sig = RandomTimeFreqMask(thickness=0.09 / 2, fs=fs).__call__(sig)
        return sig
    
class FeatureVectorsDataset_HYBRID(Dataset):
    """
    Expects df to have columns:
      - 'frag_LF': list of fragments; each fragment is list/array of channels [ch0, ch1, ...]
      - 'frag_HF': same structure as frag_LF
      - 'label'  : int
      - 'sub'    : subject id
    channels: list of 1-based channel indices to include, e.g. [1,3,5]
    Returns per item:
      - (LF, HF, label) or (LF, HF, label, sub) if test_flag
      where LF/HF have shape (num_frag, T, C)
    """
    def __init__(self, df, channels, test_flag: int = 0, train_flag: int = 0):
        self.df = df
        self.test_flag = int(test_flag)
        self.train_flag = int(train_flag)
        self.fragments_LF = df['frag_LF']
        self.fragments_HF = df['frag_HF']
        self.labels = df['label']
        self.sub = df['sub']

        self.channels = channels

    def __len__(self):
        return len(self.df)

    def _build_multi(self, channel_frag_list):
        """
        channel_frag_list: list of fragments for this sample.
          Each fragment is a list/array of per-channel 1D arrays: [ch0_sig, ch1_sig, ...].
        Returns: ndarray of shape (num_frag, T, C)
        """
        per_frag_arrays = []

        # Use 0-based indices internally
        use_idx = [c - 1 for c in self.channels]

        for frag in channel_frag_list:
            # frag[k] is signal for channel k (0-based)
            # Collect selected channels
            ch_sigs = []
            for k in use_idx:
                sig = np.asarray(frag[k], dtype=np.float32)
                # sig = normalise_signal(sig)
                ch_sigs.append(sig)

            # Stack channels to shape (T, C)
            frag_tc = np.stack(ch_sigs, axis=1)  # (T, C)
            per_frag_arrays.append(frag_tc)

        # Now stack fragments: (num_frag, T, C)
        multi = np.stack(per_frag_arrays, axis=0).astype(np.float32)
        return multi

    def __getitem__(self, idx):
        channel_frag_LF = self.fragments_LF.iloc[idx]
        channel_frag_HF = self.fragments_HF.iloc[idx]
        label = int(self.labels.iloc[idx])

        # Build LF & HF tensors: (num_frag, T, C)
        multi_sig_HF = self._build_multi(channel_frag_HF)
        multi_sig_LF = self._build_multi(channel_frag_LF)

        # Convert to torch
        LF = torch.from_numpy(multi_sig_LF)  # (N, T, C), float32
        HF = torch.from_numpy(multi_sig_HF)  # (N, T, C), float32
        y = torch.tensor(label)

        if self.test_flag:
            return LF, HF, y, self.sub.iloc[idx]
        else:
            return LF, HF, y

    
