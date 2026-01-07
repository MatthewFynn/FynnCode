"""
    data_aug_save_concatenated.py
    Author : Matthew Fynn

    Loads the data, Augements the data
    Saves 60s signal in directoyr - no segmentation
    **********All 60s recordinsg have been concatenated in MATLAB ********
"""

import os
import numpy as np
import pandas as pd
from util.fileio import read_ticking_PCG, create_multi_wav, save_ticking_signals
from data_proc.data_org import  FilenameLabelDFCreator
from processing.process import pre_process_pcg, pre_process_ecg, adjust_length
import click
from processing.filtering import noise_canc, interpolate_nans, resample
from processing.augmentation import (
    # augment_pcg,
    augment_multi_pcg,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from util.paths import EPHNOGRAM

@click.group(context_settings={'show_default': True})
def cli(**kwargs):
    pass

@cli.command()
@click.option('--label_file', '-l', required=True, help="Reference")
@click.option('--outfile', '-o', default = '', help="outfile")

#/media/matthew-fynn/TH/TH_alldata
def cli(label_file, outfile):
    pd.set_option('display.max_colwidth', None)
    indir = f'/home/{os.getlogin()}/Desktop/TH_data_joined_MATLAB_SpRe/'
    outdir = f'/home/{os.getlogin()}/Desktop/heart_data/{outfile}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    name_lab = pd.read_csv(label_file, names = ['patient', 'abnormality','round']) #data_fram version of label file
    filename_label = FilenameLabelDFCreator(name_lab, indir, aug = 0).create_filename_label_df() 
    filename_label_fil = filename_label[filename_label['filename'].str.contains('.wav', na=False)] #only .wav files

    filenames = filename_label_fil['filename']
    labels = filename_label_fil['abnormality']

    num_workers = os.cpu_count() - 3
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx in range(len(filenames)):
            filename = filenames.iloc[idx]
            label = labels.iloc[idx]

            futures.append(executor.submit(augment_and_save_multi, filename, outdir))
        
        for future in tqdm(as_completed(futures), total = len(futures), ncols=120):
            try:
                future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")
                executor.shutdown(wait=False, cancel_futures=True)
                raise e
    # print(filenames)
    # for idx in range(len(filenames)):
    #     filename = filenames.iloc[idx]
    #     label = labels.iloc[idx]
        augment_and_save_multi(filename, outdir)
        


def augment_and_save_multi(filename, outdir):
    if filename.split('/')[-1][1] == 'v':
        collect = 2
    elif filename.split('/')[-1][1] == 'c':
        collect = 2
    else:
        collect = 1
    pcg_multi = [] #initilaise multi variable
    for chan in range(1,8):
        pcg_chan, fs = read_ticking_PCG(filename=filename, channel= chan, noise_mic=False, collection=collect, max_len=60) #read signal for each channel
        pcg_chan = interpolate_nans(pcg_chan)

        pcg_filt = pcg_chan

        # pcg_multi.append(resample(pcg_filt, fs, fs_new)) #append to list
        pcg_multi.append(pcg_filt) #append to list
    ecg, _ = read_ticking_PCG(filename=filename, channel= 8, noise_mic=False, collection=collect, max_len=60) #read signal for each channel
    ecg = pre_process_ecg(interpolate_nans(ecg),_)
    ecg = resample(ecg, _, 80)
    ecg = resample(ecg, 80, _)
    ecg =  adjust_length(ecg, len(pcg_filt))
    pcg_multi.append(ecg) #append to list

    ticking_wav_orig = create_multi_wav(pcg_multi,len(pcg_multi))
    filename_new = filename.split('/')[-1]
    path_filename_new = os.path.join(outdir, filename_new)
    save_ticking_signals(ticking_wav=ticking_wav_orig, fs=fs, path=path_filename_new)

if __name__ == "__main__": 
    cli()
