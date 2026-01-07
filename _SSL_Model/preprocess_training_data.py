"""
Script to segment, normlaise, and save ocean data for SSL model training

Author: Matthew Fynn

"""

import numpy as np
import pandas as pd
import os
import sys
# Add the parent directory to sys.path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)
import click
from processing.process import normalise_signal
from util.fileio import save_signal_wav, read_signal_wav

@click.group(context_settings={'show_default': True})
def cli(**kwargs):
    pass

@cli.command()
@click.option('--input_dir', '-i', required=True, help="input directory, already resamples and saved using MATLAB")
@click.option('--output_dir', '-o', required=True, help="output directory, where fragments will be saved")
@click.option('--frag_len', '-f', required = True, help = "fragment length", type = int)
@click.option('--num_frag', '-n', required = True, help = "number of fragments extracted from from each file", type = int)


#/media/matthew-fynn/TH/SSL_data/Kerguelen2018
#home/matthew-fynn/Desktop/Desktop/SSL_data/Ker2018_4s
def cli(input_dir, output_dir, frag_len, num_frag):
    pd.set_option('display.max_colwidth', None)
    wav_files = get_wav_files_to_dataframe(input_dir)
    # print(wav_files['File_Path'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, file in enumerate(wav_files['File_Path']):
        sig, fs = read_signal_wav(file)
        # print(len(sig)/fs)
        # input()
        segment_samples = int(fs*frag_len)
        total_samples = len(sig)
        step_size = max(1, total_samples//num_frag)
        for i in range(num_frag):
            start_index = i * step_size
            end_index = start_index + segment_samples
            if end_index <= total_samples:
                segment = sig[start_index:end_index]
                new_path = os.path.join(output_dir,file.split('/')[-1][0:-4]+ f'_f{i}.wav')
                save_signal_wav(segment, fs, new_path)
                del segment
            else:
                break
        
    #segment and save into 4 second fragments.
    #i wont use all the data, but will take x segments from each file

def get_wav_files_to_dataframe(input_directory):
    wav_files = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                wav_files.append({
                    'File_Path': full_path,
                    'Directory': root
                })

    # Convert to DataFrame
    df = pd.DataFrame(wav_files)
    return df


if __name__ == "__main__": 
    cli()