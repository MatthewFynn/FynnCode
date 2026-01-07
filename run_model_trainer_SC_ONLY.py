"""
    run_model_trainer.py
    Author: Matthew Fynn

    Purpose: To create train, validate and test model (using class def)
"""
import numpy as np
import pandas as pd 
import os
import torch
from torch.utils.data import DataLoader
from data_proc.data_org import assign_split_crossfold, assign_split_crossfold_rounds, FilenameLabelDFCreator
import click
from tqdm.auto import tqdm
from data_proc.data_factory import  FeatureVectorsDataset_noWav
from util.fileio import average_results_and_save, save_args_to_file
import pickle
from models.model_factory import (
    get_optimizer_and_scheduler,
    ModelFactory
)
from trainer import Trainer

@click.group(context_settings={'show_default': True})
def cli(**kwargs):
    pass
###########################################%%%%%%%%%%%%%%%%%
# 22,27,16 --34
# 22,27,16 -- 34
# 22,26,16 -- 33
# 22,26,16 -- 31
# 22,26,16 --32
@cli.command()
@click.option('--reference', '-r', required=True, help="patient, label csv file")
@click.option('--indir', '-i', required=True, help="input directory of data")
@click.option('--indicie', '-ind', required=True, help="Indicie file csv")
@click.option('--channels', '-ch', required=True,type = list[int], help="which channels to look at")
@click.option('--numfolds', '-nf', required=True, type = int, help="number of folds in cv")
@click.option('--batch_size', '-bs', required=True, type = int, help="Batch size")
@click.option('--no_epoch', '-e', required=True, type = int, help="number of epochs")
@click.option('--opt_code', '-oc', required=True, help="optimizer code")
@click.option('--model_code', '-m', required=True, help="model_code")
@click.option('--num_cnn_layers', '-nl', default=2, type = int, help="number of CNN layers used in encoder")
@click.option('--dropout', '-dr', default=0.0, type = float, help="dropout in CNN layers")
@click.option('--frag_len', '-fl', default=2, type = float, help="frag_len: only applies to segmenting in training set")
@click.option('--num_frag', '-nfrag', default=31, type = int, help="number of fragments to extract from each signal")
@click.option('--hidden', '-hs', default=512, type = int, help="hidden layer size of MLP")
@click.option('--mlp_flag', '-mlp', default=0, type = int, help="1 if multiple layers for multi channel MLP")
@click.option('--seed', '-s', default=0, type = int, help="seed")
@click.option('--version', '-v', required=True, type = int, help="version for saved model")
@click.option('--mod_fs', '-fs', required=True, type = int, help="include augments")
@click.option('--output_dir', '-o', required=True, help="output directory where models are saved")
@click.option('--epoch_ssl', '-essl', required=True, default = 'ep69', help="which checkpoint to use for ssl model")
@click.option('--trained_ssl_path', '-tssl', required=True, default = 'ep69', help="which checkpoint to use for ssl model")
@click.option('--low_f', '-flow', default = 10, type = int, help="low cutoff freq for pcg bandpass")
@click.option('--high_f', '-fhigh', default = 900, type = int, help="high cutoff freq for pcg bandpass")
@click.option('--notes', '-not', required=True, help="notes")
@click.option('--save_model_ten', '-sav', required=True, type=int, help="save model tensors")

def cli(reference, indir, indicie, channels, seed, numfolds, 
        batch_size, no_epoch, num_cnn_layers, frag_len, 
        num_frag ,hidden, mlp_flag, model_code,trained_ssl_path, epoch_ssl,
        dropout, opt_code, version, output_dir,
        mod_fs, notes, low_f, high_f, save_model_ten):
    # seed = 31 #CHANGE THIS AFTER
    pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_rows', None)
    aug_flag = 0
    include_females = 0

    channels = [int(a) for a in channels]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Dataset_dir = os.path.join(f'/home/{os.getlogin()}/Desktop/heart_data', indir)
    #convert to dataframe and call function to divide into k folds
    # name_lab = pd.read_csv(reference, names = ['patient', 'abnormality','rounds'])

    folds = pd.read_csv(reference)

    for single_chan in channels:
        df_indicies = pd.read_csv(f'{indicie}_chan{single_chan}_NM4.csv')
        # df_indicies = pd.read_csv(f'{indicie}.csv')
        df_indicies["subject"] = df_indicies["subject"].astype(str)

        model_fact = ModelFactory(device=device)
        model_fs = mod_fs


        for f in tqdm(range(1,numfolds+1), ncols = 120):

            sp = f'split{f}'
            data_split = folds[['patient','abnormality',sp]]

            tr = data_split[data_split[sp] == 'train'][['patient','abnormality']]
            va = data_split[data_split[sp] == 'valid'][['patient','abnormality']]
            te = data_split[data_split[sp] == 'test'][['patient','abnormality']]

            tr['ind'] = tr['patient'].map(lambda x: df_indicies.loc[df_indicies["subject"] == x].iloc[:, 1:].dropna(axis=1).astype(int).values.flatten().tolist() if x in df_indicies["subject"].values else [])
            va['ind'] = va['patient'].map(lambda x: df_indicies.loc[df_indicies["subject"] == x].iloc[:, 1:].dropna(axis=1).astype(int).values.flatten().tolist() if x in df_indicies["subject"].values else [])
            te['ind'] = te['patient'].map(lambda x: df_indicies.loc[df_indicies["subject"] == x].iloc[:, 1:].dropna(axis=1).astype(int).values.flatten().tolist() if x in df_indicies["subject"].values else [])

            all_chan = [single_chan]
            tr_set = FilenameLabelDFCreator(tr, Dataset_dir, aug = aug_flag).segment_files_ind(no_seg=num_frag, seg_len=frag_len, channels = all_chan, fs_new = model_fs, low=low_f, high = high_f, train_flag=1)
            va_set = FilenameLabelDFCreator(va, Dataset_dir, aug = aug_flag).segment_files_ind(no_seg=num_frag, seg_len=frag_len, channels = all_chan, fs_new = model_fs, low=low_f, high = high_f, train_flag=0)
            te_set = FilenameLabelDFCreator(te, Dataset_dir, aug = 0).segment_files_ind(no_seg=num_frag, seg_len=frag_len, channels = all_chan, fs_new = model_fs, low=low_f, high = high_f, train_flag=0)
            # print(te_set['frag'].iloc[0])
            # input('hi')
            # with open("te_set_data_confirmed2.pkl", "wb") as f:   # "wb" = write in binary mode
            #     pickle.dump(te_set, f)      
            # input('hello')
           
            set_seed(seed)
            #this will return a df [[frag, label, sub],[frag, label, sub]]
            #FIXME - this is where the 'frag' part will be based upon the indicies
            dataset_tr = FeatureVectorsDataset_noWav(df=tr_set, channels=[1], train_flag=0)
            dataset_va = FeatureVectorsDataset_noWav(df=va_set, channels=[1],)
            dataset_te = FeatureVectorsDataset_noWav(df=te_set, channels=[1], test_flag=1)
            
            dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle = True, num_workers=2)
            dataloader_va = DataLoader(dataset_va, batch_size=int(batch_size), shuffle = False,num_workers=2)
            dataloader_te = DataLoader(dataset_te, batch_size=int(batch_size), shuffle = False,num_workers=2)

            # Initialize your model, dataloaders, loss function, optimizer, and device
            # set_seed(seed)
            config = {"fs": model_fs,
                        "num_cnn_layers": num_cnn_layers,
                        "dropout": dropout,
                        "signal_len_t": frag_len,
                        "hidden_size": hidden,
                        "mlp_flag": mlp_flag,
                        "checkpoint": epoch_ssl,
                        "trained_path": trained_ssl_path} #for mfcc model only
            ClassifierModel = model_fact.create_model(model_code=model_code, config=config)

            optimizer, lr_sche = get_optimizer_and_scheduler(ClassifierModel.parameters(), opt_code)
            criterion = ClassifierModel.criterion
            # Create an instance of the Trainer class
            # trainer = Trainer(ClassifierModel, dataloader_tr, dataloader_va, dataloader_te, criterion, optimizer, device, lr_sche)
            trainer = Trainer(ClassifierModel, dataloader_tr, dataloader_va, dataloader_te, criterion, optimizer, device)
            # Train the model
            print(f'fold{f} channel{single_chan}')
            trainer.train(epochs=no_epoch) #also tests the model (inbuilt in class definition)
            #save model here
            name = f'fold{f}' #name of folder
            out_save_file = os.path.join(output_dir,model_code, opt_code+f'_s{seed}', f'ver{version}', f'ch{single_chan}') #name of parent directory
            trainer.save_model(output_dir=out_save_file, name=name, save_param=save_model_ten)
            
        


            del ClassifierModel
            del optimizer
            del trainer
            del lr_sche


    # #AVERAGE ALL RESULTS HERE AND STORE IN PARENT FILES
    parent_dir = os.path.join(output_dir,model_code,opt_code+f'_s{seed}',f'ver{version}')
    

    average_results_and_save(parent_dir, numfolds, channels, version)
   
    parent_dir2 = os.path.join(output_dir,model_code,opt_code+f'_s{seed}')
    save_args_to_file(parent_dir2, script_name=f'ver{version}_ch{single_chan}_cli_script', hidden_options=["channel_list"])


import random
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    cli()

