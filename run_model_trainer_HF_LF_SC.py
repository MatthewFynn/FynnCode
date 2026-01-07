"""
    run_model_trainer.py
    Author: Matthew Fynn

    Purpose: To create train, validate and test model (using class def)
    This script will combine a high frequency model (wav2vec_cnn, encodec_cnn etc) with my low-frequency model (unet2D, unet1D etc)
    The models are pre-trained and will be frozen - just mlp trained
    Fragment lengths will differ for each model, so may need to average embeddings from the model with smaller fragment length
    No Frag, and everything else will be tested with exhaused search.
"""
import numpy as np
import pandas as pd 
import os
import torch
from torch.utils.data import DataLoader
from data_proc.data_org import FilenameLabelDFCreator
import click
from tqdm.auto import tqdm
from data_proc.data_factory import FeatureVectorsDataset, FeatureVectorsDataset_multi, FeatureVectorsDataset_HYBRID
from models.svm_model import SVM_fit_predict
from util.fileio import average_results_and_save, average_results_and_save_multi, average_results_and_save_multi_svm, average_results_and_save_sc_svm, save_args_to_file
import json
from types import SimpleNamespace
from models.model_factory import (
    get_optimizer_and_scheduler,
    ModelFactory
)
from trainer_HYBRID import Trainer

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
@click.option('--channel', '-ch', required=True,type = int, help="which channels to look at")
@click.option('--indicie', '-ind', required=True, help="Indicie file csv")
@click.option('--numfolds', '-nf', required=True, type = int, help="number of folds in cv")
@click.option('--batch_size', '-bs', required=True, type = int, help="Batch size")
@click.option('--no_epoch', '-e', required=True, type = int, help="number of epochs")
@click.option('--model_code', '-m', required=True, help="model_code")
@click.option('--split', '-spl', required=True, help="split")
@click.option('--opt_code', '-oc', required=True, help="optimizer code")
# @click.option('--hf_mod', '-hf', required=True, help="path of hf model")
# @click.option('--lf_mod', '-lf', required=True, help="path of lf model")
@click.option('--lf_2d', '-lf2d', default=0, type = int, help="1d or 2d low frequency model")
@click.option('--hf_mux', '-hfm', default="hub", help="enc, w2v, hub or ope")
@click.option('--num_frag', '-nfrag', default=31, type = int, help="number of fragments to extract from each signal")
@click.option('--svm_act_sc', '-svm', default=-3, type = int, help="do SVM")
@click.option('--single_chan_flag', '-sc', default=1, type = int, help="do single channel training")
@click.option('--hidden', '-hs', default=512, type = int, help="hidden layer size of MLP")
@click.option('--mlp_flag', '-mlp', default=0, type = int, help="1 if multiple layers for multi channel MLP")
@click.option('--seed', '-s', default=0, type = int, help="seed")
@click.option('--ecg', '-ecg', default=0, type = int, required =True, help="include ecg")
@click.option('--version', '-v', required=True, type = int, help="version for saved model")
@click.option('--output_dir', '-o', required=True, help="output directory where models are saved")
@click.option('--dropout', '-dr', default=0.0, type = float, help="dropout in CNN layers")
@click.option('--notes', '-not', required=True, help="notes")
@click.option('--save_model_ten', '-sav', required=True, type=int, help="save model tensors")

def cli(reference, indir, indicie, seed, numfolds, 
        batch_size, no_epoch, ecg, split,
        num_frag, single_chan_flag, svm_act_sc, 
        hidden, mlp_flag, dropout, opt_code, model_code, version, output_dir, channel,
        notes,lf_2d,hf_mux, save_model_ten):
    # seed = 31 #CHANGE THIS AFTER
    # pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    aug_flag = 0
    channels=[channel]
    #chan1
    c1_path2d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl2d_cnn/adam_s{seed}/ver138'
    c1_path1d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl_cnn/adam_s{seed}/ver141'
    c1_pathEN=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/encodec_cnn/adamw_s{seed}/ver56'
    c1_pathW2=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/wav2vec_cnn/adam_s{seed}/ver53'
    c1_pathOP=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/opera_ce/adam_s{seed}/ver0'
    c1_pathHU=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/hubert_cnn/adam_s{seed}/ver0'


    #chan2
    c2_path2d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl2d_cnn/adam_s{seed}/ver294'
    c2_path1d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl_cnn/adam_s{seed}/ver372'
    c2_pathEN=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/encodec_cnn/adamw_s{seed}/ver173'
    c2_pathW2=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/wav2vec_cnn/adam_s{seed}/ver161'
    c2_pathOP=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/opera_ce/adam_s{seed}/ver32'
    c2_pathHU=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/hubert_cnn/adam_s{seed}/ver36'


    #chan3
    c3_path2d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl2d_cnn/adam_s{seed}/ver552'
    c3_path1d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl_cnn/adam_s{seed}/ver627'
    c3_pathEN=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/encodec_cnn/adamw_s{seed}/ver334'
    c3_pathW2=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/wav2vec_cnn/adam_s{seed}/ver323'
    c3_pathOP=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/opera_ce/adam_s{seed}/ver64'
    c3_pathHU=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/hubert_cnn/adam_s{seed}/ver69'


    #chan4
    c4_path2d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl2d_cnn/adam_s{seed}/ver1028'
    c4_path1d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl_cnn/adam_s{seed}/ver844'
    c4_pathEN=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/encodec_cnn/adamw_s{seed}/ver479'
    c4_pathW2=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/wav2vec_cnn/adam_s{seed}/ver443'
    c4_pathOP=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/opera_ce/adam_s{seed}/ver97'
    c4_pathHU=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/hubert_cnn/adam_s{seed}/ver100'

    #chan5
    c5_path2d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl2d_cnn/adam_s{seed}/ver1152'
    c5_path1d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl_cnn/adam_s{seed}/ver974'
    c5_pathEN=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/encodec_cnn/adamw_s{seed}/ver632'
    c5_pathW2=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/wav2vec_cnn/adam_s{seed}/ver608'
    c5_pathOP=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/opera_ce/adam_s{seed}/ver144'
    c5_pathHU=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/hubert_cnn/adam_s{seed}/ver130'


    #chan6
    c6_path2d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl2d_cnn/adam_s{seed}/ver1374'
    c6_path1d=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/unetssl_cnn/adam_s{seed}/ver1338'
    c6_pathEN=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/encodec_cnn/adamw_s{seed}/ver749'
    c6_pathW2=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/wav2vec_cnn/adam_s{seed}/ver755'
    c6_pathOP=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/opera_ce/adam_s{seed}/ver160'
    c6_pathHU=f'/home/tickingheart/dev/code/saved_IEEEPaper/saved{split}_HF_LF_FURTHUR3/hubert_cnn/adam_s{seed}/ver163'

    lf_mod = locals()[f"c{channel}_path2d"] if lf_2d == 1 else locals()[f"c{channel}_path1d"]
    
    if hf_mux == 'enc':
        hf_mod = locals()[f"c{channel}_pathEN"]
    elif hf_mux == 'w2v':
        hf_mod = locals()[f"c{channel}_pathW2"]
    elif hf_mux == 'hub':
        hf_mod = locals()[f"c{channel}_pathHU"]
    elif hf_mux == 'ope':
        hf_mod = locals()[f"c{channel}_pathOP"]
    
    print(lf_mod)
    print(hf_mod)
    # input('hello')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Dataset_dir = os.path.join(f'/home/{os.getlogin()}/Desktop/heart_data', indir)

    folds = pd.read_csv(reference)

    # df_indicies = pd.read_csv(f'{indicie}.csv')
    df_indicies = pd.read_csv(f'{indicie}_chan{channel}HM_4NM.csv')

    df_indicies["subject"] = df_indicies["subject"].astype(str)

    model_fact = ModelFactory(device=device)

    with open(f'{lf_mod}_cli_script_arguments.txt', "r") as f:
        config_lf = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    with open(f'{hf_mod}_cli_script_arguments.txt', "r") as f:
        config_hf = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # print("")
    print(config_lf.mod_fs, config_lf.low_f, config_lf.high_f, config_lf.frag_len, config_lf.hidden)
    print(config_hf.mod_fs, config_hf.low_f, config_hf.high_f, config_hf.frag_len, config_hf.hidden)
    max_frag_len = np.max([config_lf.frag_len, config_hf.frag_len])
    # input(max_frag_len)

    for f in tqdm(range(1,numfolds+1), ncols = 120):

        sp = f'split{f}'
        data_split = folds[['patient','abnormality',sp]]

        tr = data_split[data_split[sp] == 'train'][['patient','abnormality']]
        va = data_split[data_split[sp] == 'valid'][['patient','abnormality']]
        te = data_split[data_split[sp] == 'test'][['patient','abnormality']]

        tr['ind'] = tr['patient'].map(lambda x: df_indicies.loc[df_indicies["subject"] == x].iloc[:, 1:].dropna(axis=1).astype(int).values.flatten().tolist() if x in df_indicies["subject"].values else [])
        va['ind'] = va['patient'].map(lambda x: df_indicies.loc[df_indicies["subject"] == x].iloc[:, 1:].dropna(axis=1).astype(int).values.flatten().tolist() if x in df_indicies["subject"].values else [])
        te['ind'] = te['patient'].map(lambda x: df_indicies.loc[df_indicies["subject"] == x].iloc[:, 1:].dropna(axis=1).astype(int).values.flatten().tolist() if x in df_indicies["subject"].values else [])

        tr_set_lf = FilenameLabelDFCreator(tr, Dataset_dir, aug = aug_flag).segment_files_ind_HYBRID(no_seg=num_frag, seg_len_P=max_frag_len, seg_len_C = config_lf.frag_len, channels =channels, fs_new = config_lf.mod_fs, low=config_lf.low_f, high = config_lf.high_f, train_flag=1)
        va_set_lf = FilenameLabelDFCreator(va, Dataset_dir, aug = aug_flag).segment_files_ind_HYBRID(no_seg=num_frag, seg_len_P=max_frag_len, seg_len_C = config_lf.frag_len, channels =channels, fs_new = config_lf.mod_fs, low=config_lf.low_f, high = config_lf.high_f, train_flag=0)
        te_set_lf = FilenameLabelDFCreator(te, Dataset_dir, aug = 0).segment_files_ind_HYBRID(no_seg=num_frag, seg_len_P=max_frag_len, seg_len_C = config_lf.frag_len, channels =channels, fs_new = config_lf.mod_fs, low=config_lf.low_f, high = config_lf.high_f, train_flag=0)
        
        tr_set_hf = FilenameLabelDFCreator(tr, Dataset_dir, aug = aug_flag).segment_files_ind_HYBRID(no_seg=num_frag, seg_len_P=max_frag_len, seg_len_C = config_hf.frag_len, channels =channels, fs_new = config_hf.mod_fs, low=config_hf.low_f, high = config_hf.high_f, train_flag=1)
        va_set_hf = FilenameLabelDFCreator(va, Dataset_dir, aug = aug_flag).segment_files_ind_HYBRID(no_seg=num_frag, seg_len_P=max_frag_len, seg_len_C = config_hf.frag_len, channels =channels, fs_new = config_hf.mod_fs, low=config_hf.low_f, high = config_hf.high_f, train_flag=0)
        te_set_hf = FilenameLabelDFCreator(te, Dataset_dir, aug = 0).segment_files_ind_HYBRID(no_seg=num_frag, seg_len_P=max_frag_len, seg_len_C = config_hf.frag_len, channels =channels, fs_new = config_hf.mod_fs, low=config_hf.low_f, high = config_hf.high_f, train_flag=0)
        #SHAPE
        # [fragments, label, sub] where fragments is [childfrag][channel][frag]
        
        tr_set_lf_rn = tr_set_lf.rename(columns={'frag': 'frag_LF'})
        tr_set_hf_rn = tr_set_hf.rename(columns={'frag': 'frag_HF'})
        va_set_lf_rn = va_set_lf.rename(columns={'frag': 'frag_LF'})
        va_set_hf_rn = va_set_hf.rename(columns={'frag': 'frag_HF'})
        te_set_lf_rn = te_set_lf.rename(columns={'frag': 'frag_LF'})
        te_set_hf_rn = te_set_hf.rename(columns={'frag': 'frag_HF'})

        # Combine into one dataframe
        tr_set = pd.concat([tr_set_lf_rn[['frag_LF']], tr_set_hf_rn[['frag_HF']], tr_set_lf[['label', 'sub']]], axis=1)
        va_set = pd.concat([va_set_lf_rn[['frag_LF']], va_set_hf_rn[['frag_HF']], va_set_lf[['label', 'sub']]], axis=1)
        te_set = pd.concat([te_set_lf_rn[['frag_LF']], te_set_hf_rn[['frag_HF']], te_set_lf[['label', 'sub']]], axis=1)

        set_seed(seed)
        
        #this will return a df [[frag, label, sub],[frag, label, sub]]
        dataset_tr = FeatureVectorsDataset_HYBRID(df=tr_set, channels=[1], train_flag=0)
        dataset_va = FeatureVectorsDataset_HYBRID(df=va_set, channels=[1])
        dataset_te = FeatureVectorsDataset_HYBRID(df=te_set, channels=[1], test_flag=1)
        
        dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle = True, num_workers=2)
        dataloader_va = DataLoader(dataset_va, batch_size=int(batch_size), shuffle = False,num_workers=2)
        dataloader_te = DataLoader(dataset_te, batch_size=int(batch_size), shuffle = False,num_workers=2)

        # Initialize your model, dataloaders, loss function, optimizer, and device
        # set_seed(seed)
        config = {
                    "dropout": dropout,
                    "hidden_size": hidden,
                    "mlp_flag": mlp_flag,
                    "LF_dir": lf_mod,
                    "HF_dir": hf_mod,
                    "channel": channel,
                    "fold": f}
        ClassifierModel = model_fact.create_model(model_code=model_code, config=config)

        optimizer, lr_sche = get_optimizer_and_scheduler(ClassifierModel.parameters(), opt_code)
        criterion = ClassifierModel.criterion
        # Create an instance of the Trainer class
        # trainer = Trainer(ClassifierModel, dataloader_tr, dataloader_va, dataloader_te, criterion, optimizer, device, lr_sche)
        trainer = Trainer(ClassifierModel, dataloader_tr, dataloader_va, dataloader_te, criterion, optimizer, device)
        # Train the model
        trainer.train(epochs=no_epoch) #also tests the model (inbuilt in class definition)
        #save model here
        name = f'fold{f}' #name of folder
        out_save_file = os.path.join(output_dir,model_code, opt_code+f'_s{seed}', f'ver{version}', f'ch{channel}') #name of parent directory
        trainer.save_model(output_dir=out_save_file, name=name, save_param=save_model_ten)
        
        
        #SINGLE CHANNEL SVM
        if svm_act_sc != 0:
            #dataset_te does not change - only have to ensure no augments in the training set due to memory constraints
            dataset_tr = FeatureVectorsDataset_HYBRID(df=tr_set, train_flag=0)
            dataset_va = FeatureVectorsDataset_HYBRID(df=va_set, train_flag=0)

            svm_class_sc = SVM_fit_predict(ClassifierModel, svm_act_sc, device, channel, indir, 0)
            svm_class_sc.fit_svm([dataset_tr,dataset_va])
            svm_class_sc.predict_svm(dataset_te)

            out_save_file = os.path.join(output_dir,model_code, opt_code+f'_s{seed}', f'ver{version}', f'ch{channel}', name)
            svm_class_sc.save_results(out_save_file)


        del ClassifierModel
        del optimizer
        del trainer
        del lr_sche

    # #AVERAGE ALL RESULTS HERE AND STORE IN PARENT FILES
    parent_dir = os.path.join(output_dir,model_code,opt_code+f'_s{seed}',f'ver{version}')
    
    # average_results_and_save
    average_results_and_save(parent_dir, numfolds, channels, version)
    if svm_act_sc != 0:
        average_results_and_save_sc_svm(parent_dir, numfolds, channels, svm_act = svm_act_sc, ver = version)
    


    parent_dir2 = os.path.join(output_dir,model_code,opt_code+f'_s{seed}')
    save_args_to_file(parent_dir2, script_name=f'ver{version}_cli_script', hidden_options=["channel_list"])


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

