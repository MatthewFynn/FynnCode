"""
Data split functions for cross fold validation
"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from util.fileio import read_ticking_PCG
import numpy as np
from processing.filtering import resample
from processing.process import pre_process_pcg, pre_process_ecg
import math


def assign_split_crossfold(annotations, folds, random_state=None):

    # Ensure the 'abnormality' column exists in annotations
    if 'abnormality' not in annotations.columns:
        raise ValueError("The 'abnormality' column is required in the annotations DataFrame.")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = pd.DataFrame(index=annotations.index)
    fold_indices = list(skf.split(annotations, annotations['abnormality']))

    for fold, (train_val_index, test_index) in enumerate(fold_indices, start=1):
        # The test fold is the current fold
        # The validation fold is the next fold, with wrapping around to the start
        next_fold = (fold % folds)  # This gives the next fold index, wraps to 0 after the last fold
        _, valid_index = fold_indices[next_fold]
        # print(fold, next_fold, next_fold2)
        # Assign split names
        split_name = 'split1' if fold == 1 else f'split{fold}'
        splits[split_name] = "train"
        splits.loc[valid_index, split_name] = "valid"
        # splits.loc[valid_index2, split_name] = "valid" # ---------------------------------
        # splits.loc[valid_index3, split_name] = "valid" # ---------------------------------
        splits.loc[test_index, split_name] = "test"

    # Combine the original annotations with the new splits DataFrame
    new_annotations = pd.concat([annotations, splits], axis=1).sort_values(by='patient')

    return new_annotations, splits

def assign_split_crossfold_rounds(annotations, folds, random_state=None):
    # Ensure required columns exist
    if 'abnormality' not in annotations.columns:
        raise ValueError("The 'abnormality' column is required.")
    if 'rounds' not in annotations.columns:
        raise ValueError("The 'rounds' column is required.")

    # Combine 'abnormality' and 'rounds' into a new stratification label
    stratify_labels = annotations['abnormality'].astype(str) + "_" + annotations['rounds'].astype(str)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = pd.DataFrame(index=annotations.index)
    fold_indices = list(skf.split(annotations, stratify_labels))

    for fold, (train_val_index, test_index) in enumerate(fold_indices, start=1):
        next_fold = (fold % folds)
        _, valid_index = fold_indices[next_fold]
        
        split_name = f"split{fold}"
        splits[split_name] = "train"
        splits.loc[valid_index, split_name] = "valid"
        splits.loc[test_index, split_name] = "test"

    new_annotations = pd.concat([annotations, splits], axis=1).sort_values(by='patient')
    return new_annotations, splits

class FilenameLabelDFCreator:
    def __init__(self, df, directory, aug: int):
        """
        Initialize the class with DataFrame and directory.

        Parameters:
        - df (pd.DataFrame): Original DataFrame with columns 'patient' and 'label'.
        - directory (str): Path to the directory containing patient recordings.
        """
        if 'patient' not in df.columns or 'abnormality' not in df.columns:
            raise ValueError("DataFrame must contain 'patient' and 'abnormality' columns")
        
        self.df = df
        self.directory = directory
        self.aug = aug

    def create_filename_label_df(self): #THIS WILL NEVER BE USED _ I CONCATENATE ALL SIGNALS TOGETHER NOW
        """
        Create a DataFrame with filenames and corresponding labels based on patient identifiers.

        Returns:
        - pd.DataFrame: New DataFrame with columns 'filename' and 'label'.
        """
        new_data = []
        for patient in self.df['patient'].unique():
            label = self.df[self.df['patient'] == patient]['abnormality'].iloc[0]
            patient_files = [f for f in os.listdir(self.directory) if patient in f]
            if self.aug == 0:
                filtered_files = [f for f in patient_files if 'aug' not in f]
                patient_files = filtered_files
            
            for file in patient_files:
                new_data.append({'filename': os.path.join(self.directory,file), 'abnormality': label})

        return pd.DataFrame(new_data)
    
    def create_filename_label_df_one(self):
        """
        Create a DataFrame with filenames and corresponding labels based on patient identifiers,
        using only one 60s recording per patient.

        Returns:
        - pd.DataFrame: New DataFrame with columns 'filename' and 'label'.
        """
        new_data = []

        for patient in self.df['patient'].unique():
            label = self.df[self.df['patient'] == patient]['abnormality'].iloc[0]
            ind = self.df[self.df['patient'] == patient]['ind'].iloc[0]
            patient_files = [f for f in os.listdir(self.directory) if patient in f]
            # print(patient_files)
            # input('hi')
            if len(patient_files) == 0:
                print(patient)
                input('error')
            if patient_files:
                # Choose only the first file found for this patient
                Mother_file = patient_files[0][0::] #get the first 15 characters - unique recording
                # print(len(patient_files[0]))
                # print(Mother_file)
                # input('hi')
                patient_files_one = [g for g in os.listdir(self.directory) if Mother_file in g]
                if self.aug == 0:
                    filtered_files = [f for f in patient_files_one if 'aug' not in f]
                    patient_files_one = filtered_files
                    
                for file in patient_files_one:
                    new_data.append({'filename': os.path.join(self.directory,file), 'abnormality': label, 'ind': ind})
                #now get all of the fragments for this one 60s recording

        return pd.DataFrame(new_data)
    
    def segment_files_ind(self, no_seg: int, seg_len: float, channels: list, fs_new: int, low: int, high: int, train_flag: int):
        #no_seg - number of segments to fragmeht the signal into
        #seg_len - length of fragment
        #train_flag - for balancing number of segments in training set only
        #low and high are for pcg bandpass filtering cut off frequencies

        dataframe = self.create_filename_label_df_one()
        labels = dataframe['abnormality']
        # print(dataframe['filename'])
        # input()
        # print(dataframe[0:5])
        if train_flag == 1:
            label0_no = len(labels) - sum(labels)
            label1_no = sum(labels)

            if label0_no <= label1_no : #this means there are more CAD
                no_seg0 = int(label1_no/label0_no * no_seg)
                no_seg1 = no_seg 
            else:
                no_seg1 = int(label0_no/label1_no * no_seg)
                no_seg0 = no_seg 
        else:
            no_seg0, no_seg1 = no_seg, no_seg #validation and test sets - same fragments for both
        fragments = [] #this is for the FeatureVectorDataSet eventually. append a list [np.array(frag), label, subject number]
        frag_len = int(seg_len*fs_new)
        # print(no_seg1,no_seg0)
        # input('here')
        for idx, row in dataframe.iterrows():
            # print(row['filename'])
            # input()
            # print(row['abnormality'])

            sub = row['filename'].split('/')[-1][0:5]
            good_ind = row['ind']
            # print(len(good_ind))
            # input('hi')
            if len(good_ind) % 2 != 0:
                print('error - length of indicies must be even')
                input('ERROR')

            # print('')
            # print(sub)
            # print(good_ind)
            # input()
            keep = 1
            pcg_sig_all = [] #this will hold all of the relevent channels - still suitable for single channel data!!
            for c in channels:
                pcg_sig_concat, fs = read_ticking_PCG(row['filename'], channel=c, noise_mic=0, collection=-1, max_len=60) #This will be the concatenated signal
                pcg_sig_good = []
                for i in range(0, len(good_ind), 2): #THIS WILL EXTRACT THE GOOD AREAS OUT OF THE SIGNAL
                    pcg_sig_temp = pcg_sig_concat[good_ind[i]-1:good_ind[i+1]] #the indices are generated from matlab!!

                    
                    if good_ind[i+1]-good_ind[i]>seg_len*fs:
                        # input('here')
                        pcg_sig_temp = resample(pcg_sig_temp,fs,fs_new) #this wont affect signla if already sampled at fs_new
                        #FIXME ADDING PREPROCESSING HERE FOR THE ENTIRE GOOD PORTION OF SIGNAL
                        pcg_sig_temp_fil = pre_process_pcg(pcg_sig_temp, fs_new, low, high)
                        pcg_sig_good.append(pcg_sig_temp_fil)
                # input(len(pcg_sig_good))
                pcg_sig_all.append(pcg_sig_good)    
                if len(pcg_sig_good) == 0:
                    print('woah')
                    print('NO GOOD SIG FOR ',sub)
                    keep = 0
                 #THERE WILL BE MULTIPLE PARENT SEGMENTS PER CHANNEL NOW
            # print(channels)
            # print(len(pcg_sig_all))
            # print(len(pcg_sig_all[0]))
            if keep!=0:
                num_seg = no_seg0 if row['abnormality'] == 0 else no_seg1 #number of fragments to be extracted from the concatenated signal in total

                Parent_lengths = [len(pcg_sig_all[0][j]) for j in range(len(pcg_sig_all[0]))]
                # print(Parent_lengths)
                # print(sum(Parent_lengths))
                # input('HEY')

                #GET TOTAL NUMBER OF SEGMENTS TO EXTRACT FROM EACH PARENT SIGNAL
                exact_values = [Parent_lengths[j] / sum(Parent_lengths) * num_seg for j in range(len(Parent_lengths))] #store the amount of segments to extract from each Parent Segment
                # input(exact_values)
                Parent_segNum = [int(x) for x in exact_values]
                remaining = num_seg - sum(Parent_segNum)
                # Distribute the remaining segments to indices with the largest rounding errors
                errors = [(exact_values[j] - Parent_segNum[j], j) for j in range(len(Parent_lengths))]
                errors.sort(reverse=True, key=lambda x: x[0])  # Sort by largest rounding error
                # Assign remaining segments
                for i in range(remaining):
                    Parent_segNum[errors[i][1]] += 1

                # print(Parent_lengths)
                # print(Parent_segNum)
                # input('lol')

                for idx, num_seg in enumerate(Parent_segNum):
                    # xxx=1
                    pcg_sig_all_temp = [a[idx] for a in pcg_sig_all]
                    total_length = Parent_lengths[idx]
                    # print(num_seg, total_length, len(pcg_sig_all_temp))
                    required_coverage = frag_len * num_seg  # Total length needed to cover with fragments
                    step_size = (total_length - frag_len) // (num_seg - 1) if num_seg > 1 else 0  # Adjust for gaps if needed

                    start = 0

                    for i in range(num_seg):
                        end = start + frag_len
                        pcg_frag_all = [pcg[start:end] for pcg in pcg_sig_all_temp]  # Extracting fragments for all channels
                        # pcg_frag_all = [[1,9,idx] for pcg in pcg_sig_all_temp]  # for debugging

                        fragments.append([pcg_frag_all, row['abnormality'], sub])

                        # Adjust start position for next fragment
                        if num_seg > 1:
                            if required_coverage > total_length:  # Overlapping case
                                overlap = (required_coverage - total_length) // (num_seg-1)
                                start = end - overlap if end-overlap+frag_len < len(pcg_sig_all_temp[0]) else len(pcg_sig_all_temp[0])-frag_len #give more overlap to last segment so it is same number of samples
                            else:  # Gaps exist
                                start = min(start + step_size, len(pcg_sig_all_temp[0]) - frag_len)
                        # print(len(pcg_frag_all[0]), xxx)
                        # xxx+=1
                
                    # input('here')
        fragments_df = pd.DataFrame(fragments, columns = ["frag", "label", "sub"])
        return fragments_df
    
    def segment_files_ind_HYBRID(self, no_seg: int, seg_len_P: float, seg_len_C: float, channels: list, fs_new: int, low: int, high: int, train_flag: int):
        #no_seg - number of segments to fragmeht the signal into
        #seg_len_P - length of fragment (max of HF or LF model)
        #seg_len_C - length of fragment (frag length of actual model)
        #train_flag - for balancing number of segments in training set only
        #low and high are for pcg bandpass filtering cut off frequencies

        dataframe = self.create_filename_label_df_one()
        labels = dataframe['abnormality']
        
        # print(dataframe[0:5])
        if train_flag == 1:
            label0_no = len(labels) - sum(labels)
            label1_no = sum(labels)

            if label0_no <= label1_no : #this means there are more CAD
                no_seg0 = int(label1_no/label0_no * no_seg)
                no_seg1 = no_seg 
            else:
                no_seg1 = int(label0_no/label1_no * no_seg)
                no_seg0 = no_seg 
        else:
            no_seg0, no_seg1 = no_seg, no_seg #validation and test sets - same fragments for both
        fragments = [] #this is for the FeatureVectorDataSet eventually. append a list [np.array(frag), label, subject number]
        frag_len = int(seg_len_P*fs_new)

        for idx, row in dataframe.iterrows():
            # print(row['filename'])
            # input()
            # print(row['abnormality'])

            sub = row['filename'].split('/')[-1][0:5]
            good_ind = row['ind']
            if len(good_ind) % 2 != 0:
                print('error - length of indicies must be even')
                input('ERROR')

            # print('')
            # print(sub)
            # print(good_ind)
            # input()
            keep = 1
            pcg_sig_all = [] #this will hold all of the relevent channels - still suitable for single channel data!!
            for c in channels:
                pcg_sig_concat, fs = read_ticking_PCG(row['filename'], channel=c, noise_mic=0, collection=-1, max_len=60) #This will be the concatenated signal
                pcg_sig_good = []
                for i in range(0, len(good_ind), 2): #THIS WILL EXTRACT THE GOOD AREAS OUT OF THE SIGNAL
                    # print(good_ind[i],good_ind[i+1])
                    pcg_sig_temp = pcg_sig_concat[good_ind[i]-1:good_ind[i+1]] #the indices are generated from matlab!!
                    # print(good_ind[i],good_ind[i+1],len(pcg_sig_temp))
                    # print(len(pcg_sig_temp))
                    
                    if good_ind[i+1]-good_ind[i]>seg_len_P*fs:
                        # input('here')
                        pcg_sig_temp = resample(pcg_sig_temp,fs,fs_new) #this wont affect signla if already sampled at fs_new
                        #FIXME ADDING PREPROCESSING HERE FOR THE ENTIRE GOOD PORTION OF SIGNAL
                        pcg_sig_temp_fil = pre_process_pcg(pcg_sig_temp, fs_new, low, high)
                        pcg_sig_good.append(pcg_sig_temp_fil)
                pcg_sig_all.append(pcg_sig_good)    
                if len(pcg_sig_good) == 0:
                    print('woah')
                    print('NO GOOD SIG FOR ',sub)
                    keep = 0
                 #THERE WILL BE MULTIPLE PARENT SEGMENTS PER CHANNEL NOW
            # print(channels)
            # print(len(pcg_sig_all))
            # print(len(pcg_sig_all[0]))
            if keep!=0:
                num_seg = no_seg0 if row['abnormality'] == 0 else no_seg1 #number of fragments to be extracted from the concatenated signal in total

                Parent_lengths = [len(pcg_sig_all[0][j]) for j in range(len(pcg_sig_all[0]))]
                # print(Parent_lengths)
                # print(sum(Parent_lengths))
                # input('HEY')

                #GET TOTAL NUMBER OF SEGMENTS TO EXTRACT FROM EACH PARENT SIGNAL
                exact_values = [Parent_lengths[j] / sum(Parent_lengths) * num_seg for j in range(len(Parent_lengths))] #store the amount of segments to extract from each Parent Segment
                Parent_segNum = [int(x) for x in exact_values]
                remaining = num_seg - sum(Parent_segNum)
                # Distribute the remaining segments to indices with the largest rounding errors
                errors = [(exact_values[j] - Parent_segNum[j], j) for j in range(len(Parent_lengths))]
                errors.sort(reverse=True, key=lambda x: x[0])  # Sort by largest rounding error
                # Assign remaining segments
                for i in range(remaining):
                    Parent_segNum[errors[i][1]] += 1

                # print(Parent_lengths)
                # print(Parent_segNum)

                for idx, num_seg in enumerate(Parent_segNum):
                    # xxx=1
                    pcg_sig_all_temp = [a[idx] for a in pcg_sig_all]
                    total_length = Parent_lengths[idx]
                    # print(num_seg, total_length, len(pcg_sig_all_temp))
                    required_coverage = frag_len * num_seg  # Total length needed to cover with fragments
                    step_size = (total_length - frag_len) // (num_seg - 1) if num_seg > 1 else 0  # Adjust for gaps if needed

                    start = 0

                    for i in range(num_seg):
                        end = start + frag_len
                        pcg_frag_all = [pcg[start:end] for pcg in pcg_sig_all_temp]  # Extracting fragments for all channels
                        # pcg_frag_all = [[1,9,idx] for pcg in pcg_sig_all_temp]  # for debugging

                        if seg_len_P == seg_len_C:
                            fragments.append([[pcg_frag_all], row['abnormality'], sub]) #made pcg_frag_all in a list size 1 as the others will have list size >1
                        else: #This gets extracts smaller segments so it is compatible with the model (only if model seg len is less than parent seg len)
                            current_len = len(pcg_frag_all[0]*fs_new)
                            require_len = int(seg_len_C*fs_new) 
                            no_child_segs = int(np.ceil(current_len/require_len))
                            jmp = (current_len-require_len) // (no_child_segs-1)
                            s = 0
                            pcg_frag_all_2 = []
                            for __ in range(no_child_segs):
                                e = s + require_len
                                pcg_child = [pcg[s:e] for pcg in pcg_frag_all]
                                pcg_frag_all_2.append(pcg_child)
                                s = s+jmp
                            # print(no_child_segs)
                            # print(pcg_frag_all[0][0:5])
                            # print(pcg_frag_all_2[0][0][0:5])
                            # print(pcg_frag_all[0][-5::])
                            # print(pcg_frag_all_2[-1][0][-5::])
                            # input()

                            fragments.append([pcg_frag_all_2, row['abnormality'], sub]) #pcg_frag_all_2 will deffs be a list

                        # Adjust start position for next fragment
                        if num_seg > 1:
                            if required_coverage > total_length:  # Overlapping case
                                overlap = (required_coverage - total_length) // (num_seg-1)
                                start = end - overlap if end-overlap+frag_len < len(pcg_sig_all_temp[0]) else len(pcg_sig_all_temp[0])-frag_len #give more overlap to last segment so it is same number of samples
                            else:  # Gaps exist
                                start = min(start + step_size, len(pcg_sig_all_temp[0]) - frag_len)
                        # print(len(pcg_frag_all[0]), xxx)
                        # xxx+=1
                
                    # input('here')
        fragments_df = pd.DataFrame(fragments, columns = ["frag", "label", "sub"])

        return fragments_df
        
        

    def create_df(self, code):
        """
        Choose which function to use based on the provided code.

        Parameters:
        - code (str): Code indicating which function to use ('all' or 'one').

        Returns:
        - pd.DataFrame: Resulting DataFrame based on chosen method.
        """
        if code == 'all':
            return self.create_filename_label_df()
        elif code == 'one':
            return self.create_filename_label_df_one()
        else:
            raise ValueError("Invalid code. Use 'all' or 'one'.")