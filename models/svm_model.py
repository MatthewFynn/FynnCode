import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import svm
from data_proc.data_factory import FeatureVectorsDataset_multi
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, matthews_corrcoef
import os, json
from collections import Counter

class SVM_fit_predict():

    def __init__ (self, model, seq, device, chan_list, dir, model_fs,**kwargs):
        self.model = model.to(device) #neural network model (with sequential MLP layers at end)
        self.seq = seq #index of layer to be accessed as SVM features
        self.chan_list = chan_list
        self.dir = dir #directory where data is stored
        self.svm = svm.SVC()
        self.device = device
        self.model_fs = model_fs

    
    def get_activations(self, dataset):
        hook_handle = self.model.classifier[self.seq].register_forward_hook(self.save_activation)
        
        dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)

        act_list = []
        label_list = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                act_list.append(self.activations)
                if len(target.shape) > 1:
                    target_labels = torch.argmax(target, dim=1)
                else:
                    target_labels = target
                label_list.append(target_labels.cpu().numpy())

            hook_handle.remove()
        act_array = np.concatenate(act_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        return act_array, labels
    
    def get_activations_test(self, dataset):
        hook_handle = self.model.classifier[self.seq].register_forward_hook(self.save_activation)
        
        #can alter set to remove aug if needed .. later
        dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)

        act_list = []
        label_list = []
        sub_ids = []
        with torch.no_grad():
            for batch_idx, (data, target, sub_id) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                act_list.append(self.activations)

                if len(target.shape) > 1:
                    target_labels = torch.argmax(target, dim=1)
                else:
                    target_labels = target
                label_list.append(target_labels.cpu().numpy())
                
                sub_ids.extend(sub_id)

            hook_handle.remove()
        act_array = np.concatenate(act_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        return act_array, labels, sub_ids

    def save_activation(self, module, input, output):
        activation = output
        self.activations = activation.cpu().detach().numpy()

    def fit_svm(self, dataset):
        try:
            for idx, indiv_dataset in enumerate(dataset):
                if idx == 0:
                    feat, lab = self.get_activations(dataset=indiv_dataset)
                else:
                    feat_i, lab_i = self.get_activations(dataset=indiv_dataset)
                    feat = np.concatenate((feat, feat_i))
                    lab = np.concatenate((lab, lab_i))
        except IndexError:
            feat, lab = self.get_activations(dataset=dataset)
        self.svm.fit(feat, lab)
        # print(feat.shape, lab.shape)

    def predict_svm(self, dataset):
        feat, lab, sub_ids = self.get_activations_test(dataset)
        predictions = self.svm.predict(feat)
        # print('pred shape: ', predictions.shape)
        accuracy = accuracy_score(lab, predictions)
        sensitivity = recall_score(lab, predictions, pos_label=1)
        # Specificity (True negative rate)
        tn, fp, fn, tp = confusion_matrix(lab, predictions).ravel()
        specificity = tn / (tn + fp)
        # Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(lab, predictions)
        # Store the metrics in a dictionary
        self.res_dict = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'MCC': mcc
        }

        self.frag_to_sub_metrics(sub_ids, predictions=predictions, labels=lab)
        

    def frag_to_sub_metrics(self, subjects, predictions, labels):
        df_frag = pd.DataFrame({
            'subject': subjects,
            'prediction': predictions,
            'label': labels
        })
        # Group by subject and apply majority voting
        def majority_vote(series):
            return Counter(series).most_common(1)[0][0]
        
        subject_predictions = df_frag.groupby('subject')['prediction'].agg(majority_vote)
        subject_labels = df_frag.groupby('subject')['label'].agg(majority_vote)

        # Combine into a final DataFrame
        subject_df = pd.DataFrame({
            'subject': subject_predictions.index,
            'subject_prediction': subject_predictions.values,
            'subject_label': subject_labels.values
        })
        y_true = subject_df['subject_label']
        y_pred = subject_df['subject_prediction']

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        acc = (tp + tn) / (tp + tn + fp + fn)
        sen = tp / (tp + fn)  # Also called recall or TPR
        spe = tn / (tn + fp)  # True negative rate
        mcc = matthews_corrcoef(y_true, y_pred)
        self.res_dict_sub = {
            "accuracy": acc,
            "sensitivity": sen, "specificity": spe, "MCC": mcc
        }
    
    def save_results(self, output_dir):
        txt_name = os.path.join(output_dir, f'svm{self.seq}.txt')
        with open(txt_name, "w") as file:
            json.dump({"results_frag": self.res_dict, "results_sub": self.res_dict_sub}, file, indent = 4)

        #change to also calculate the subject based metrics + save



