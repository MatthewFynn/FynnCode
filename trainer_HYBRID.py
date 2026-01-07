"""
    trainer.py
    Author: Matthew Fynn

    Purpose: train, val and test all in one class
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import random
import numpy as np
import os
import json
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import pandas as pd
from collections import Counter
from util.metrics import calculate_metrics
import copy

def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device,  scheduler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler


    def train_epoch(self):
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        all_targets = []
        all_predictions = []


        for batch_idx, (data1, data2, target) in tqdm(enumerate(self.train_loader)):
            # print("")
            # print(len(data))
            # print(data[0].shape)
            # print(data[1].shape)
            # print(target.shape)

            # print(data1[2,0,:,0])
            # print(data2[2,0,:,0])

            data1,data2, target = data1.to(self.device),data2.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()  # Zero the gradients
            outputs = self.model.forward_hybrid(data1,data2)  # #FIXME NEED TO MAKE MODEL NOW
            loss = self.criterion(outputs, target)  # Compute loss
            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # the magnitude of the gradient won't be bigger than 1.0
            self.optimizer.step()  # Update weights
            running_loss += loss.item()
            # Calculate predictions
            predictions = torch.argmax(outputs, dim=1)
            if len(target.shape) > 1:
                target_labels = torch.argmax(target, dim=1)
            else:
                target_labels = target

            # Accumulate all targets and predictions for metrics calculation
            all_targets.extend(target_labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

        avg_loss = running_loss / len(self.train_loader)
        acc, sen, spe, mcc, f1_neg, f1_pos = calculate_metrics(torch.tensor(all_targets), torch.tensor(all_predictions))

        return avg_loss, acc, sen, spe, mcc

    def validate_epoch(self):
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():  # Disable gradient calculation
            for (data1, data2, target) in self.val_loader:
                data1,data2, target = data1.to(self.device),data2.to(self.device), target.to(self.device)
                outputs = self.model.forward_hybrid(data1,data2) 
                loss = self.criterion(outputs, target)
                running_loss += loss.item()
                # Calculate predictions
                predictions = torch.argmax(outputs, dim=1)
                if len(target.shape) > 1:
                    target_labels = torch.argmax(target, dim=1)
                else:
                    target_labels = target
                # Accumulate all targets and predictions for metrics calculation
                all_targets.extend(target_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        acc, sen, spe, mcc, f1_neg, f1_pos = calculate_metrics(torch.tensor(all_targets), torch.tensor(all_predictions))

        return avg_loss, acc, sen, spe, mcc, f1_neg, f1_pos


    def test(self):
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0

        all_targets = []
        all_predictions = []
        subjects = []
        logits_master = []

        with torch.no_grad():  # Disable gradient calculation
            for data1, data2, target, sub_ids in self.test_loader:
                # print(data.shape)
                # print(target)
                # print(sub_ids)
                # input()
                data1,data2, target = data1.to(self.device),data2.to(self.device), target.to(self.device)
                outputs = self.model.forward_hybrid(data1,data2) 
                logits = torch.softmax(outputs, dim = 1)
                logits_master.extend(logits.cpu().numpy())
                loss = self.criterion(outputs, target)
                
                running_loss += loss.item()
                # Calculate predictions
                predictions = torch.argmax(outputs, dim=1)
                if len(target.shape) > 1:
                    target_labels = torch.argmax(target, dim=1)
                else:
                    target_labels = target

                # Accumulate all targets and predictions for metrics calculation
                all_targets.extend(target_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                subjects.extend(sub_ids)


        avg_loss = running_loss / len(self.test_loader)
        acc, sen, spe, mcc, f1_neg, f1_pos = calculate_metrics(torch.tensor(all_targets), torch.tensor(all_predictions))
        self.res_dict = {
            "avg_loss": avg_loss,
            "accuracy": acc,
            "sensitivity": sen, "specificity": spe, "MCC": mcc
        }
        # for key, value in self.res_dict.items():
        #     print(f"{key}: {value:.4f}", end='  ')
        #     print()

        self.frag_to_sub_metrics(subjects, all_predictions, all_targets, logits_master)
        
        
    
    def train(self, epochs):
        self.train_val_metrics = []
        model_quality = -1
        for epoch in tqdm(range(0,epochs), ncols = 120):
            # current_lr = self.optimizer.param_groups[0]['lr']
            # print(f"Epoch [{epoch+1}/{epochs}], Learning Rate: {current_lr}")

            tra_loss, tra_acc, tra_sen, tra_spe, tra_mcc = self.train_epoch()
            val_loss, val_acc, val_sen, val_spe, val_mcc, val_f1neg, val_f1pos= self.validate_epoch()

            # print(f"Epoch {epoch+1}/{epochs}")
            # print(f"Tra Loss: {tra_loss:.4f}, Tra Acc: {tra_acc:.4f}. Tra Sen: {tra_sen:.4f}, Tra Spe: {tra_spe:.4f}, Tra MCC: {tra_mcc:.4f}")
            # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Sen: {val_sen:.4f}, Val Spe: {val_spe:.4f}, Val MCC: {val_mcc:.4f}")

            # mcc_q = (1/10)*tra_mcc + (9/10)*val_mcc
            # qual = val_mcc
            qual = (val_f1neg+val_f1pos)/2
            # Append formatted strings to the list
            self.train_val_metrics.append(f"Epoch {epoch+1}/{epochs}")
            self.train_val_metrics.append(f"Tra Loss: {tra_loss:.4f}, Tra Acc: {tra_acc:.4f}, Tra Sen: {tra_sen:.4f}, Tra Spe: {tra_spe:.4f}, Tra MCC: {tra_mcc:.4f}")
            self.train_val_metrics.append(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Sen: {val_sen:.4f}, Val Spe: {val_spe:.4f}, Val MCC: {val_mcc:.4f}, Val F1_neg: {val_f1neg:.4f}, Val F1_neg: {val_f1pos:.4f}")
            

            if epoch>-1:
                #check model quality
                if qual > model_quality:
                    model_quality = qual
                    self.train_val_metrics.append(" new best model")
                    # self.best_model_state_dict = self.model.state_dict() #checkpoint
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_val_metrics = [val_acc, val_sen, val_spe]


            self.train_val_metrics.append(f"best: {model_quality:.4f}")  
            self.train_val_metrics.append("")# Add a blank line between epochs for readability
        
            if self.scheduler is not None:
                self.scheduler.step()

        #load best model
        if epochs >0:
            self.model.load_state_dict(self.best_model_state_dict)
            self.model.to(self.device)
        # Test the final model after training
        self.test()
        
    def frag_to_sub_metrics(self, subjects, predictions, labels, logits):
        df_frag = pd.DataFrame({
            'subject': subjects,
            'label': labels,
            'prediction': predictions,
            'logits': logits,
        })

        # Group by subject and apply majority voting
        def majority_vote(series):
            return Counter(series).most_common(1)[0][0]
        def argmax_logits(logit_arrays):
            avg_logits = np.mean(logit_arrays, axis=0)
            return np.argmax(avg_logits) #returns 1 or 0
        
        def argmax_logits_2(logit_arrays):
            avg_logits = np.mean(logit_arrays, axis=0)
            return avg_logits[1] #returns logit of class 1
        
        subject_predictions = df_frag.groupby('subject')['prediction'].agg(majority_vote)
        sub_predictions_soft_arg = df_frag.groupby('subject')['logits'].agg(argmax_logits)
        sub_predictions_soft_log = df_frag.groupby('subject')['logits'].agg(argmax_logits_2)
        subject_labels = df_frag.groupby('subject')['label'].agg(majority_vote)

        # Combine into a final DataFrame
        subject_df = pd.DataFrame({
            'subject': subject_predictions.index,
            'subject_label': subject_labels.values,
            'subject_prediction': subject_predictions.values,
            'subject_prediction_soft_arg': sub_predictions_soft_arg.values,
            'subject_prediction_soft_log': sub_predictions_soft_log.values,

        })
        
        # print(subject_df)
        # input()

        #VOTING
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
        
        #SOFT VOTING
        y_true = subject_df['subject_label']
        y_pred = subject_df['subject_prediction_soft_arg']
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # Calculate metrics
        acc = (tp + tn) / (tp + tn + fp + fn)
        sen = tp / (tp + fn)  # Also called recall or TPR
        spe = tn / (tn + fp)  # True negative rate
        mcc = matthews_corrcoef(y_true, y_pred)
        self.res_dict_sub_soft = {
            "accuracy": acc,
            "sensitivity": sen, "specificity": spe, "MCC": mcc
        }

        self.frag_pred = df_frag
        self.subj_pred = subject_df

    
    def save_model(self, output_dir, name, save_param=1):
        save_dir = os.path.join(output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        if save_param == 1:
            state_dict = self.model.state_dict()
            self.model.save_pretrained(
                save_dir,
                state_dict=state_dict,
            )

        txt_name = os.path.join(save_dir,'results.txt')
        with open(txt_name, "w") as file:
            json.dump({"results_frag": self.res_dict, "results_sub": self.res_dict_sub, "results_sub_soft": self.res_dict_sub_soft}, file, indent = 4)

        self.frag_pred.to_csv(os.path.join(save_dir,'fragment_pred.csv'), index=False)
        self.subj_pred.to_csv(os.path.join(save_dir,'subject_pred.csv'), index=False)

        #save training and val results
        txt_name2 = os.path.join(save_dir,'train+val_metrics.txt')
        with open(txt_name2, "w") as file:
            for line in self.train_val_metrics:
                file.write(line + "\n")
        txt_name3 = os.path.join(save_dir,'best_val_metrics.npy')
        best_val_metrics_arr = np.array(self.best_val_metrics, dtype=np.float32)
        np.save(txt_name3, best_val_metrics_arr)

        




