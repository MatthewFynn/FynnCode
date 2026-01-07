import torch
from typing import Optional, Tuple
from sklearn.metrics import matthews_corrcoef, f1_score

def calculate_metrics(targets, outputs) -> Tuple[float, float, float, float, float, float]:
    # Ensure outputs and targets are tensors on the same device
    outputs = outputs.to(targets.device)

    # Convert boolean tensors to integer tensors for counting
    TP = torch.sum((outputs == 1).int() & (targets == 1).int())
    TN = torch.sum((outputs == 0).int() & (targets == 0).int())
    FP = torch.sum((outputs == 1).int() & (targets == 0).int())
    FN = torch.sum((outputs == 0).int() & (targets == 1).int())

    # Calculate accuracy, sensitivity, specificity
    accuracy = (TP + TN) / (TP + TN + FP + FN).float()
    sensitivity = TP / (TP + FN).float() if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP).float() if (TN + FP) > 0 else 0
    mcc = matthews_corrcoef(targets.cpu().numpy(), outputs.cpu().numpy())

    f1s = f1_score(targets.cpu(), outputs.cpu(), average=None)  # returns [f1_class_0, f1_class_1]
    f1_neg, f1_pos = f1s

    return accuracy.item(), sensitivity.item(), specificity.item(), mcc, f1_neg, f1_pos