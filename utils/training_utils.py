import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

def macro_statistics(y_pred, y_true, raw=True):
    if raw:
        y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # statistics for each class
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(TP + FP == 0, 0, TP / (TP + FP))
        recall = np.where(TP + FN == 0, 0, TP / (TP + FN))
        f1 = np.where(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall))
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # macro average
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    macro_accuracy = np.mean(accuracy)

    return macro_accuracy, macro_precision, macro_recall, macro_f1

def adjust_learning_rate(optimizer, lr, epoch, decay=0.1):
    if epoch >= 10:
        lr *= decay
    if epoch >= 20:
        lr *= decay
    if epoch >= 40:
        lr *= decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class EarlyStopping(object):
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss