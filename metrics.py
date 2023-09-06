from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg


def compute_reg_metrics(ytrue, ypred):

    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    r, _ = pearsonr(ytrue, ypred)
    mae = mean_absolute_error(ytrue, ypred)
    return rmse, r2, r, mae


def compute_cls_metrics(y_true, y_prob):
    
    y_pred = np.array(y_prob) > 0.5
   
    auc = roc_auc_score(y_true, y_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    aupr = -np.trapz(precision, recall)
   
    F1 = f1_score(y_true, y_pred, average = 'binary')

    acc = accuracy_score(y_true, y_pred)
   
    mcc = matthews_corrcoef(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return auc, aupr, F1, acc
