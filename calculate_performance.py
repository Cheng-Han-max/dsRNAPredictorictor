import math
from sklearn import metrics


def performance_fun(y_pred, labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []

    for i in range(len(labels)):
        if y_pred[i] > 0.5 and labels[i] == 1:
            TP += 1
        elif y_pred[i] > 0.5 and labels[i] == 0:
             FP += 1
             FP_index.append(i)
        elif y_pred[i] < 0.5 and labels[i] == 1:
             FN += 1
             FN_index.append(i)
        elif y_pred[i] < 0.5 and labels[i] == 0:
             TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    Mcc = (TP * TN - FP * FN) / math.sqrt((TN + FN) * (FP + TN) * (TP + FN) * (TP + FP))
    precision = TP / (TP + FP)
    f1_score = (2 * precision * Sn) / (precision + Sn)
    roc_auc = metrics.roc_auc_score(labels, y_pred)
    return TP, TN, FP, FN, Sn, Sp, Acc, Mcc, precision, f1_score,  roc_auc