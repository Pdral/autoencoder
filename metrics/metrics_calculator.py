import numpy as np
import math

def fb(Precision, Sensitivity, b):
    aux = b**2
    den = (aux*Precision + Sensitivity)
    return (1 + aux) * Precision * Sensitivity / den if den != 0 else 0

confusion_matrix_calculator = {
    'TP': lambda Ytst, Ypred: len(np.where((Ytst > 0) & (Ypred > 0))[0]),
    'TN': lambda Ytst, Ypred: len(np.where((Ytst < 0) & (Ypred < 0))[0]),
    'FP': lambda Ytst, Ypred: len(np.where((Ytst < 0) & (Ypred > 0))[0]),
    'FN': lambda Ytst, Ypred: len(np.where((Ytst > 0) & (Ypred < 0))[0])
}

metrics_calculator = {
    # 'TP_rate': lambda TP, TN, FP, FN: TP/(TP+FN),
    # 'TN_rate': lambda TP, TN, FP, FN: TN/(TN+FP),
    # 'FP_rate': lambda TP, TN, FP, FN: FP/(TN+FP),
    # 'FN_rate': lambda TP, TN, FP, FN: FN/(TP+FN),
    'Precision': lambda TP, TN, FP, FN: 100 * TP / (TP + FP),
    'ACC': lambda TP, TN, FP, FN: 100 * (TP+TN)/(TP+TN+FP+FN),
    'Sensitivity': lambda TP, TN, FP, FN: 100 * TP/(TP + FN),
    'Especificity': lambda TP, TN, FP, FN: 100 * TN / (TN + FP)
}

complex_metrics_calculator = {
    'MG': lambda Sensitivity, Especificity, Precision: math.sqrt(Sensitivity * Especificity),
    'F1': lambda Sensitivity, Especificity, Precision: fb(Precision, Sensitivity, 1),
    'F2': lambda Sensitivity, Especificity, Precision: fb(Precision, Sensitivity, 2)
}

def safe_calc(func, *args, default=0):
    try:
        return func(*args)
    except (ZeroDivisionError, ValueError): 
        return default

def calculate_metrics(Ytst, Ypred):
    metrics = {}
    for confusion_metric in confusion_matrix_calculator:
        calculator = confusion_matrix_calculator[confusion_metric]
        metrics[confusion_metric] = safe_calc(calculator, Ytst, Ypred)
    for metric in metrics_calculator:
        calculator = metrics_calculator[metric]
        metrics[metric] = safe_calc(calculator, metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN'])
    for complex_metric in complex_metrics_calculator:
        calculator = complex_metrics_calculator[complex_metric]
        metrics[complex_metric] = safe_calc(calculator, metrics['Sensitivity'], metrics['Especificity'], metrics['Precision'])
    return metrics
