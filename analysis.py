import pandas as pd
from itertools import combinations
import re
from os import listdir
import json

def fb(Precision, Sensitivity, b):
    aux = b**2
    den = (aux*Precision + Sensitivity)
    return (1 + aux) * Precision * Sensitivity / den if den != 0 else 0

# models = {
#     'pca': 0,
#     'mlp-1camadas': 0,
#     'mlp-3camadas': 0,
#     'oc-svm': 0,
#     'if': 0,
#     'ensemble': 0,
# }
# directory = 'features_comparison'
# prefix = 'metrics-'

models = {
    'pca-min': 0,
    'pca-quart': 0,
    'mlp-1camadas-min': 0,
    'mlp-1camadas-quart': 0,
    'mlp-3camadas-min': 0,
    'mlp-3camadas-quart': 0,
    'oc-svm-min': 0,
    'oc-svm-quart': 0,
    'if-min': 0,
    'if-quart': 0,
    'ensemble-retro': 0,
    'ensemble-full': 0
}
directory = 'refined2'
prefix = ''

absolute_path = "data"
patients = listdir(absolute_path)

types = ['full', 'reduced']

extensions_options = [
    "-cov.txt",
    "-time_features.txt",
    "-freq_features.txt"
]
extensions_set = [extensions for length in range(1, len(extensions_options)+1) for extensions in list(combinations(extensions_options, length))]
name_pattern = r'-([^_.]+)[_.]'

beta = 2

for patient in patients:
    print(f'Analisando paciente {patient}')
    best_fb = (0, 'none')
    for type in types:
        for extensions in extensions_set:
            extensions_names = [re.search(name_pattern, name).group(1) for name in extensions]
            extensions_string = "-".join(extensions_names)
            data = pd.read_csv(f'metrics/{directory}/{type}/{extensions_string}/{prefix}{patient}-online.csv',
                               decimal=',', sep=';', index_col=0)
            fb_scores = [fb(data.loc[model, 'Precision'], data.loc[model, 'Sensitivity'], beta) for model in data.index.tolist()]
            column_name = f'F{beta}'
            data[column_name] = fb_scores
            best_model = data[column_name].idxmax()
            fb_value = data.loc[best_model, column_name]
            best_fb = (float(fb_value), best_model + ' (' + extensions_string + '-' + type + ')') if fb_value > best_fb[0] else best_fb
            models[best_model] += 1
    print(f'Melhor desempenho para o paciente: {best_fb}\n')
print(f"Pontuação final: {str(models)}")