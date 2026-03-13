import pickle
from os import listdir
from models.pca import pca
import re
import numpy as np
import pandas as pd
import math
from itertools import combinations
from models.split_data import split_data
from models.online_pca_encoder import online_pca_encoder
from models.online_mlp_encoder import online_mlp_encoder
from models.online_oc_svm_detector import online_oc_svm_detector
from models.online_if_detector import online_if_detector
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial
from sklearn.metrics import roc_curve, auc

absolute_path = "data"
patients = listdir(absolute_path)
# patients = ['chb05', 'chb07', 'chb14', 'chb18']
extensions_options = [
    # "-cov.txt",
    # "-time_features.txt",
    "-freq_features.txt"
]
extensions_set = [extensions for length in range(1, len(extensions_options)+1) for extensions in list(combinations(extensions_options, length))]
types = [
    'full',
    # 'reduced'
]
k_min = 8
k_quart = 5
q = 0.25

for patient in patients:
    print(f'Paciente {patient}\n')
    for type in types:
        print(f'Type: {type}\n')
        path = "data/" + patient
        name_pattern = r'-([^_.]+)[_.]'
        for extensions in extensions_set:
            extensions_names = [re.search(name_pattern, name).group(1) for name in extensions]
            extensions_string = '-'.join(extensions_names)
            print(f'Extensões: {extensions_names}\n')

            metrics = []

            # Os Nfiles primeiros arquivos serão utilizados para treino
            Nfiles = 4

            Xtrn, (Xtst, Ytst) = split_data(path, extensions, Nfiles, True if type=='reduced' else False)
            attr_size = Xtrn.shape[0]

            # Definição de modelos
            # models = [
            #     online_pca_encoder(tol = 0.99, k=k_min, filter='min'),
            #     online_pca_encoder(tol = 0.99, k=k_quart, q=q, filter='quart'),
            #     online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2),), k=k_min, filter='min'),
            #     online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2),), k=k_quart, q=q, filter='quart'),
            #     online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2), math.ceil(attr_size/4), math.ceil(attr_size/2)), k=k_min, filter='min'),
            #     online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2), math.ceil(attr_size/4), math.ceil(attr_size/2)), k=k_quart, q=q, filter='quart'),
            #     online_oc_svm_detector(kernel='sigmoid', gamma=0.5, nu=0.4, k=k_min, filter='min'),
            #     online_oc_svm_detector(kernel='sigmoid', gamma=0.5, nu=0.4, k=k_quart, q=q, filter='quart'),
            #     online_if_detector(n_estimators=10, contamination=0.01, random_state=42, max_samples=512, max_features=1.0, k=k_min, filter='min'),
            #     online_if_detector(n_estimators=10, contamination=0.01, random_state=42, max_samples=512, max_features=1.0, k=k_quart, q=q, filter='quart')
            # ]

            models = [
                online_pca_encoder(tol = 0.99),
                online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2),)),
                online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2), math.ceil(attr_size/4), math.ceil(attr_size/2))),
                online_oc_svm_detector(kernel='sigmoid', gamma=0.5, nu=0.4),
                online_if_detector(n_estimators=10, contamination=0.01, random_state=42, max_samples=512, max_features=1.0)
            ]

            for model in models:

                print(f'({extensions_string}) Modelo {model.name}')

                # Treino
                print("Treino")
                model.train(Xtrn)

                # Teste
                print("Teste")
                Xtstrec = model.reconstruct(Xtst)
                loss = model.calculate_loss(Xtst, Xtstrec)
                filtered_loss = model.filter(pd.Series(loss))
                fpr, tpr, thresholds = roc_curve(Ytst, filtered_loss)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('Taxa de Falsos Positivos')
                plt.ylabel('Taxa de Verdadeiros Positivos')
                plt.title('Curva ROC')
                plt.legend(loc="lower right")
                plt.savefig(f'graphs/roc2/{type}/{extensions_string}/{patient}-{model.name}.png', dpi=300, bbox_inches='tight')
                plt.close()