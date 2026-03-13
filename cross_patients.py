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
from models.refined_detector import retroactive_check, ominous_check
from metrics.metrics_calculator import calculate_metrics

absolute_path = "data"
# patients = listdir(absolute_path)
patients = ['chb01', 'chb03', 'chb05', 'chb07', 'chb14']
extensions_options = [
    "-cov.txt",
    "-time_features.txt",
    "-freq_features.txt"
]
extensions_set = [extensions for length in range(1, len(extensions_options)+1) for extensions in list(combinations(extensions_options, length))]
types = ['full']
k=1

for extensions in extensions_set:
    name_pattern = r'-([^_.]+)[_.]'
    extensions_names = [re.search(name_pattern, name).group(1) for name in extensions]
    extensions_string = '-'.join(extensions_names)
    print(f'Extensões: {extensions_names}\n')
    for type in types:
        print(f'Type: {type}\n')
        for patient in patients:
            print(f'Paciente {patient}\n')
        
            path = "data/" + patient
        
            metrics = []

            # Os Nfiles primeiros arquivos serão utilizados para treino
            Nfiles = 4
            k_min = 8
            k_quart = 5
            q = 0.25    

            Xtrn, (Xtst, Ytst) = split_data(path, extensions, Nfiles, True if type=='reduced' else False)
            attr_size = Xtrn.shape[0]

            if patient == 'chb01':
                # Definição de modelos
                models = [
                    online_pca_encoder(tol = 0.99, k=k_min, filter='min'),
                    online_pca_encoder(tol = 0.99, k=k_quart, q=q, filter='quart'),
                    online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2),), k=k_min, filter='min'),
                    online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2),), k=k_quart, q=q, filter='quart'),
                    online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2), math.ceil(attr_size/4), math.ceil(attr_size/2)), k=k_min, filter='min'),
                    online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2), math.ceil(attr_size/4), math.ceil(attr_size/2)), k=k_quart, q=q, filter='quart'),
                    online_oc_svm_detector(kernel='sigmoid', gamma=0.5, nu=0.4, k=k_min, filter='min'),
                    online_oc_svm_detector(kernel='sigmoid', gamma=0.5, nu=0.4, k=k_quart, q=q, filter='quart'),
                    online_if_detector(n_estimators=10, contamination=0.01, random_state=42, max_samples=512, max_features=1.0, k=k_min, filter='min'),
                    online_if_detector(n_estimators=10, contamination=0.01, random_state=42, max_samples=512, max_features=1.0, k=k_quart, q=q, filter='quart')
                ]

            Yens_retro = 0
            Yens_full = 0

            for model in models:

                print(f'Modelo {model.name}')

                if patient == 'chb01':
                    # Treino
                    print("Treino")
                    model.train(Xtrn)

                # Teste
                print("Teste")
                Ypred = model.test(Xtst)
                if model.filter == 'min':
                    Yens_retro += Ypred
                else:
                    Yens_full += Ypred

                # Métricas
                print("Calculando métricas\n")
                metrics.append(calculate_metrics(Ytst, Ypred))

            # Avaliação do comitê
            print(f'Modelo comitê')
            Yens_retro = np.sign(Yens_retro)
            Yens_full = np.sign(Yens_full)
            print("Calculando métricas\n")
            metrics.append(calculate_metrics(Ytst, Yens_retro))
            metrics.append(calculate_metrics(Ytst, Yens_full))

            # Dados das métricas obtidas
            final_metrics = pd.DataFrame(metrics)
            indexes = [model.name for model in models]
            indexes.append('ensemble-retro')
            indexes.append('ensemble-full')
            final_metrics.index = indexes
            final_metrics.to_csv(f'metrics/cross_patient/{type}/{extensions_string}/{patient}-online.csv',decimal=',', sep=';')