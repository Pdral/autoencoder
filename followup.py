import re
from os import listdir
import numpy as np
import pandas as pd
from itertools import combinations
from models.split_data import split_data
from models.online_pca_encoder import online_pca_encoder
from metrics.metrics_calculator import calculate_metrics

absolute_path = "data"
# patients = listdir(absolute_path)
patients = ['chb07', 'chb14', 'chb18']
extensions_options = [
    "-cov.txt",
    # "-time_features.txt",
    # "-freq_features.txt"
]
extensions_set = [extensions for length in range(1, len(extensions_options)+1) for extensions in list(combinations(extensions_options, length))]
model_names = []

for patient in patients:
    print(f'Paciente {patient}\n')
    path = "data/" + patient
    for extensions in extensions_set:
        name_pattern = r'-([^_.]+)[_.]'
        extensions_names = [re.search(name_pattern, name).group(1) for name in extensions]
        print(f'Extensões: {extensions_names}\n')

        metrics = []
        models_names = []

        # Os Nfiles primeiros arquivos serão utilizados para treino
        Nfiles = 4

        Xtrn, (Xtst, Ytst) = split_data(path, extensions, Nfiles, True)
        attr_size = Xtrn.shape[0]

        # Definição de modelos
        retro_models = [online_pca_encoder(tol = 0.99, filter='min', k=k) for k in range(1, 11)]
        ominous_models = [online_pca_encoder(tol = 0.99, filter='median', k=k, q=q) for k in range(1, 6) for q in [0.25, 0.5, 0.75]]

        # Yens = 0

        for model in retro_models:

            print(f'Modelo {model.name}')

            # Treino
            print("Treino")
            model.train(Xtrn)

            # Teste
            print("Teste")
            Ypred = model.test(Xtst)

            metrics.append(calculate_metrics(Ytst, Ypred))
            models_names.append(model.name)

        for model in ominous_models:

            print(f'Modelo {model.name}')

            # Treino
            print("Treino")
            model.train(Xtrn)

            # Teste
            print("Teste")
            Ypred = model.test(Xtst)

            metrics.append(calculate_metrics(Ytst, Ypred))
            models_names.append(f'{model.name}-{model.q}')

        # Dados das métricas obtidas
        final_metrics = pd.DataFrame(metrics)
        final_metrics.index = models_names
        final_metrics.to_csv(f'metrics/follow-up/{patient}.csv',
                             decimal=',', sep=';')