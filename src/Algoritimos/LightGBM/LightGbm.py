import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from datetime import datetime


def Algoritimo_LightGBM(dados_treino_e_teste: dict):
    x_treino = dados_treino_e_teste['x_treino']
    y_treino = dados_treino_e_teste['y_treino']
    x_teste = dados_treino_e_teste['x_teste']
    y_teste = dados_treino_e_teste['y_teste']

    # Dataset para treino
    dataset = lgb.Dataset(x_treino, label=y_treino)

    # Parâmetros
    parametros = {'num_leaves': 250,  # número de folhas
                  'objective': 'binary',  # classificação Binária
                  'max_depth': 2,
                  'learning_rate': .05,
                  'max_bin': 100}

    lgbm = lgb.train(parametros, dataset, num_boost_round=200)

    # Marcação do tempo de execução
    inicio = datetime.now()
    lgbm = lgb.train(parametros, dataset)
    fim = datetime.now()

    tempo = fim - inicio

    previsoes_lgbm = lgbm.predict(x_teste)

    print(previsoes_lgbm.shape)

    # Quando for menor que 5 considera 0 e quando for maior ou igual a 5 considera 1
    for i in range(0, 634):
        if previsoes_lgbm[i] >= .5:
            previsoes_lgbm[i] = 1
        else:
            previsoes_lgbm[i] = 0

    print(f'acurácia dados de teste é de: {accuracy_score(y_teste, previsoes_lgbm) * 100.0:.2f}%')

    # matrix de confusão
    matriz = confusion_matrix(y_teste, previsoes_lgbm)
    # print('matriz de confusão', matriz)

    # avaliação do algoritimo dados de treino.
    previsores_naive = lgbm.predict(x_treino)
    # print('teste com naive bayes', previsores_naive)
    # print('dados de x treino', x_treino)

    print(previsores_naive.shape)

    # verificando a acurácia.
    print(f'acurácia dados de treino é de: {accuracy_score(y_treino, previsores_naive) * 100:.2f}%')