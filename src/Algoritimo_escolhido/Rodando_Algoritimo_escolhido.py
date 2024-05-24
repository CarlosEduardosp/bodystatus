import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.model.model_pessoa import Pessoa
from typing import Type, List


def rodar_Algoritimo_escolhido(pessoa: Type[Pessoa]) -> List:

    # Configurar pandas para comportamento futuro
    pd.set_option('future.no_silent_downcasting', True)

    # Lendo o arquivo csv
    alvo = pd.read_csv('../../alvo.csv', sep=',', encoding='utf-8', header=None)
    previsores = pd.read_csv('../../previsores.csv', sep=',', encoding='utf-8', header=None)

    # Convertendo 'alvo' para um array 1D se necessário
    if alvo.shape[1] == 1:  # Verifica se alvo tem apenas uma coluna
        alvo = alvo.values.ravel()  # Transforma em um array 1D
    else:
        raise ValueError("O arquivo 'alvo.csv' deve conter apenas uma coluna")

    # treino do algoritimo
    logistica = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=1, max_iter=500, penalty="l2", tol=0.0001, C=1, solver="lbfgs")
    )
    logistica.fit(previsores, alvo)

    # Exemplo de dados de entrada para previsão
    pessoa = np.array(pessoa.listar_atributos())

    # Transformando 'pessoa' para a forma correta (2D array)
    pessoa = pessoa.reshape(1, -1)

    # Fazendo a previsão
    resultado = logistica.predict(pessoa)

    return resultado
