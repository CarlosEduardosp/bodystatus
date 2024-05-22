from src.treino_e_teste.treino_e_teste import TreinoTeste
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from src.Algoritimos.RegressaoLogistica.regressao_logistica import regressaoLogistica
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def avaliarRegressaoLogistica(todos_os_previsores: list, alvo: list, ):

    print('----------------------------------------------')
    print('Testando e Avaliando Regressão Logística: ')
    print('----------------------------------------------')

    # dados de treino de todos os previsores de uma vez.
    for i in todos_os_previsores:
        print(f'Teste com {i['id']}')

        dados_treino_e_teste = TreinoTeste(i['previsores'], alvo)
        # print(dados_treino_e_teste)

        # aplicando o algoritimo regressão logistica.
        response = regressaoLogistica(dados_treino_e_teste)

        # validação cruzada

        # Separando os dados em folds
        kfold = KFold(n_splits=30, shuffle=True, random_state=5)

        # Criando o modelo
        modelo = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=1, max_iter=1000, penalty="l2", tol=0.0001, C=1, solver="lbfgs")
        )
        resultado = cross_val_score(modelo, i['previsores'], alvo, cv=kfold)

        # Usamos a média e o desvio padrão
        print(f"Validação Cruzada, Acurácia Média: {resultado.mean() * 100.0}")