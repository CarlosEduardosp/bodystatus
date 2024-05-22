from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def regressaoLogistica(dados_treino_e_teste: list):
    """
    :param dados_treino_e_teste:
    :return:
    """

    x_treino = dados_treino_e_teste['x_treino']
    y_treino = dados_treino_e_teste['y_treino']
    x_teste = dados_treino_e_teste['x_teste']
    y_teste = dados_treino_e_teste['y_teste']

    # treino do algoritimo
    logistica = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=1, max_iter=500, penalty="l2", tol=0.0001, C=1, solver="lbfgs")
    )
    logistica.fit(x_treino, y_treino)

    # avaliação do algoritimo dados de teste.
    previsoes_logistica = logistica.predict(x_teste)
    # print('teste com Regressão Logística', previsoes_logistica)
    # print('dados de y teste', y_teste)

    # verificando a acurácia.
    acuracia_teste = accuracy_score(y_teste, previsoes_logistica)
    print(f'acurácia dados de teste é de: {accuracy_score(y_teste, previsoes_logistica) * 100.0:.2f}%')

    # matrix de confusão
    matriz = confusion_matrix(y_teste, previsoes_logistica)
    # print('matriz de confusão', matriz)

    # avaliação do algoritimo dados de treino.
    previsores_treino = logistica.predict(x_treino)
    # print('teste com Regressão Logística', previsores_treino)
    # print('dados de x treino', x_treino)

    # verificando a acurácia.
    acuracia_treino = accuracy_score(y_treino, previsores_treino)
    print(f'acurácia dados de treino é de: {acuracia_treino * 100:.2f}%')
