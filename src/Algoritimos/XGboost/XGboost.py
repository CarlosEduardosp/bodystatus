from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def Algoritimo_XGboost(dados_treino_e_teste: list):


    x_treino = dados_treino_e_teste['x_treino']
    y_treino = dados_treino_e_teste['y_treino']
    x_teste = dados_treino_e_teste['x_teste']
    y_teste = dados_treino_e_teste['y_teste']

    # treino do algoritimo
    xg = XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=250, objective='binary:logistic', random_state=3)
    xg.fit(x_treino, y_treino)

    # avaliação do algoritimo dados de teste.
    previsores_xg = xg.predict(x_teste)
    # print('teste com naive bayes', previsores_xg)
    # print('dados de y teste', y_teste)

    # verificando a acurácia.
    acuracia_teste = accuracy_score(y_teste, previsores_xg)
    print(f'acurácia dados de teste é de: {acuracia_teste * 100:.2f}%')

    # matrix de confusão
    matriz = confusion_matrix(y_teste, previsores_xg)
    # print('matriz de confusão', matriz)

    # avaliação do algoritimo dados de treino.
    previsores_naive = xg.predict(x_treino)
    # print('teste com naive bayes', previsores_naive)
    # print('dados de x treino', x_treino)

    # verificando a acurácia.
    acuracia_treino = accuracy_score(y_treino, previsores_naive)
    print(f'acurácia dados de treino é de: {acuracia_treino * 100:.2f}%')
