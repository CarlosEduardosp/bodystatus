from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def Algoritimo_KNN(dados_treino_e_teste: list):


    x_treino = dados_treino_e_teste['x_treino']
    y_treino = dados_treino_e_teste['y_treino']
    x_teste = dados_treino_e_teste['x_teste']
    y_teste = dados_treino_e_teste['y_teste']

    # treino do algoritimo
    knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=1)
    knn.fit(x_treino, y_treino)

    # avaliação do algoritimo dados de teste.
    previsoes_knn = knn.predict(x_teste)
    # print('teste com naive bayes', previsores_naive)
    # print('dados de y teste', y_teste)

    # verificando a acurácia.
    acuracia_teste = accuracy_score(y_teste, previsoes_knn)
    print(f'acurácia dados de teste é de: {acuracia_teste * 100:.2f}%')

    # matrix de confusão
    matriz = confusion_matrix(y_teste, previsoes_knn)
    # print('matriz de confusão', matriz)

    # avaliação do algoritimo dados de treino.
    previsores_knn = knn.predict(x_treino)
    # print('teste com naive bayes', previsores_naive)
    # print('dados de x treino', x_treino)

    # verificando a acurácia.
    acuracia_treino = accuracy_score(y_treino, previsores_knn)
    print(f'acurácia dados de treino é de: {acuracia_treino * 100:.2f}%')