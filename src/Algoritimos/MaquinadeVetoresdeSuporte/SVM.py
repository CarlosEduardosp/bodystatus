from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def SVM(dados_treino_e_teste: list):

    x_treino = dados_treino_e_teste['x_treino']
    y_treino = dados_treino_e_teste['y_treino']
    x_teste = dados_treino_e_teste['x_teste']
    y_teste = dados_treino_e_teste['y_teste']

    # treino do algoritimo
    svm = SVC(kernel='rbf', random_state=1, C=2)
    svm.fit(x_treino, y_treino)

    # avaliação do algoritimo dados de teste.
    previsoes_svm = svm.predict(x_teste)
    # print('teste com SVM', previsores_svm)
    # print('dados de y teste', y_teste)

    # verificando a acurácia.
    acuracia_teste = accuracy_score(y_teste, previsoes_svm)
    print(f'acurácia dados de teste é de: {accuracy_score(y_teste, previsoes_svm) * 100.0:.2f}%')

    # matrix de confusão
    matriz = confusion_matrix(y_teste, previsoes_svm)
    # print('matriz de confusão', matriz)

    # avaliação do algoritimo dados de treino.
    previsores_naive = svm.predict(x_treino)
    # print('teste com naive bayes', previsores_naive)
    # print('dados de x treino', x_treino)

    # verificando a acurácia.
    acuracia_treino = accuracy_score(y_treino, previsores_naive)
    print(f'acurácia dados de treino é de: {acuracia_treino * 100:.2f}%')
