from src.treino_e_teste.treino_e_teste import TreinoTeste
from src.Algoritimos.NaiveBayes.naivebayes import Algoritimo_naiveBayes
from src.Algoritimos.MaquinadeVetoresdeSuporte.SVM import SVM
from src.Algoritimos.RegressaoLogistica.regressao_logistica import regressaoLogistica
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def testar_e_avaliar(todos_os_previsores: list, alvo: list, codigo_algoritimo: 1):
    """
    :param todos_os_previsores:
    :param alvo:
    :param codigo_algoritimo:
    :return:
    """


    if codigo_algoritimo == 1: # NaiveBayes

        print('----------------------------------------------')
        print('Testando e Avaliando NaiveBayes: ')
        print('----------------------------------------------')

        # dados de treino de todos os previsores de uma vez.
        for i in todos_os_previsores:
            print(f'Teste com {i['id']}')

            dados_treino_e_teste = TreinoTeste(i['previsores'], alvo)
            # print(dados_treino_e_teste)

            # aplicando o algoritimo naive bayes.
            response = Algoritimo_naiveBayes(dados_treino_e_teste)

            # validação cruzada

            # Separando os dados em folds
            kfold = KFold(n_splits=30, shuffle=True, random_state=5)

            # Criando o modelo
            modelo = GaussianNB()
            resultado = cross_val_score(modelo, i['previsores'], alvo, cv=kfold)

            # Usamos a média e o desvio padrão
            print(f"Validação Cruzada, Acurácia Média: {resultado.mean() * 100.0}")

    elif codigo_algoritimo == 2: # SVM - Máquinas de vetores de suporte

        print('----------------------------------------------')
        print('Testando e Avaliando SVM: ')
        print('----------------------------------------------')

        # dados de treino de todos os previsores de uma vez.
        for i in todos_os_previsores:
            print(f'Teste com {i['id']}')

            dados_treino_e_teste = TreinoTeste(i['previsores'], alvo)
            # print(dados_treino_e_teste)

            # aplicando o algoritimo naive bayes.
            response = SVM(dados_treino_e_teste)

            # validação cruzada

            # Separando os dados em folds
            kfold = KFold(n_splits=30, shuffle=True, random_state=5)

            # Criando o modelo
            modelo = SVC(kernel='rbf', random_state=1, C = 20)
            resultado = cross_val_score(modelo, i['previsores'], alvo, cv=kfold)

            # Usamos a média e o desvio padrão
            print(f"Validação Cruzada, Acurácia Média: {resultado.mean() * 100.0}")

    elif codigo_algoritimo == 3:  # Regressão Logística

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
