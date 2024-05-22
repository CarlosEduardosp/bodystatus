from src.treino_e_teste.treino_e_teste import TreinoTeste
from src.Algoritimos.NaiveBayes.naivebayes import Algoritimo_naiveBayes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB


def avaliarNaiveBayes(todos_os_previsores: list, alvo: list, ):
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