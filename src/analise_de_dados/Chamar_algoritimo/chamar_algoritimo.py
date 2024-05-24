from src.analise_de_dados.Algoritimos import avaliarNaiveBayes
from src.analise_de_dados.Algoritimos import avaliarSvm
from src.analise_de_dados.Algoritimos.RegressaoLogistica import avaliarRegressaoLogistica
from src.analise_de_dados.Algoritimos.Aprendizagem_Baseada_Instancia_KNN import AlgoritimoKnn
from src.analise_de_dados.Algoritimos import avaliarArvoredeDecisao
from src.analise_de_dados.Algoritimos.RandomForest import avaliarRandomForest
from src.analise_de_dados.Algoritimos import avaliarXGBoost
from src.analise_de_dados.Algoritimos.LightGBM.LightGbm import avaliarLightGBM
from src.analise_de_dados.Algoritimos.CATBOOST.catboost import avaliarCatBoost


def chamarAlgoritimo(todos_os_previsores: list, alvo: list, codigo_algoritimo: int):
    """
    :param todos_os_previsores:
    :param alvo:
    :param codigo_algoritimo:
    :return:
    """

    if codigo_algoritimo == 1:  # NaiveBayes

        avaliarNaiveBayes(todos_os_previsores, alvo)

    elif codigo_algoritimo == 2:  # SVM - Máquinas de vetores de suporte

        avaliarSvm(todos_os_previsores, alvo)

    elif codigo_algoritimo == 3:  # Regressão Logística

        avaliarRegressaoLogistica(todos_os_previsores, alvo)

    elif codigo_algoritimo == 4:  # Aprendizagem baseada em instâncias

        # instanciando a classe AlgoritimoKnn
        knn = AlgoritimoKnn()
        # chamando o metodo Avaliar e validação cruzada.
        knn.avaliar_e_validacao_cruzada(todos_os_previsores, alvo)

    elif codigo_algoritimo == 5:  # Árvore de Decisão

        avaliarArvoredeDecisao(todos_os_previsores, alvo)

    elif codigo_algoritimo == 6:  # Random Forest

        avaliarRandomForest(todos_os_previsores, alvo)

    elif codigo_algoritimo == 7:  # XGBoost

        avaliarXGBoost(todos_os_previsores, alvo)

    elif codigo_algoritimo == 8:  # LightGBM

        avaliarLightGBM(todos_os_previsores, alvo)

    elif codigo_algoritimo == 9:  # catboost

        avaliarCatBoost(todos_os_previsores, alvo)