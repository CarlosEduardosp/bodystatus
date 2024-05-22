from src.Algoritimos.NaiveBayes.naivebayes import avaliarNaiveBayes
from src.Algoritimos.MaquinadeVetoresdeSuporte.SVM import avaliarSvm
from src.Algoritimos.RegressaoLogistica.regressao_logistica import avaliarRegressaoLogistica
from src.Algoritimos.Aprendizagem_Baseada_Instancia_KNN.knn import AlgoritimoKnn
from src.Algoritimos.ArvoredeDecisao.ArvoredeDecisao import avaliarArvoredeDecisao
from src.Algoritimos.RandomForest.RandomForest import avaliarRandomForest
from src.Algoritimos.XGboost.XGboost import avaliarXGBoost
from src.Algoritimos.LightGBM.LightGbm import avaliarLightGBM
from src.Algoritimos.CATBOOST.catboost import avaliarCatBoost


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