from src.Avaliar_algoritimo.avaliar_naivebayes.avaliar_naivebayes import avaliarNaiveBayes
from src.Avaliar_algoritimo.avaliar_svm.avaliar_svm import avaliarSvm
from src.Avaliar_algoritimo.avaliar_regressaoLogistica.avaliar_regressaoLogistica import avaliarRegressaoLogistica
from src.Avaliar_algoritimo.avaliar_KNN.avaliar_KNN import avaliarKnn
from src.Avaliar_algoritimo.avaliar_arvorededecisao.avaliar_arvoredeDecisao import avaliarArvoredeDecisao
from src.Avaliar_algoritimo.avaliar_randomForest.avalizar_randomForest import avaliarRandomForest
from src.Avaliar_algoritimo.avaliar_xgboost.avaliar_xgboost import avaliarXGBoost
from src.Avaliar_algoritimo.avaliar_lightgbm.avaliar_lightgbm import avaliarLightGBM
from src.Avaliar_algoritimo.Catboost.avaliar_catboost import avaliarCatBoost

def testar_e_avaliar(todos_os_previsores: list, alvo: list, codigo_algoritimo: int):
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

        avaliarKnn(todos_os_previsores, alvo)

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