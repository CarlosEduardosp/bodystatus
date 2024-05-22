from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier


def Algoritimo_catboost(dados_treino_e_teste: dict):
    from src.analise_de_dados.analise_de_dados import df

    # Se precisar redefinir as variáveis de treino e teste, use train_test_split aqui.
    previsores4 = df.iloc[:, 0:16]
    alvo4 = df.iloc[:, 16]

    x_treino, x_teste, y_treino, y_teste = train_test_split(previsores4, alvo4, test_size=0.3, random_state=0)

    categoricas = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']

    catboost = CatBoostClassifier(task_type='CPU', iterations=100, learning_rate=0.1, depth=8, random_state=5,
                                  eval_metric="Accuracy")

    catboost.fit(x_treino, y_treino, cat_features=categoricas, plot=False, eval_set=(x_teste, y_teste))

    previsoes_cat = catboost.predict(x_teste)

    print(f'Acurácia nos dados de teste: {(accuracy_score(y_teste, previsoes_cat) * 100.0):.2f}%')

    previsoes_treino = catboost.predict(x_treino)

    print(f'Acurácia nos dados de treino: {(accuracy_score(y_treino, previsoes_treino) * 100.0):.2f}%')

