from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.analise_de_dados.treino_e_teste.treino_e_teste import TreinoTeste
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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


def avaliarCatBoost(todos_os_previsores: list, alvo: list, ):
    print('----------------------------------------------')
    print('Testando e Avaliando CatBoost: ')
    print('----------------------------------------------')

    # dados de treino de todos os previsores de uma vez.
    for i in todos_os_previsores:
        print(f'Teste com {i['id']}')

        dados_treino_e_teste = TreinoTeste(i['previsores'], alvo)
        # print(dados_treino_e_teste)

        # aplicando o algoritimo XGboost.
        response = Algoritimo_catboost(dados_treino_e_teste)

        # validação cruzada

        # Separando os dados em folds
        kfold = KFold(n_splits=2, shuffle=True, random_state=5)

        # Criando o modelo
        modelo = make_pipeline(
            StandardScaler(),
            CatBoostClassifier(task_type='CPU', iterations=100, learning_rate=0.1, depth=8, random_state=5,
                               eval_metric="Accuracy")
        )
        resultado = cross_val_score(modelo, i['previsores'], alvo, cv=kfold)

        # Usamos a média e o desvio padrão
        print(f"Validação Cruzada, Acurácia Média: {resultado.mean() * 100.0:.2f}")