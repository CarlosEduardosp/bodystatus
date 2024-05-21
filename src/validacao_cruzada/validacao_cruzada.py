from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def validacaoCruzada():


    # Separando os dados em folds
    kfold = KFold(n_splits=30, shuffle=True, random_state=5)