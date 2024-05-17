from sklearn.preprocessing import LabelEncoder
import numpy as nd

def Previsores2(df: list):

    """
    :return: previsores2 = conjunto de variáveis previsoras com as variáveis categóricas transformadas
    em numéricas pelo labelencoder.
    """

    previsores2 = df.iloc[:, 0:16].values

    previsores2[:, 1] = LabelEncoder().fit_transform(previsores2[:, 1])
    previsores2[:, 4] = LabelEncoder().fit_transform(previsores2[:, 4])
    previsores2[:, 5] = LabelEncoder().fit_transform(previsores2[:, 5])
    previsores2[:, 8] = LabelEncoder().fit_transform(previsores2[:, 8])
    previsores2[:, 9] = LabelEncoder().fit_transform(previsores2[:, 9])
    previsores2[:, 11] = LabelEncoder().fit_transform(previsores2[:, 11])
    previsores2[:, 14] = LabelEncoder().fit_transform(previsores2[:, 14])
    previsores2[:, 15] = LabelEncoder().fit_transform(previsores2[:, 15])

    return previsores2
