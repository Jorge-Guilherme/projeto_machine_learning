# bibliotecas

import numpy as np
from iris import rawIrisData as db
from iris_train import data_train as dt

# função para os mínimos quadrados
def least_squares(x, y):
    n = len(x)
    
    # soma das regressões
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x ** 2)
    sum_xy = np.sum(x * y)
    
    # cálculo dos coeficientes
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept


for i in range(len(dt)): # iteração para acessar todos os i vetores do database iris
    
    #dados
    x = db
    y = dt[i]

    # recebendo os valores da função
    slope, intercept = least_squares(x, y)

    # cálculando as predições
    y_pred = slope * x + intercept

    # cálculo do erro RMS
    rms = np.sqrt(np.mean((y - y_pred) ** 2))

    # print dos coeficientes e erro correspondente de cada coordenada
    print(f"Coeficiente Angular do [{i}]º:", slope)
    print(f"Regressão Linear do [{i}]º:", intercept)
    print(f"Erro RMS do [{i}]º:", rms)
    print("\n")
