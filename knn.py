# Importando as bibliotecas

import numpy as np
from iris import rawIrisData as database
import iris_train as trainbase

# Função para calcular a distância euclidiana entre dois pontos
def points_distance(p1, p2):
    x = 0
    y = 1
    z = 2
    w = 3

    return np.sqrt((p1[x] - p2[x])**2 + (p1[y] - p2[y])**2 + (p1[z] - p2[z])**2 + (p1[w] - p2[w])**2)

# Função para o algoritmo KNN
def knn(vet_train, vet_test, k):
    spected_class = 0
    label = []
    for w in vet_test:
        d = []
        for j in vet_train:
            dist = points_distance(w, j)
            d.append((dist, j[-1]))  # Armazena a distância e a classe correspondente

        d.sort()  # Classifica as distâncias em ordem crescente
        k_nearest = d[:k]  # Seleciona os k vizinhos mais próximos

        # Conta a ocorrência de cada classe nos vizinhos mais próximos
        counts = {}
        for dist, class_label in k_nearest:
            counts[class_label] = counts.get(class_label, 0) + 1

        # Determina a classe mais comum entre os k vizinhos mais próximos
        kn = max(counts, key=counts.get)
        label.append(kn)

        if w[-1] == kn: # Calcular acurácia
            spected_class += 1

        print(f"Classe do Algoritmo: [{int(kn)}] Classe Esperada: [{int(w[-1])}]\n")

    return label, spected_class

# Dividindo os dados do Iris em treino e testes
vet_train = database
vet_test = trainbase.data_train

k = 3  # Valor de k para o KNN

preview, spected_class = knn(vet_train, vet_test, k)

# Calcular acurácia do treinamento

qtd_vet = len(vet_test)

acuracy = (spected_class * 100) / qtd_vet

print(f"Taxa de Acerto: {acuracy:,.2f}%")
