# Importando as bibliotecas

import numpy as np
import matplotlib.pyplot as plt
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

    return label

# Dividindo os dados do Iris em treino e testes
vet_train = database
vet_test = trainbase.data_train_classtwo

k = 3  # Valor de k para o KNN

preview = knn(vet_train, vet_test, k)

# Calcular acurácia do treinamento

spected_class = 2
correct = 0

qtd_vet = len(vet_test)

for i in preview:
    if i == spected_class:
        correct += 1 

acuracy = (correct * 100) / qtd_vet

print(f"Sua taxa de acerto foi de: {acuracy:,.2f}%")

#############################################################################################

# Plotar os pontos de treinamento e de teste em 3D
x_train = vet_train[:, 0]
y_train = vet_train[:, 1]
z_train = vet_train[:, 2]

x_test = vet_test[:, 0]
y_test = vet_test[:, 1]
z_test = vet_test[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotar os pontos de treinamento
ax.scatter(x_train, y_train, z_train, c=z_train, label="Dados de Treino", cmap='viridis', marker='o')

# Plotar o ponto de teste
ax.scatter(x_test, y_test, z_test, c=preview, label="Dado de Teste", cmap='viridis', marker='x', s=100)

# Definir rótulos dos eixos e título do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('KNN')

ax.legend()

plt.show()
