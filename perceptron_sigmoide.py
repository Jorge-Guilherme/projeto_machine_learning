# bibliotecas
import numpy as np
import iris as db
import matplotlib.pyplot as plt

# criação do neurônio
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.errors = []

    # função de ativação
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) # função sigmoide

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if self.sigmoid(z) >= 0.5 else 0

    def train(self, X, y):
        for epoch in range(self.epochs):
            epoch_error = 0
        
            for i in range(len(X)):
                x = X[i]
                target = y[i]

                pred = self.predict(x)
                error = target - pred
                epoch_error += error ** 2

                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error
            
            # guardando os erros
            self.errors.append(np.sqrt(epoch_error / len(X)))


    def evaluate(self, X, y):
        correct = 0
        for i in range(len(X)):
            x = X[i]
            pred = self.predict(x)
            if pred == y[i]:
                correct += 1
        return correct / len(X)

# dados
data_train = np.asarray(db.rawIrisData)
X_train = data_train[:, :-1]
y_train = data_train[:, -1]

input_size = X_train.shape[1]

# treinamento do neurônio
perceptron = Perceptron(input_size)
perceptron.train(X_train, y_train)

# dados testes
data_test = np.asarray(db.testing)
X_test = data_test[:, :-1]
y_test = data_test[:, -1]

# Plotar o processo de treinamento
plt.plot(range(perceptron.epochs), perceptron.errors)
plt.xlabel("Época")
plt.ylabel("Erro")
plt.title("Erro por Época - SIGMOIDE")
plt.show()
