# bibliotecas
import numpy as np
import iris as db
import matplotlib.pyplot as plt

# implementação do neurônio
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    # função de ativação
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if z >= 0 else 0 # função degrau

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]

                pred = self.predict(x)

                if pred != target:
                    self.weights += self.learning_rate * (target - pred) * x
                    self.bias += self.learning_rate * (target - pred)

    def evaluate(self, X, y):
        correct = 0
        for i in range(len(X)):
            x = X[i]
            pred = self.predict(x)
            if pred == y[i]:
                correct += 1
        return correct / len(X)

# dados de treinamento
data_train = np.asarray(db.rawIrisData)
X_train = data_train[:, :-1]
y_train = data_train[:, -1]

input_size = X_train.shape[1]

perceptron = Perceptron(input_size)

# treinamento do perceptron
errors = []

for epoch in range(perceptron.epochs):
    epoch_error = 0
    
    for i in range(len(X_train)):
        x = X_train[i]
        target = y_train[i]

        pred = perceptron.predict(x)
        error = target - pred
        epoch_error += error ** 2
        
        if pred != target:
            perceptron.weights += perceptron.learning_rate * error * x
            perceptron.bias += perceptron.learning_rate * error
            
    # cálculo do RMS
    epoch_rms = np.sqrt(epoch_error / len(X_train))
    errors.append(epoch_rms)

# plot do RMS por época
plt.plot(range(perceptron.epochs), errors)
plt.xlabel("Época")
plt.ylabel("RMS")
plt.title("RMS por época - DEGRAU")
plt.show()

# dados de teste
data_test = np.asarray(db.testing)
X_test = data_test[:, :-1]
y_test = data_test[:, -1]
