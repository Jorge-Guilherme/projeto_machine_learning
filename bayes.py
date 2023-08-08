import numpy as np
import iris as db

# rótulos
labels_train = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])

# vetores para treinamento
data_train = np.asarray([
[5.2,3.4,1.4,0.2], [4.7,3.2,1.6,0.2], [4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4], [5.2,4.1,1.5,0.1], [5.5,4.2,1.4,0.2], [4.9,3.1,1.5,0.1],
[5.0,3.2,1.2,0.2], [5.5,3.5,1.3,0.2], [4.9,3.1,1.5,0.1], [4.4,3.0,1.3,0.2], [5.1,3.4,1.5,0.2], [5.0,3.5,1.3,0.3], [4.5,2.3,1.3,0.3],
[4.4,3.2,1.3,0.2], [5.0,3.5,1.6,0.6], [5.1,3.8,1.9,0.4], [4.8,3.0,1.4,0.3], [5.1,3.8,1.6,0.2], [4.6,3.2,1.4,0.2], [5.3,3.7,1.5,0.2],
[5.0,3.3,1.4,0.2], [7.0,3.2,4.7,1.4], [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4.0,1.3], [6.5,2.8,4.6,1.5], [5.7,2.8,4.5,1.3],
[6.3,3.3,4.7,1.6], [4.9,2.4,3.3,1.0], [6.6,2.9,4.6,1.3], [5.2,2.7,3.9,1.4], [5.0,2.0,3.5,1.0], [5.9,3.0,4.2,1.5], [6.0,2.2,4.0,1.0],
[6.1,2.9,4.7,1.4], [5.8,2.7,5.1,1.9], [7.1,3.0,5.9,2.1], [6.3,2.9,5.6,1.8], [6.5,3.0,5.8,2.2], [7.6,3.0,6.6,2.1], [4.9,2.5,4.5,1.7],
[7.3,2.9,6.3,1.8], [6.7,2.5,5.8,1.8], [7.2,3.6,6.1,2.5], [6.5,3.2,5.1,2.0], [6.4,2.7,5.3,1.9], [6.8,3.0,5.5,2.1], [5.7,2.5,5.0,2.0],
[5.8,2.8,5.1,2.4], [6.4,3.2,5.3,2.3],[6.5,3.0,5.5,1.8]])

# cáculo da probabilidade normal
def gaussian(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# algoritmo do bayseano
def gaussian_naive_bayes(X_train, labels_train, X_test):
    classes = np.unique(labels_train)
    num_classes = len(classes)
    num_features = X_train.shape[1]
    
    # listas para armazenar as médias e os desvios padrões
    means = np.zeros((num_classes, num_features))
    stds = np.zeros((num_classes, num_features))
    
    # calculo das médias e desvios padrões
    for i, c in enumerate(classes):
        X_c = X_train[labels_train == c]
        means[i] = X_c.mean(axis=0)
        stds[i] = X_c.std(axis=0)
        
        print(f"Classe {c}:")
        for j in range(num_features):
            print(f"Atributo {j+1} - Desvio Padrão: {stds[i, j]:.2f}")
    
    # classificações dos testes
    y_pred = []
    for x in X_test:
        class_probs = []
        for i, c in enumerate(classes):
            probs = gaussian(x, means[i], stds[i])
            class_prob = np.prod(probs)
            class_probs.append(class_prob)
        
        # determina o rótulo de acordo com a probabilidade
        y_pred.append(classes[np.argmax(class_probs)])
    
    return np.array(y_pred)

# medição da acurácia
def acuracia(rotulos_preditos, rotulos_esperados):
    predicoes_corretas = np.sum(rotulos_preditos == rotulos_esperados)
    total_instancias = rotulos_esperados.size
    acc = (predicoes_corretas*100) / total_instancias
    return acc

# dados
X_test = db.rawIrisData

# saída
return_labels = gaussian_naive_bayes(data_train, labels_train.flatten(), X_test)

# rótulo dos vetores testes
spected_labels = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])

# chamando a função para calcular a acertividade
spected_labels = spected_labels.flatten()
accuracy = acuracia(return_labels, spected_labels)

print(f"Taxa de Acerto: {accuracy:,.2f}%")
