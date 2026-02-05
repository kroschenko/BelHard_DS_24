import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# загрузка данных
df = pd.read_csv('seeds_dataset.txt', sep=r'\s+', header=None)

X = df.iloc[:, 0:7]
y = df.iloc[:, 7]
# нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
class KMeans:
    def __init__(self, data, k, max_iter=100):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.n, self.m = data.shape

    def initialize_centroids(self):
        indices = np.random.choice(self.n, self.k, replace=False)
        return self.data[indices]

    def assign_clusters(self, centroids):
        labels = []
        for x in self.data:
            distances = np.linalg.norm(x - centroids, axis=1)
            labels.append(np.argmin(distances))
        return np.array(labels)

    def update_centroids(self, labels):
        centroids = np.zeros((self.k, self.m))
        for i in range(self.k):
            points = self.data[labels == i]
            centroids[i] = points.mean(axis=0)
        return centroids

    def run(self):
        centroids = self.initialize_centroids()

        for _ in range(self.max_iter):
            labels = self.assign_clusters(centroids)
            new_centroids = self.update_centroids(labels)

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return labels

# запуск k-means
method = KMeans(X_scaled, 3)
labels = method.run()

result = pd.DataFrame({
    "Истинный класс": y,
    "Кластер k-means": labels
})

print("\nТаблица соответствий:")
print(pd.crosstab(result["Истинный класс"],
                  result["Кластер k-means"]))

class Kohonen:
    def __init__(self, data, k, lr=0.5, epochs=1000):
        self.data = data
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.n, self.m = data.shape

        # случайные начальные веса нейронов
        self.weights = np.random.rand(self.k, self.m)

    def train(self):
        for _ in range(self.epochs):
            for x in self.data:
                # расстояния до нейронов
                distances = np.linalg.norm(self.weights - x, axis=1)
                winner = np.argmin(distances)

                # обновляем веса победителя
                self.weights[winner] += self.lr * (x - self.weights[winner])

    def run(self):
        self.train()

        labels = []
        for x in self.data:
            distances = np.linalg.norm(self.weights - x, axis=1)
            labels.append(np.argmin(distances))

        return np.array(labels)

    # запуск сети Кохонена
kohonen = Kohonen(X_scaled, 3)
kohonen_labels = kohonen.run()

table = pd.crosstab(
    y,
    kohonen_labels,
    rownames=["Истинный класс"],
    colnames=["Кластер Кохонена"]
)

print(table)
