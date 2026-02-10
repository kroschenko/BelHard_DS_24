import numpy as np
import random as rnd
import math
import pandas as pd


class KohonenMap:
    def __init__(self, shape, dimension, rate=0.1, sigma=2.0, tau2=1000.0):
        self.weights = np.random.random((shape[0], shape[1], dimension))
        self.initial_rate = rate
        self.rate = rate
        self.initial_sigma = sigma
        self.sigma = sigma
        self.shape = shape
        self.tau1 = 1000.0 / math.log(self.initial_sigma)
        self.tau2 = tau2

    def core(self, data, iterations_limit):
        samples_count = len(data)
        for iteration in range(1, iterations_limit + 1):
            index = rnd.randint(0, samples_count - 1)
            sample = data[index]
            win_row, win_col = self._define_win_neuron(sample)
            self._update_weights(sample, win_row, win_col)
            self._decay_parameters(iteration)

    def train(self, data):
        self.core(data, 1000)
        self.rate = 0.01
        self.core(data, 25000)

    def print_clusters(self, data):
        clustering = []
        for sample in data:
            win_row, win_col = self._define_win_neuron(sample)
            clustering.append((win_row.item(), win_col.item()))

        unique_clusters = {}
        for c in clustering:
            unique_clusters[c] = unique_clusters.get(c, 0) + 1
        print("Распределение по кластерам ", unique_clusters)

    def _define_win_neuron(self, sample):
        diff = self.weights - sample
        dists = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(dists), (self.shape[0], self.shape[1]))

    def _update_weights(self, sample, win_row, win_col):
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                dist_sq = (i - win_row) ** 2 + (j - win_col) ** 2
                neighborhood = math.exp(-dist_sq / (2 * (self.sigma ** 2)))
                self.weights[i, j] += self.rate * neighborhood * (sample - self.weights[i, j])

    def _decay_parameters(self, iteration):
        self.sigma = self.initial_sigma * math.exp(-iteration / self.tau1)
        self.rate = self.initial_rate * math.exp(-iteration / self.tau2)


df = pd.read_csv("iris.csv").drop(columns=["variety"])
hm = KohonenMap((1, 3), 4)
print(df.values.shape)
hm.train(df.values)
hm.print_clusters(df.values)
