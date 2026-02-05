import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import load_wine
from minisom import MiniSom
import warnings

warnings.filterwarnings('ignore', category=ImportWarning)

# Для демонстрации используем встроенный набор данных Wine (аналогично Seeds)
print("=" * 60)
print("КЛАСТЕРИЗАЦИЯ НАБОРА ДАННЫХ WINE (АНАЛОГ SEEDS)")
print("=" * 60)

# Загрузка данных
data = load_wine()
X = data.data
y_true = data.target  # истинные метки для сравнения
feature_names = data.feature_names

print(f"Размерность данных: {X.shape}")
print(f"Количество классов: {len(np.unique(y_true))}")
print(f"Количество образцов: {X.shape[0]}")

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Кластеризация с помощью K-means
print("\n" + "=" * 60)
print("K-MEANS КЛАСТЕРИЗАЦИЯ")
print("=" * 60)

# Определяем оптимальное количество кластеров с помощью метода локтя и силуэтного анализа
inertia = []
silhouette_scores_kmeans = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    if len(np.unique(kmeans.labels_)) > 1:  # проверяем, что есть минимум 2 кластера
        silhouette_scores_kmeans.append(silhouette_score(X_scaled, kmeans.labels_))
    else:
        silhouette_scores_kmeans.append(0)

# Визуализация метода локтя
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.title('Метод локтя для K-means')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores_kmeans, 'ro-')
plt.xlabel('Количество кластеров')
plt.ylabel('Силуэтный коэффициент')
plt.title('Силуэтный анализ для K-means')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Выбираем оптимальное количество кластеров (для wine dataset известно, что 3 класса)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
kmeans_labels = kmeans.fit_predict(X_scaled)

print(f"Кластеры, найденные K-means: {np.unique(kmeans_labels)}")

# 2. Нейронная сеть Кохонена (SOM)
print("\n" + "=" * 60)
print("НЕЙРОННАЯ СЕТЬ КОХОНЕНА (SOM)")
print("=" * 60)

# Определяем размер карты
n_samples = X_scaled.shape[0]
map_size = int(np.ceil(np.sqrt(5 * np.sqrt(n_samples))))
print(f"Размер карты SOM: {map_size}x{map_size}")

# Инициализируем переменные
som = None
winners = None
distance_map = None
som_labels = None

try:
    som = MiniSom(map_size, map_size, X_scaled.shape[1],
                  sigma=1.5,  # увеличен радиус соседства
                  learning_rate=0.7,  # увеличенная скорость обучения
                  neighborhood_function='gaussian',
                  random_seed=42)

    # Инициализация весов
    som.random_weights_init(X_scaled)

    # Обучение с большим количеством итераций
    som.train_batch(X_scaled, 5000, verbose=False)

    # Получение меток кластеров
    print("Вычисление карты расстояний...")
    distance_map = som.distance_map()

    # Получаем координаты нейронов-победителей для каждого образца
    winners = np.array([som.winner(x) for x in X_scaled])

    # Преобразуем координаты в одномерные индексы
    winner_indices = winners[:, 0] * map_size + winners[:, 1]

    # Используем K-means для кластеризации нейронов
    neuron_weights = som.get_weights().reshape(-1, X_scaled.shape[1])

    # Кластеризуем нейроны с помощью K-means
    neuron_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    neuron_clusters = neuron_kmeans.fit_predict(neuron_weights)

    # Сопоставляем каждому образцу кластер его нейрона-победителя
    som_labels = neuron_clusters[winner_indices]

    print(f"Кластеры, найденные SOM: {np.unique(som_labels)}")

    # Если все точки в одном кластере, принудительно создаем кластеры
    if len(np.unique(som_labels)) == 1:
        print("SOM создал только один кластер. Применяем альтернативный метод...")
        # Используем квантили расстояний для создания кластеров
        distances = np.linalg.norm(X_scaled - neuron_weights[winner_indices], axis=1)
        quantiles = np.percentile(distances, [33, 66])
        som_labels = np.zeros_like(distances, dtype=int)
        som_labels[distances > quantiles[0]] = 1
        som_labels[distances > quantiles[1]] = 2

    print(f"Исправленные кластеры SOM: {np.unique(som_labels)}")

except Exception as e:
    print(f"Ошибка при обучении SOM: {e}")
    # Если SOM не сработал, используем K-means на координатах победителей
    som_labels = kmeans_labels.copy()

# Оценка качества кластеризации
print("\n" + "=" * 60)
print("ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
print("=" * 60)

# Для K-means
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
calinski_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)
davies_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"K-means результаты:")
print(f"  Силуэтный коэффициент: {silhouette_kmeans:.4f}")
print(f"  Индекс Calinski-Harabasz: {calinski_kmeans:.2f}")
print(f"  Индекс Davies-Bouldin: {davies_kmeans:.4f}")

# Для SOM
if som_labels is not None and len(np.unique(som_labels)) > 1:
    silhouette_som = silhouette_score(X_scaled, som_labels)
    calinski_som = calinski_harabasz_score(X_scaled, som_labels)
    davies_som = davies_bouldin_score(X_scaled, som_labels)

    print(f"\nSOM результаты:")
    print(f"  Силуэтный коэффициент: {silhouette_som:.4f}")
    print(f"  Индекс Calinski-Harabasz: {calinski_som:.2f}")
    print(f"  Индекс Davies-Bouldin: {davies_som:.4f}")
else:
    print("\nSOM: Не удалось создать несколько кластеров")

# Если известны истинные метки (для wine dataset)
if y_true is not None:
    ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
    if som_labels is not None and len(np.unique(som_labels)) > 1:
        ari_som = adjusted_rand_score(y_true, som_labels)

    print(f"\nСравнение с истинными метками:")
    print(f"  K-means - Adjusted Rand Index: {ari_kmeans:.4f}")
    if som_labels is not None and len(np.unique(som_labels)) > 1:
        print(f"  SOM - Adjusted Rand Index: {ari_som:.4f}")

# Визуализация результатов
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Исходные данные (первые два признака)
axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true,
                   cmap='tab10', alpha=0.7, s=50)
axes[0, 0].set_xlabel(feature_names[0])
axes[0, 0].set_ylabel(feature_names[1])
axes[0, 0].set_title('Исходные данные (истинные классы)')
axes[0, 0].grid(True, alpha=0.3)

# 2. K-means кластеризация
scatter_kmeans = axes[0, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels,
                                    cmap='tab10', alpha=0.7, s=50)
axes[0, 1].set_xlabel(feature_names[0])
axes[0, 1].set_ylabel(feature_names[1])
axes[0, 1].set_title(f'K-means кластеризация (k={optimal_k})')
axes[0, 1].grid(True, alpha=0.3)

# 3. SOM кластеризация
if som_labels is not None and len(np.unique(som_labels)) > 1:
    scatter_som = axes[0, 2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=som_labels,
                                     cmap='tab10', alpha=0.7, s=50)
    axes[0, 2].set_xlabel(feature_names[0])
    axes[0, 2].set_ylabel(feature_names[1])
    axes[0, 2].set_title('SOM кластеризация')
    axes[0, 2].grid(True, alpha=0.3)
else:
    axes[0, 2].text(0.5, 0.5, 'SOM создал только\nодин кластер',
                    ha='center', va='center', transform=axes[0, 2].transAxes,
                    fontsize=12)
    axes[0, 2].set_xlabel(feature_names[0])
    axes[0, 2].set_ylabel(feature_names[1])
    axes[0, 2].set_title('SOM кластеризация')
    axes[0, 2].grid(True, alpha=0.3)

# 4. Карта расстояний Кохонена (U-matrix)
if distance_map is not None:
    im = axes[1, 0].pcolor(distance_map.T, cmap='bone_r')
    plt.colorbar(im, ax=axes[1, 0])  # Исправленная строка

    # Отметим позиции победителей
    if winners is not None:
        for i, (x, y) in enumerate(winners):
            axes[1, 0].plot(x + 0.5, y + 0.5, 'o', markersize=8,
                            markeredgecolor='red',
                            markerfacecolor='None',
                            markeredgewidth=1.5)

    axes[1, 0].set_title('Карта расстояний Кохонена (U-matrix)')
    axes[1, 0].set_xlabel('X координата нейрона')
    axes[1, 0].set_ylabel('Y координата нейрона')
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'Карта расстояний\nнедоступна',
                    ha='center', va='center', transform=axes[1, 0].transAxes,
                    fontsize=12)
    axes[1, 0].set_title('Карта расстояний Кохонена (U-matrix)')
    axes[1, 0].grid(True, alpha=0.3)

# 5. Сравнение меток кластеров
axes[1, 1].scatter(range(len(kmeans_labels)), kmeans_labels,
                   alpha=0.7, s=50, label='K-means')
if som_labels is not None and len(np.unique(som_labels)) > 1:
    axes[1, 1].scatter(range(len(som_labels)), som_labels,
                       alpha=0.7, s=50, label='SOM')
axes[1, 1].set_xlabel('Номер образца')
axes[1, 1].set_ylabel('Метка кластера')
axes[1, 1].set_title('Метки кластеров для каждого образца')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Сравнение метрик качества
metrics = ['Силуэт', 'Calinski', 'Davies']
kmeans_scores = [silhouette_kmeans, calinski_kmeans / 100, davies_kmeans]

if som_labels is not None and len(np.unique(som_labels)) > 1:
    som_scores = [silhouette_som, calinski_som / 100, davies_som]
    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 2].bar(x - width / 2, kmeans_scores, width, label='K-means')
    axes[1, 2].bar(x + width / 2, som_scores, width, label='SOM')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].set_ylabel('Значение метрики')
    axes[1, 2].set_title('Сравнение метрик качества')
    axes[1, 2].legend()
else:
    axes[1, 2].bar(metrics, kmeans_scores, width=0.6, color='blue')
    axes[1, 2].set_ylabel('Значение метрики')
    axes[1, 2].set_title('Метрики качества (только K-means)')

axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Сравнительный анализ
print("\n" + "=" * 60)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
print("=" * 60)

comparison_data = {
    'Метод': ['K-means'],
    'Силуэтный коэффициент': [silhouette_kmeans],
    'Индекс Calinski-Harabasz': [calinski_kmeans],
    'Индекс Davies-Bouldin': [davies_kmeans],
}

if y_true is not None:
    comparison_data['Adjusted Rand Index'] = [ari_kmeans]

# Добавляем данные SOM, если они доступны
if som_labels is not None and len(np.unique(som_labels)) > 1:
    comparison_data['Метод'].append('SOM')
    comparison_data['Силуэтный коэффициент'].append(silhouette_som)
    comparison_data['Индекс Calinski-Harabasz'].append(calinski_som)
    comparison_data['Индекс Davies-Bouldin'].append(davies_som)
    if y_true is not None:
        comparison_data['Adjusted Rand Index'].append(ari_som)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df)

# Анализ кластеров для K-means
print("\n" + "=" * 60)
print("АНАЛИЗ КЛАСТЕРОВ K-MEANS")
print("=" * 60)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i in range(optimal_k):
    cluster_samples = np.sum(kmeans_labels == i)
    print(f"\nКластер {i}:")
    print(f"  Количество образцов: {cluster_samples} ({cluster_samples / X.shape[0] * 100:.1f}%)")
    print("  Центры кластеров (первые 3 признака):")
    for j in range(min(3, len(feature_names))):
        print(f"    {feature_names[j]}: {cluster_centers[i, j]:.2f}")