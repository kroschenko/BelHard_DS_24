import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# 1. Загрузить данные
df = pd.read_csv('heart.csv')

# Разделить на признаки и целевую переменную
X = df.drop('target', axis=1)
y = df['target']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделить на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

# 2. Подбор оптимального k для KNN
print("\nПодбор оптимального k для KNN:")
best_k = 1
best_accuracy = 0

for k in range(1, 31):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_temp)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

    print(f"k = {k:2d}: accuracy = {accuracy:.4f}")

print(f"\nОптимальное значение k: {best_k} (accuracy: {best_accuracy:.4f})")

# Обучить модели
models = {
    'Наивный Байес': GaussianNB(),
    'Логистическая регрессия': LogisticRegression(random_state=42, max_iter=1000),
    f'KNN (k={best_k})': KNeighborsClassifier(n_neighbors=best_k)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# 3. Матрицы ошибок и метрики
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics_data = []

for idx, (name, result) in enumerate(results.items()):
    y_pred = result['y_pred']

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Предсказан 0', 'Предсказан 1'],
                yticklabels=['Фактический 0', 'Фактический 1'])
    ax.set_title(f'Матрица ошибок: {name}')
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Фактический класс')

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics_data.append({
        'Модель': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

plt.tight_layout()
plt.show()

# Вывод метрик в таблице
metrics_df = pd.DataFrame(metrics_data)
print("\n" + "=" * 50)
print("Сравнительная таблица метрик:")
print("=" * 50)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 4. ROC-анализ
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор (AUC = 0.5)', alpha=0.7)

for name, result in results.items():
    y_pred_proba = result['y_pred_proba']
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
        print(f"{name}: AUC = {auc:.4f}")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые классификаторов')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()