import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpay as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

df = pd.read_csv("heart.csv")
print(df.head())
X = df.drop(labels='target', axis=1)
y = df['target']

#разбиваем на тестовую и обучающую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#проверка
print(y.value_counts(normalize=True))
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

#масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

 #обучение моделей

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
y_proba_nb = nb.predict_proba(X_test_scaled)[:, 1]

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

from sklearn.neighbors import KNeighborsClassifier
acc_series = []
for k in range(1, 19):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    y_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred_knn)
    acc_series.append(accuracy)

plt.plot(range(1, 19), acc_series)
plt.show()

#матрица ошибок по каждой модели

cm_nb = confusion_matrix(y_test, y_pred_nb)
print('nb:', cm_nb)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print('lr:', cm_lr)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print('knn:', cm_knn)

#рассчет точности и отчет по классификации по каждой модели

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"\nТочность модели на тестовой выборке(Accuracy): {accuracy_nb:.4f}")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_nb))

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nТочность модели на тестовой выборке(Accuracy): {accuracy_lr:.4f}")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_lr))

accuracy_knn = accuracy_score(y_test, y_pred_lr)
print(f"\nТочность модели на тестовой выборке(Accuracy): {accuracy_knn:.4f}")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_knn))

#ROC-анализ и визуализация кривых

plt.figure(figsize=(8, 6))
for name, proba in {
    "Naive Bayes": y_proba_nb,
    "Logistic Regression": y_proba_lr,
    "kNN": y_proba_knn
}.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые")
plt.legend()
plt.show()


