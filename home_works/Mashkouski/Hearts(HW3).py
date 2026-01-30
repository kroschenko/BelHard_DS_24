import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import SplineTransformer, StandardScaler

#1
df = pd.read_csv("heart.csv")

print(df)
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

print("Сбалансированность: ", y_train.value_counts())

#2
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_nb = gnb.predict(X_test)
print(f"Количество неправильно классифицированных точек из {X_test.shape[0]} : {(y_test != y_pred_nb).sum()}")

#3
#* Логистическая регрессия
st = SplineTransformer(n_knots=4, degree=3)
X_train_spline = st.fit_transform(X_train)
X_test_spline = st.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_spline, y_train)

feature_name = "age"
feature_idx = X.columns.get_loc(feature_name)

x_vals = np.linspace(
    X_train.iloc[:, feature_idx].min(),
    X_train.iloc[:, feature_idx].max(),
    200
)

X_vis = pd.DataFrame(
    np.tile(X_train.mean().values, (len(x_vals), 1)),
    columns=X_train.columns
)
X_vis[feature_name] = x_vals

X_vis_spline = st.transform(X_vis)
y_prob_curve = lr.predict_proba(X_vis_spline)[:, 1]

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_prob_curve, label="P(болезнь)")
plt.scatter(X_train.iloc[:, feature_idx], y_train, alpha=0.3, s=15, label="Обучающие данные")
plt.xlabel("Возраст (age)")
plt.ylabel("Вероятность болезни")
plt.title("Логистическая регрессия")
plt.grid(True)
plt.legend()
plt.show()

y_pred_lr = lr.predict(X_test_spline)
acc_lr = accuracy_score(y_test, y_pred_lr)
print("\nЛогистическая регрессия")
print(f"Accuracy: {acc_lr:.4f}")
print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred_lr))
print("Отчет:\n", classification_report(y_test, y_pred_lr))

y_probs_lr = lr.predict_proba(X_test_spline)[:, 1]
print("Первые 5 вероятностей:", y_probs_lr[:5])
#*

feat1, feat2 = "age", "thalach"

X2 = df[[feat1, feat2]]
y = df["target"]
X_train, X_test, y_train2, y_test = train_test_split(
    X2, y, test_size=0.2, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# логистическая регрессия через SGD
sgd = SGDClassifier(
    loss="log_loss",
    max_iter=5000,
    tol=1e-4
)
sgd.fit(X_train_s, y_train2)
y_pred = sgd.predict(X_test_s)
print("Точность (Accuracy):", accuracy_score(y_test, y_pred))

plt.figure(figsize=(8, 5))

plt.scatter(
    X_train_s[y_train2 == 0, 0],
    X_train_s[y_train2 == 0, 1],
    label="Здоров",
    alpha=0.6,
    s=25
)

plt.scatter(
    X_train_s[y_train2 == 1, 0],
    X_train_s[y_train2 == 1, 1],
    label="Болен",
    alpha=0.6,
    s=25
)

x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
probs = sgd.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.contour(
    xx, yy, probs,
    levels=[0.5],
    linewidths=2
)

plt.xlabel("Возраст (масштабированный)")
plt.ylabel("Максимальный пульс (масштабированный)")
plt.title("Логистическая регрессия (SGD): точки и граница решений")
plt.legend()
plt.grid(True)
plt.show()


#* k-ближайшие соседи
acc_series = []
k_values = range(1, 10)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_tmp = knn.predict(X_test)
    acc_series.append(accuracy_score(y_test, y_pred_tmp))

acc_array = np.array(acc_series)
best_k = np.argmax(acc_array) + 1
best_acc = acc_array[best_k - 1]

plt.plot(list(k_values), acc_array, marker="o")
plt.scatter(best_k, best_acc)
plt.text(best_k, best_acc, f"  k={best_k}", verticalalignment='bottom')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("k-ближайшие соседи")
plt.show()

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_knn = knn_best.predict(X_test)

print("\nk-ближайшие соседи")
print(f"Лучшее k = {best_k}, accuracy = {best_acc:.4f}")
print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred_knn))
print("Отчет:\n", classification_report(y_test, y_pred_knn))

#4 ROC-анализ
fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)
roc_auc = auc(fpr, tpr)
print(f"\nПлощадь под ROC-кривой(AUC) для модели: {roc_auc:.4f}")

plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Случайный (AUC = 0.5)')
plt.xlabel('False Positive Rate (FPR) / (1 - Специфичность)')
plt.ylabel('True Positive Rate (TPR) / Чувствительность')
plt.title('Receiver Operating Characteristic(ROC) Analysis')
plt.legend()
plt.show()

