import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('Fish.csv')

# Удаление строк с нулевым весом (ошибка в данных)
df = df[df['Weight'] > 0]

# Выделение признаков и целевой переменной
features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
X = df[features]
y = df['Weight']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Модель многослойного персептрона
print("=" * 60)
print("МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН")
print("=" * 60)

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),  # 3 скрытых слоя
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    learning_rate_init=0.001,
    alpha=0.001  # регуляризация
)

mlp.fit(X_train_scaled, y_train)

# Предсказания
y_pred_mlp_train = mlp.predict(X_train_scaled)
y_pred_mlp_test = mlp.predict(X_test_scaled)

# Оценка качества
r2_train_mlp = r2_score(y_train, y_pred_mlp_train)
r2_test_mlp = r2_score(y_test, y_pred_mlp_test)

rmse_train_mlp = np.sqrt(mean_squared_error(y_train, y_pred_mlp_train))
rmse_test_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp_test))

print(f"R2 на обучающей выборке: {r2_train_mlp:.4f}")
print(f"R2 на тестовой выборке: {r2_test_mlp:.4f}")
print(f"RMSE на обучающей выборке: {rmse_train_mlp:.2f}")
print(f"RMSE на тестовой выборке: {rmse_test_mlp:.2f}")
print(f"Количество итераций: {mlp.n_iter_}")

# 2. Линейная регрессия
print("\n" + "=" * 60)
print("ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("=" * 60)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Предсказания
y_pred_lr_train = lr.predict(X_train_scaled)
y_pred_lr_test = lr.predict(X_test_scaled)

# Оценка качества
r2_train_lr = r2_score(y_train, y_pred_lr_train)
r2_test_lr = r2_score(y_test, y_pred_lr_test)

rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_pred_lr_train))
rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr_test))

print(f"R2 на обучающей выборке: {r2_train_lr:.4f}")
print(f"R2 на тестовой выборке: {r2_test_lr:.4f}")
print(f"RMSE на обучающей выборке: {rmse_train_lr:.2f}")
print(f"RMSE на тестовой выборке: {rmse_test_lr:.2f}")

# Коэффициенты линейной регрессии
print("\nКоэффициенты линейной регрессии:")
for feature, coef in zip(features, lr.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Свободный член: {lr.intercept_:.4f}")

# Сравнение моделей
print("\n" + "=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

comparison = pd.DataFrame({
    'Модель': ['MLP (тест)', 'Линейная регрессия (тест)'],
    'R2': [r2_test_mlp, r2_test_lr],
    'RMSE': [rmse_test_mlp, rmse_test_lr]
})

print(comparison)

# Визуализация результатов
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# График для MLP
axes[0].scatter(y_test, y_pred_mlp_test, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2)
axes[0].set_xlabel('Истинный вес')
axes[0].set_ylabel('Предсказанный вес')
axes[0].set_title('Многослойный персептрон')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.05, 0.95, f'R² = {r2_test_mlp:.4f}\nRMSE = {rmse_test_mlp:.2f}',
            transform=axes[0].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# График для линейной регрессии
axes[1].scatter(y_test, y_pred_lr_test, alpha=0.6, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2)
axes[1].set_xlabel('Истинный вес')
axes[1].set_ylabel('Предсказанный вес')
axes[1].set_title('Линейная регрессия')
axes[1].grid(True, alpha=0.3)
axes[1].text(0.05, 0.95, f'R² = {r2_test_lr:.4f}\nRMSE = {rmse_test_lr:.2f}',
            transform=axes[1].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.show()

# Анализ важности признаков для MLP (через анализ весов)
print("\n" + "=" * 60)
print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ДЛЯ MLP")
print("=" * 60)

# Получаем веса первого слоя
first_layer_weights = mlp.coefs_[0]
feature_importance = np.abs(first_layer_weights).mean(axis=1)

importance_df = pd.DataFrame({
    'Признак': features,
    'Важность': feature_importance
}).sort_values('Важность', ascending=False)

print(importance_df)