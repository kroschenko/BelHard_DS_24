import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# загрузка данных Fish.csv
print("Загрузка данных Fish...")
data = pd.read_csv('Fish.csv')

print(f"\nИнформация о данных:")
print(data.head())
print(f"\nОписание данных:")
print(data.describe())

y = data["Weight"]
X = data.drop(columns=["Weight"])

X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns.tolist()
print(f"\nКоличество образцов: {X.shape[0]}")
print(f"Количество образцов: {X.shape[1]}")
print(f"Признаки: {', '.join(feature_names)}")
print(f"Целевая переменная: Weight (граммы)")
print(f"Диапазон веса: {y.min()} - {y.max()} r")

# разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

print(f"\nРазделение данных:")
print(f"Обучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")

# масштабирование признаков
print("\nМасштабирование признаков...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# создание и обучение MLP модели
print("\nОбучение MLP модели...")

mlp_simple = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    #learning_rate='adaptive',
    #learning_rate_init=0.0005,
    max_iter=2000,
    random_state=42
)

# обучение модели
mlp_simple.fit(X_train_scaled, y_train)

print("Обучение завершено!")
print(f"Количество итераций: {mlp_simple.n_iter_}")
print(f"Финальная потеря: {mlp_simple.best_loss_:.4f}")

# предсказания и оценка
y_train_pred = mlp_simple.predict(X_train_scaled)
y_test_pred = mlp_simple.predict(X_test_scaled)

# метрики на обучающей выборке
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# метрики на тестовой выборке
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nМетрики на обучающей выборке (MLP):")
print(f"  MAE: {train_mae:.2f} г")
print(f"  RMSE: {train_rmse:.2f} г")
print(f"  R2 Score: {train_r2:.4f}")

print(f"\nМетрики на тестовой выборке (MLP):")
print(f"  MAE: {test_mae:.2f} г")
print(f"  RMSE: {test_rmse:.2f} г")
print(f"  R2 Score: {test_r2:.4f}")

# линейная модель
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# предсказания и оценка для линейной модели
y_train_pred = lr.predict(X_train_scaled)
y_test_pred = lr.predict(X_test_scaled)

# метрики на обучающей выборке
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# метрики на тестовой выборке
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nМетрики на обучающей выборке (линейная модель):")
print(f"  MAE: {train_mae:.2f} г")
print(f"  RMSE: {train_rmse:.2f} г")
print(f"  R2 Score: {train_r2:.4f}")

print(f"\nМетрики на тестовой выборке (линейная модель):")
print(f"  MAE: {test_mae:.2f} г")
print(f"  RMSE: {test_rmse:.2f} г")
