import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Загрузка данных
df = pd.read_csv("Fish.csv")


# 2. One-Hot Encoding для Species
df = pd.get_dummies(df, columns=["Species"], drop_first=True)

# 3. Признаки и цель
X = df.drop("Weight", axis=1)
y = df["Weight"]

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,shuffle=True, random_state=77
)

# 5. Нормализация
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 6. Многослойный персептрон
mlp = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    batch_size=16,
    max_iter=300,
    learning_rate="constant",
    learning_rate_init=0.01,
    random_state=77
)

mlp.fit(X_train_s, y_train)
print("Обучение завершено!")
print(f"Количество итераций: {mlp.n_iter_}")
print(f"Финальная потеря: {mlp.best_loss_:.4f}")

# 7. Предсказание
y_test_pred = mlp.predict(X_test_s)
y_train_pred = mlp.predict(X_train_s)

# 8. Метрики
print("\n Тестовая метрика")
mae_t = mean_absolute_error(y_test, y_test_pred)
rmse_t = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_t = r2_score(y_test, y_test_pred)

print("MAE =", round(mae_t, 2))
print("RMSE =", round(rmse_t, 2))
print("R2 =", round(r2_t, 3))

print("\n Обучающая метрика")
mae = mean_absolute_error(y_train, y_train_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2 = r2_score(y_train, y_train_pred)

print("MAE =", round(mae, 2))
print("RMSE =", round(rmse, 2))
print("R2 =", round(r2, 3))


lr = LinearRegression()
lr.fit(X_train_s, y_train)

# предсказания и оценка для линейной модели
y_train_pred = lr.predict(X_train_s)
y_test_pred = lr.predict(X_test_s)

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

print("\n Обучающая метрика(LR)")
print("MAE =", round(train_mae, 2))
print("RMSE =", round(train_rmse, 2))
print("R2 =", round(train_r2, 3))

print("\n Тестовая метрика(LR)")
print("MAE =", round(test_mae, 2))
print("RMSE =", round(test_rmse, 2))
print("R2 =", round(test_r2, 3))
