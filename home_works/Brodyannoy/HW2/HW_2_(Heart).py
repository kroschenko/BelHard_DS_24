# HW2 "Heart"

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("heart.csv")
col_ru = {
    "age": "Возраст",
    "sex": "Пол",
    "cp": "Тип боли в груди",
    "trestbps": "Давление в покое",
    "chol": "Холестерин",
    "fbs": "Сахар в крови натощак",
    "restecg": "ECG в покое",
    "thalach": "Максимальная ЧСС",
    "exang": "Стенокардия при нагрузке",
    "oldpeak": "Депрессия ST при нагрузке",
    "slope": "Наклон сегмента ST",
    "ca": "Количество сосудов",
    "thal": "Талассемия",
    "target": "Сердечное заболевание"
}
print(pd.Series(col_ru))

#1
print(df.head())
print(df.describe())
print(df.isna().sum())


#2
counts = df["target"].value_counts().sort_index()

plt.figure()
plt.bar(["Здоровые", "Больные"], counts)
plt.xlabel("Сердечное заболевание")
plt.ylabel("Кол-во")
plt.title("Кол-во здоровых и больных пациентов")
plt.show()


#3
plt.figure()
plt.scatter(df["age"],df["thalach"],c=df["target"].map({0: "green", 1: "red"}))
plt.xlabel("Возраст")
plt.ylabel("Максимальный пульс")
plt.title("Зависимость максимального пульса от возраста")
plt.legend(["Больные"])
plt.show()


#4
#категории
df['sex'] = df['sex'].map({0: 'female', 1: 'male'})

# создаю доп. колонки (One-Hot Encoding)
df_sex_encoded = pd.get_dummies(df['sex'], prefix='sex')
df = pd.concat([df, df_sex_encoded.astype(int)], axis=1)
df = df.drop("sex", axis=1)
print(df.head())


#5
# группировка по наличию болезни и расчёт среднего холестерина
mean_chol = df.groupby('target')['chol'].mean()

mean_chol.index = ['Здоровые', 'Больные']
print(mean_chol)


#6
cols_normalize = ['age', 'trestbps', 'chol', 'thalach']

# Min-Max нормализация (0–1) - для нейронных сетей
df[cols_normalize] = (df[cols_normalize] - df[cols_normalize].min()) / (df[cols_normalize].max() - df[cols_normalize].min())
print(df[cols_normalize].describe())

# Z-score стандартизация (вычитание среднего и деление на стандартное отклонение) - логистическая регрессия
df[cols_normalize] = (df[cols_normalize] - df[cols_normalize].mean()) / df[cols_normalize].std()
print(df[cols_normalize].describe())
