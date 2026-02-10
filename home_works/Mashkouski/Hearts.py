import pandas as pd
import matplotlib.pyplot as plt

#1
df = pd.read_csv("heart.csv")

print("\nПервые 5 строк")
print(df.head())

print("\nКоличество пропусков по столбцам")
print(df.isnull().sum())


counts = df['target'].value_counts().sort_index()

#2
plt.figure()
plt.bar(['Здоровые', 'Больные'], counts.values)
plt.title('Количество здоровых и больных пациентов')
plt.xlabel('Группа')
plt.ylabel('Количество пациентов')
plt.show()

#3
plt.figure()
healthy = df[df['target'] == 0]
sick = df[df['target'] == 1]

plt.scatter(healthy['age'], healthy['thalach'], label='Здоровые', alpha=0.7)
plt.scatter(sick['age'], sick['thalach'], label='Больные', alpha=0.7)

plt.xlabel("Возраст")
plt.ylabel("Максимальный пульс (thalach)")
plt.title("Зависимость максимального пульса от возраста")
plt.legend()
plt.show()

#4
#4
df['hum'] = df['sex'].map({0: 'female', 1: 'male'})
df_encoded = pd.get_dummies(df, columns=['hum']).astype(int)

df_encoded.drop(columns=['sex'], inplace=True)

print(df[['sex','hum']].head())
print("\nOne-Hot Encoding:")
print(df_encoded.filter(like='hum').head())

#5
res = df.groupby('target')['chol'].mean()
print(res)

#6
features = ['age', 'trestbps', 'chol', 'thalach']

for column in features:
    df[column] = (( df[column] - df[column].min()) /
    (df[column].max() - df[column].min()))

print("После Min-Max нормализации:")

print(df.describe())

