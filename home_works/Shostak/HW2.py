#пункт 1
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv("heart.csv")
print(df.head())
print(df.isnull().sum())
print(df['target'].value_counts())

#пункт 2
df['target'].value_counts().plot(kind='bar')
plt.xlabel('наличие болезни сердца')
plt.ylabel('количество пациентов')
plt.title('количество здоровых и больных')
plt.show()

#пункт 3
plt.figure(figsize=(8 ,6))
sns.scatterplot(
    data=df,
    x='age',
    y='thalach',
    hue='target'
)
plt.xlabel('возраст')
plt.ylabel('максимальный пульс')
plt.title('зависимость пульса от возраста')
plt.show()

#пункт 4
df['sex'] = df['sex'].map({0: 'female' , 1: 'male'})
df['sex'].value_counts()
df = pd.get_dummies(df, columns=['sex'])
df[['sex_female' , 'sex_male']] = df[['sex_female' , 'sex_male']].astype(int)
print(df.head())

#пункт 5
df.groupby('target')['chol'].mean()
chol_mean = df.groupby('target')['chol'].mean()
print(f'Средний холестерин у здоровых: {chol_mean[0]: .2f}')
print(f'Средний холестерин у больных: {chol_mean[1]: .2f}')

#пункт 6
cols_normalize = ['age', 'trestbps', 'chol', 'thalach']
df[cols_normalize] = (df[cols_normalize] - df[cols_normalize].min()) / (df[cols_normalize].max() - df[cols_normalize].min())
print(df[cols_normalize].describe())

