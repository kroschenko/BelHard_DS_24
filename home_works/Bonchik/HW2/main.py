import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')


def task1():
    print("=" * 50)
    print("ЗАДАНИЕ 1: Загрузка данных и проверка пропусков")
    print("=" * 50)

    print("\nПервые 5 строк данных:")
    print(df.head())

    print("\nОбщая информация:")
    df.info()

    print("\nПроверка на пропущенные значения:")
    print("-" * 50)
    print("Количество пропусков по столбцам:")
    print(df.isnull().sum())
    print("-" * 50)
    total_missing = df.isnull().sum().sum()
    print(f"Общее количество пропусков: {total_missing}")
    return df


def task2():
    print("=" * 50)
    print("ЗАДАНИЕ 2: Столбчатая диаграмма здоровые vs больные")
    print("=" * 50)

    healthy_count = (df['target'] == 0).sum()
    diseased_count = (df['target'] == 1).sum()

    print(f"Здоровые пациенты (target=0): {healthy_count}")
    print(f"Больные пациенты (target=1): {diseased_count}")

    plt.figure(figsize=(8, 6))
    categories = ['Здоровые', 'Больные']
    counts = [healthy_count, diseased_count]
    colors = ['lightblue', 'lightcoral']

    bars = plt.bar(categories, counts, color=colors, edgecolor='black')
    plt.title('Количество здоровых и больных пациентов')
    plt.xlabel('Категория пациента')
    plt.ylabel('Количество пациентов')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
    return df


def task3():
    print("=" * 50)
    print("ЗАДАНИЕ 3: Диаграмма рассеяния пульс vs возраст")
    print("=" * 50)

    plt.figure(figsize=(10, 6))
    healthy = df[df['target'] == 0]
    sick = df[df['target'] == 1]

    plt.scatter(healthy['age'], healthy['thalach'],
                color='lightblue', alpha=0.7, s=50,
                label='Здоровые (target=0)')
    plt.scatter(sick['age'], sick['thalach'],
                color='lightcoral', alpha=0.7, s=50,
                label='Больные (target=1)')

    plt.title('Зависимость максимального пульса от возраста', fontsize=14)
    plt.xlabel('Возраст (age)', fontsize=12)
    plt.ylabel('Максимальный пульс (thalach)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return df


def task4():
    print("=" * 50)
    print("ЗАДАНИЕ 4: Преобразование признака sex и One-Hot Encoding")
    print("=" * 50)

    print("\nТекущие значения в столбце 'sex':")
    print(df['sex'].unique())
    print(f"0 = женщина, 1 = мужчина")

    df['sex_category'] = df['sex'].map({0: 'female', 1: 'male'})
    print("\nПреобразованный столбец 'sex_category':")
    print(df[['sex', 'sex_category']].head(10))

    sex_encoded = pd.get_dummies(df['sex_category'], prefix='sex')
    df_new = pd.concat([df, sex_encoded], axis=1)

    print("\nРезультат One-Hot Encoding:")
    print(df_new[['sex', 'sex_category', 'sex_female', 'sex_male']].head(10))

    print("\nПроверка преобразования:")
    print(f"Количество женщин (sex_female=1): {df_new['sex_female'].sum()}")
    print(f"Количество мужчин (sex_male=1): {df_new['sex_male'].sum()}")
    return df_new


def task5():
    print("=" * 50)
    print("ЗАДАНИЕ 5: Средний уровень холестерина")
    print("=" * 50)

    mean_chol_healthy = df[df['target'] == 0]['chol'].mean()
    mean_chol_sick = df[df['target'] == 1]['chol'].mean()

    print(f"\nСредний уровень холестерина (chol):")
    print(f"У здоровых пациентов (target=0): {mean_chol_healthy:.2f}")
    print(f"У больных пациентов (target=1): {mean_chol_sick:.2f}")

    diff = mean_chol_sick - mean_chol_healthy
    print(f"\nРазница: {diff:.2f}")
    return df


def task6():
    print("=" * 50)
    print("ЗАДАНИЕ 6: Нормализация признаков")
    print("=" * 50)

    features_to_normalize = ['age', 'trestbps', 'chol', 'thalach']
    print(f"\nПризнаки для нормализации: {features_to_normalize}")

    for feature in features_to_normalize:
        min_val = df[feature].min()
        max_val = df[feature].max()
        df[f'norm_minmax_{feature}'] = (df[feature] - min_val) / (max_val - min_val)

        mean_val = df[feature].mean()
        std_val = df[feature].std()
        df[f'norm_mean_{feature}'] = (df[feature] - mean_val) / std_val

    norm_minmax_cols = [f'norm_minmax_{f}' for f in features_to_normalize]
    norm_mean_cols = [f'norm_mean_{f}' for f in features_to_normalize]

    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
    fig1.suptitle('Сравнение распределений до и после Min-Max нормализации', fontsize=16)

    for i, feature in enumerate(features_to_normalize):
        axes1[0, i].hist(df[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes1[0, i].set_title(f'Исходный: {feature}')
        axes1[1, i].hist(df[f'norm_minmax_{feature}'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes1[1, i].set_title(f'Min-Max: {feature}')

    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle('Сравнение распределений до и после нормализации средним', fontsize=16)

    for i, feature in enumerate(features_to_normalize):
        axes2[0, i].hist(df[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes2[0, i].set_title(f'Исходный: {feature}')
        axes2[1, i].hist(df[f'norm_mean_{feature}'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes2[1, i].set_title(f'Нормализация средним: {feature}')

    plt.tight_layout()
    plt.show()

    print("\nПроверка корректности нормализации:")
    print("\nMin-Max нормализация (диапазон [0, 1]):")
    for feature in features_to_normalize:
        col_name = f'norm_minmax_{feature}'
        min_val = df[col_name].min()
        max_val = df[col_name].max()
        print(f"  {col_name}: min={min_val:.4f}, max={max_val:.4f}")

    print("\nНормализация средним (среднее ≈ 0, std ≈ 1):")
    for feature in features_to_normalize:
        col_name = f'norm_mean_{feature}'
        mean_val = df[col_name].mean()
        std_val = df[col_name].std()
        print(f"  {col_name}: mean={mean_val:.4f}, std={std_val:.4f}")

    return df


def main_menu():
    global df

    while True:
        print("\n" + "=" * 60)
        print("АНАЛИЗ ДАННЫХ HEART DISEASE")
        print("=" * 60)
        print("1. Загрузка данных и проверка пропусков")
        print("2. Столбчатая диаграмма: здоровые vs больные")
        print("3. Диаграмма рассеяния: пульс vs возраст")
        print("4. Преобразование признака sex и One-Hot Encoding")
        print("5. Средний уровень холестерина для здоровых и больных")
        print("6. Нормализация признаков")
        print("0. Выход")
        print("=" * 60)

        choice = input("\nВыберите задание (0-6): ").strip()

        if choice == '0':
            print("Выход из программы...")
            break
        elif choice == '1':
            df = task1()
        elif choice == '2':
            df = task2()
        elif choice == '3':
            df = task3()
        elif choice == '4':
            df = task4()
        elif choice == '5':
            df = task5()
        elif choice == '6':
            df = task6()
        else:
            print("Неверный выбор. Попробуйте снова.")

        input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    main_menu()