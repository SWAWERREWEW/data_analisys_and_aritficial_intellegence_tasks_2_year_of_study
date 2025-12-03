def task2_1():
    """Задание 2.1. Логистическая регрессия, случайный лес и дерево решений для предсказания возрастной группы
пассажира. Вам предоставлен датасет Titanic. Ваша задача — написать скрипт на Python для оценки моделей машинного
обучения, которые предсказывают возрастную группу пассажира.
Инструкции:
Импортируйте необходимые библиотеки, включая pandas, numpy, scikit-learn, matplotlib и seaborn.
Загрузите датасет Titanic из CSV-файла.
Предобработайте датасет Titanic:
Удалите столбцы, не полезные для предсказания (например: 'PassengerId', 'Name', 'Ticket', 'Cabin').
Заполните пропущенные значения в датасете (например, замените пропущенные значения возраста на средний возраст).
Преобразуйте категориальные столбцы (такие как 'Sex' и 'Embarked') в числа с помощью one-hot кодирования.
Создайте новую целевую переменную 'Age_Category' с помощью pd.cut() со следующими интервалами:
0: 0-18 лет (дети и подростки)
1: 19-30 лет (молодые взрослые)
2: 31-50 лет (взрослые среднего возраста)
3: 51+ лет (пожилые)
Используйте эти интервалы для присвоения каждому пассажиру возрастной категории.
Разделите данные на обучающую и тестовую выборки с помощью train-test split.
Обучите три модели: логистическую регрессию, случайный лес и дерево решений.
Выведите точность, матрицу ошибок в виде тепловой карты и отчет классификации для каждой модели на тестовой выборке.

https://icedrive.net/s/6YBzzgF2B1fCBNx59vVR5kX41PZP
Titanic_dataset.csv - Icedrive"""
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("Загрузка датасета Titanic...")
    titanic_data = sns.load_dataset('titanic')

    print("\nПервые 5 строк датасета Titanic:")
    print(titanic_data.head())

    print("\nОсновная информация о датасете:")
    titanic_data.info()

    print("\nОписательная статистика для числовых столбцов:")
    print(titanic_data.describe())

    print("\nНачало предобработки датасета...")

    # Удаление столбцов, не полезных для предсказания
    # Удаляем 'survived', так как это была бы утечка данных для предсказания возраста.
    # Удаляем 'class' и 'embark_town' как дубликаты 'pclass' и 'embarked'.
    # Удаляем 'who' и 'alive' по аналогичным причинам.
    # Мы сохраняем pclass, sibsp, parch, fare, alone, adult_male, так как они могут быть полезными признаками.
    columns_to_drop = ['deck', 'survived', 'class', 'who', 'embark_town', 'alive']
    titanic_data = titanic_data.drop(columns=columns_to_drop)
    print(f"Удалены столбцы: {columns_to_drop}")

    # Заполнение пропущенных значений возраста на средний возраст
    titanic_data['age'].fillna(titanic_data['age'].mean(), inplace=True)
    print(f"Пропущенные значения в 'age' заполнены средним: {titanic_data['age'].mean():.2f}")

    # Заполнение пропущенных значений в 'embarked' модой
    titanic_data['embarked'].fillna(titanic_data['embarked'].mode()[0], inplace=True)
    print(f"Пропущенные значения в 'embarked' заполнены модой: {titanic_data['embarked'].mode()[0]}")

    print("\nПроверка пропущенных значений после обработки:")
    print(titanic_data.isnull().sum())

    # Преобразование категориальных столбцов в числа с помощью one-hot кодирования
    # Выбираем категориальные столбцы для one-hot кодирования
    categorical_cols = ['sex', 'embarked']

    # Применяем One-Hot кодирование
    # drop_first=True для избежания мультиколлинеарности
    titanic_data = pd.get_dummies(titanic_data, columns=categorical_cols, drop_first=True)
    print(f"\nПрименено One-Hot кодирования для столбцов: {categorical_cols}")
    print("Датасет после One-Hot кодирования:")
    print(titanic_data.head())

    # Создание новой целевой переменной 'Age_Category'
    # Используем pd.cut() со следующими интервалами:
    # 0: 0-18 лет (дети и подростки)
    # 1: 19-30 лет (молодые взрослые)
    # 2: 31-50 лет (взрослые среднего возраста)
    # 3: 51+ лет (пожилые)

    bins = [0, 18, 30, 50, np.inf]
    labels = [0, 1, 2, 3]

    titanic_data['Age_Category'] = pd.cut(titanic_data['age'], bins=bins, labels=labels, right=True, include_lowest=True)
    print("\nСоздана целевая переменная 'Age_Category':")
    print(titanic_data[['age', 'Age_Category']].head())
    print("Распределение Age_Category:")
    print(titanic_data['Age_Category'].value_counts())

    # Удаляем оригинальный столбец 'age', так как мы создали 'Age_Category'
    titanic_data = titanic_data.drop('age', axis=1)
    print("Удален оригинальный столбец 'age'.")
    print("Количество столбцов после предобработки:", titanic_data.shape[1])
    print(titanic_data.head())

    X = titanic_data.drop('Age_Category', axis=1)
    y = titanic_data['Age_Category']

    # Преобразуем булевы столбцы в int (0 или 1) для моделей, если они еще не были преобразованы get_dummies
    for col in X.select_dtypes(include='bool').columns:
        X[col] = X[col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nРазделение данных на обучающую и тестовую выборки (test_size=0.2, random_state=42):")
    print(f"Размер обучающей выборки X_train: {X_train.shape}")
    print(f"Размер тестовой выборки X_test: {X_test.shape}")
    print(f"Распределение целевой переменной в обучающей выборке:\n{y_train.value_counts(normalize=True)}")
    print(f"Распределение целевой переменной в тестовой выборке:\n{y_test.value_counts(normalize=True)}")

    models = {
        'Логистическая регрессия': LogisticRegression(random_state=42, solver='liblinear', max_iter=200),
        'Дерево решений': DecisionTreeClassifier(random_state=42),
        'Случайный лес': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    results = {}

    print("\n--- Обучение и оценка моделей ---")

    for name, model in models.items():
        print(f"\nОбучение модели: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Оценка
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=[f'Категория {i}' for i in labels])

        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

        print(f"\nМодель: {name}")
        print(f"Точность (Accuracy): {accuracy:.4f}")
        print("\nМатрица ошибок:")
        print(conf_matrix)
        print("\nОтчет классификации:")
        print(class_report)

        # Вывод матрицы ошибок
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Категория {i}' for i in labels],
                    yticklabels=[f'Категория {i}' for i in labels])
        plt.xlabel('Предсказанная категория')
        plt.ylabel('Истинная категория')
        plt.title(f'Матрица ошибок для {name}')
        plt.show()

    print("\nВсе модели оценены")

if __name__ == '__main__': task2_1()
