def task3_1():
    """
Задание 3.1. Логистическая регрессия, Случайный лес и Дерево решений для предсказания пола пассажира

Вам предоставлен датасет Titanic. Ваша задача — написать скрипт на Python для построения моделей машинного обучения
для предсказания пола пассажира.

Инструкции:
Импортируйте необходимые библиотеки, включая pandas, numpy, модули scikit-learn, matplotlib и seaborn.
Загрузите датасет Titanic из CSV-файла.
Предобработайте данные:
Удалите неактуальные столбцы.
Обработайте пропущенные значения.
Выполните one-hot кодирование категориальных переменных.
Подготовьте признаки (X) и целевую переменную (y), где y — это пол (1 для мужчин, 0 для женщин).
Разделите данные на обучающую и тестовую выборки.
Масштабируйте признаки с помощью StandardScaler.
Обучите три модели: Логистическая регрессия, Случайный лес и Дерево решений.
Выведите точность, матрицу ошибок в виде тепловой карты и отчет классификации для каждой модели на тестовой выборке.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    titanic_data = sns.load_dataset('titanic')
    print("Датасет Titanic загружен.")

    print("\nПервые 5 строк датасета:")
    print(titanic_data.head())
    print("\nОсновная информация о датасете:")
    titanic_data.info()
    print("\nОписательная статистика для числовых столбцов:")
    print(titanic_data.describe())

    # Удаление неактуальных столбцов
    columns_to_drop = ['who', 'adult_male', 'alive', 'survived', 'deck', 'class', 'embarked', 'alone']
    titanic_data_processed = titanic_data.drop(columns=columns_to_drop)

    print(f"\nУдалены столбцы: {columns_to_drop}")
    print("Оставшиеся столбцы:", titanic_data_processed.columns.tolist())

    # - 'age': Числовой, заполним медианой.
    # - 'embark_town': Категориальный, заполним модой.
    imputer_age = SimpleImputer(strategy='median')
    imputer_embark_town = SimpleImputer(strategy='most_frequent')

    titanic_data_processed['age'] = imputer_age.fit_transform(titanic_data_processed[['age']])
    titanic_data_processed['embark_town'] = imputer_embark_town.fit_transform(titanic_data_processed[['embark_town']]).flatten()

    print("\nПропущенные значения после обработки:")
    print(titanic_data_processed.isnull().sum())

    # One-hot кодирование категориальных переменных
    categorical_cols = ['pclass', 'embark_town']
    titanic_data_processed = pd.get_dummies(titanic_data_processed, columns=categorical_cols, drop_first=True)

    print("\nДатасет после One-hot кодирования:")
    print(titanic_data_processed.head())
    print("Столбцы после кодирования:", titanic_data_processed.columns.tolist())

    # Подготовка признаков (X) и целевой переменной (y)
    # Целевая переменная 'sex': 1 для мужчин, 0 для женщин.
    # Признаки X: все остальные столбцы.

    # Преобразование 'sex' в числовой формат
    titanic_data_processed['sex'] = titanic_data_processed['sex'].map({'male': 1, 'female': 0})
    y = titanic_data_processed['sex']
    X = titanic_data_processed.drop(columns=['sex'])

    print("\nРазмерность X:", X.shape)
    print("Размерность y:", y.shape)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nРазмер обучающей выборки (X_train): {X_train.shape}")
    print(f"Размер тестовой выборки (X_test): {X_test.shape}")
    print(f"Размер обучающей целевой переменной (y_train): {y_train.shape}")
    print(f"Размер тестовой целевой переменной (y_test): {y_test.shape}")

    # Масштабирование признаков с помощью StandardScaler
    scaler = StandardScaler()

    # Обучаем scaler только на обучающих данных и применяем к обеим выборкам
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Преобразуем обратно в DataFrame для удобства, сохраняя названия столбцов
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    print("\nПризнаки масштабированы с помощью StandardScaler.")
    print("Пример масштабированных признаков (X_train_scaled head):")
    print(X_train_scaled.head())

    # Обучение и оценка моделей
    models = {
        'Логистическая регрессия': LogisticRegression(random_state=42),
        'Дерево решений': DecisionTreeClassifier(random_state=42),
        'Случайный лес': RandomForestClassifier(random_state=42)
    }

    results = {}

    print("\nОбучение и оценка моделей")

    for name, model in models.items():
        print(f"\nМодель: {name}")

        # Обучение модели
        model.fit(X_train_scaled, y_train)

        # Предсказание на тестовой выборке
        y_pred = model.predict(X_test_scaled)

        # Вычисление метрик
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=['Female', 'Male'])

        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

        # Вывод точности
        print(f"Точность (Accuracy): {accuracy:.4f}")

        # Вывод матрицы ошибок в виде тепловой карты
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Предсказано Female', 'Предсказано Male'],
                    yticklabels=['Факт Female', 'Факт Male'])
        plt.title(f'Матрица ошибок для {name}')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.show()

        # Вывод отчета классификации
        print("\nОтчет классификации:\n", class_report)

    print("Завершено")

if __name__ == '__main__': task3_1()