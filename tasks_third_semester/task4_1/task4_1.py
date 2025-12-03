def tasks4_1():
    """
Упражнение - 4
Задание 4.1. Бинарная классификация с использованием сверточной нейронной сети (CNN) для предсказания пола пассажира.

Вам предоставлен набор данных Titanic. Ваша задача — написать Python-скрипт для построения сверточной нейронной сети
(CNN) для решения задачи бинарной классификации. Цель — предсказать пол («Мужчина» или «Женщина») пассажиров Titanic на
основе различных признаков. Входные признаки: используйте следующие признаки из набора данных Titanic в качестве
входных для модели CNN: Survived, Pclass, Age, SibSp, Parch, Fare, Embarked_Q и Embarked_S.

Набор данных: https://icedrive.net/s/6YBzzgF2B1fCBNx59vVR5kX41PZP

Инструкции:
Импортируйте необходимые библиотеки, включая pandas, numpy, модули scikit-learn, matplotlib, seaborn, tensorflow и
keras. Загрузите набор данных Titanic из CSV-файла.
Предобработка данных:
Удалите ненужные столбцы, такие как Name, Ticket и Cabin.
Заполните пропущенные значения.
Выполните one-hot кодирование категориальных признаков (например, Embarked).
Нормализуйте числовые признаки (Age, Fare).
Разделите данные на обучающую и тестовую выборки.
Создайте модель CNN с использованием TensorFlow/Keras или PyTorch:
Входной слой.
Сверточные и пуллинговые слои.
Полносвязные слои.
Выходной слой с 2 нейронами (softmax-активация).
Обучите модель, используя функцию потерь categorical_crossentropy и оптимизатор adam. Обучите модель на обучающих
данных и проверьте её на тестовых данных. Выведите графики точности и потерь на обучении и валидации по эпохам, матрицу
ошибок и отчет классификации.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import confusion_matrix, classification_report

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical

    np.random.seed(42)
    tf.random.set_seed(42)

    Titanic_data = sns.load_dataset('titanic')

    print("Первые 5 строк датасета Titanic:")
    print(Titanic_data.head())
    print("\nОсновная информация о датасете:")
    Titanic_data.info()


    # Удаление ненужных столбцов
    columns_to_drop_if_exist = ['name', 'ticket', 'cabin', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive',
                                'alone']
    Titanic_data = Titanic_data.drop(columns=[col for col in columns_to_drop_if_exist if col in Titanic_data.columns])

    # Преобразуем целевую переменную 'sex' в числовой формат (0 для male, 1 для female)
    Titanic_data['sex_encoded'] = Titanic_data['sex'].map({'male': 0, 'female': 1})
    y = to_categorical(Titanic_data['sex_encoded'], num_classes=2)

    # Определяем входные признаки
    feature_columns = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = Titanic_data[feature_columns].copy()

    # Заполнение пропущенных значений и One-Hot кодирование для Embarked
    # Используем ColumnTransformer для обработки различных типов столбцов.

    numerical_features = ['age', 'fare']
    categorical_features = ['embarked']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Заполнение NaN в Embarked модой
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Объединяем с ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Создаем список дополнительных признаков, которые не будут масштабироваться или кодироваться
    # Примечание: Pclass - категориальный признак, но в задании он не указан для one-hot кодирования.
    # Будем обрабатывать его как числовой, если не указано иное.
    passthrough_features = ['survived', 'pclass', 'sibsp', 'parch']

    # Применяем предобработку
    X_processed_temp = preprocessor.fit_transform(X)

    # Получаем имена столбцов после one-hot кодирования
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names) + passthrough_features

    X_processed_df = pd.DataFrame(X_processed_temp, columns=all_feature_names)

    # (Survived, Pclass, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S)
    # Имена столбцов 'age', 'fare' будут нормализованы, 'embarked_Q', 'embarked_S' будут one-hot закодированы.
    # 'survived', 'pclass', 'sibsp', 'parch' будут взяты как есть.

    final_features = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']

    # Проверяем, какие Embarked столбцы были созданы и добавляем их, если они Q или S
    for col in ohe_feature_names:
        if 'embarked_Q' in col:
            final_features.append(col)
        if 'embarked_S' in col:
            final_features.append(col)

    # Фильтруем X_processed_df, чтобы использовать только нужные столбцы
    X_final = X_processed_df[final_features]

    # Проверяем, что все требуемые столбцы присутствуют
    required_features_check = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
    if 'embarked_Q' in X_final.columns:
        required_features_check.append('embarked_Q')
    if 'embarked_S' in X_final.columns:
        required_features_check.append('embarked_S')

    missing_required = [f for f in required_features_check if f not in X_final.columns]
    if missing_required:
        print("""Внимание: Не удалось найти следующие требуемые признаки: """ + f"{missing_required}" +
". Возможно, они не были сгенерированы в процессе one-hot кодирования (например, если нет пассажиров из Q или S).")

        # Удаляем отсутствующие признаки из списка final_features
        final_features = [f for f in final_features if f not in missing_required]
        X_final = X_final[final_features]

    print("\nКонечные признаки для модели:")
    print(X_final.head())
    print(f"Количество признаков: {X_final.shape[1]}")

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

    # Изменение формы данных для CNN
    X_train_reshaped = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"\nФорма обучающей выборки для CNN: {X_train_reshaped.shape}")
    print(f"Форма тестовой выборки для CNN: {X_test_reshaped.shape}")
    print(f"Форма целевой обучающей выборки: {y_train.shape}")
    print(f"Форма целевой тестовой выборки: {y_test.shape}")

    #Создание модели CNN
    model = Sequential([

        # Входной слой и первый сверточный слой.
        # Добавляем Dropout для регуляризации
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Второй сверточный слой
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Преобразование в одномерный вектор для полносвязных слоев
        Flatten(),

        # Полносвязные слои
        # Добавляем Dropout
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),

        # Выходной слой: 2 нейрона с softmax активацией для бинарной классификации
        Dense(2, activation='softmax')
    ])

    #Компиляция модели
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # бучение модели
    print("\nНачало обучения модели...")
    history = model.fit(X_train_reshaped, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_test_reshaped, y_test),
                        verbose=1)

    # Оценка модели
    print("\nОценка модели на тестовых данных:")
    loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"Потери на тестовых данных: {loss:.4f}")
    print(f"Точность на тестовых данных: {accuracy:.4f}")

    # Визуализация результатов

    # Графики точности и потерь
    plt.figure(figsize=(12, 5))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title('Точность модели по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('Потери модели по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Матрица ошибок и отчет классификации
    y_pred_probs = model.predict(X_test_reshaped)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Матрица ошибок
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Мужчина', 'Женщина'], yticklabels=['Мужчина', 'Женщина'])
    plt.xlabel('Предсказанный пол')
    plt.ylabel('Истинный пол')
    plt.title('Матрица ошибок')
    plt.show()

    # Отчет классификации
    print("\nОтчет классификации:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Мужчина', 'Женщина']))


if __name__ == '__main__': tasks4_1()