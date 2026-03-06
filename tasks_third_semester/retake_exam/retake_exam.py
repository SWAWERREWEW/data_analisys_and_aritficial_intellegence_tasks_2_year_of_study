def retake_exam1():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    import random
    import os

    # Устанавливаем случайное зерно для воспроизводимости
    np.random.seed(42)
    random.seed(42)

    def create_digit_image(digit, img_size=(16, 16)):
        """
        Создает изображение цифры на сером фоне

        Parameters:
        -----------
        digit : int
            Цифра от 0 до 9
        img_size : tuple
            Размер изображения (ширина, высота)

        Returns:
        --------
        image : PIL.Image
            Изображение цифры
        """
        # Создаем серое изображение
        img = Image.new('L', img_size, color=128)  # 128 - средний серый
        draw = ImageDraw.Draw(img)

        # Пытаемся использовать шрифт, если нет - используем стандартный
        try:
            # Пробуем загрузить шрифт
            font = ImageFont.truetype("arial.ttf", size=12)
        except:
            font = ImageFont.load_default()

        # Рисуем цифру (белым цветом на сером фоне)
        # Центрируем цифру
        draw.text((2, 1), str(digit), fill=255, font=font)

        return img

    def create_dataset(num_samples_per_class=100, img_size=(16, 16)):
        """
        Создает набор данных изображений цифр

        Parameters:
        -----------
        num_samples_per_class : int
            Количество изображений для каждой цифры (0-9)
        img_size : tuple
            Размер изображений

        Returns:
        --------
        images : list
            Список изображений PIL
        labels : list
            Список меток (цифр)
        """
        images = []
        labels = []

        for digit in range(10):  # Цифры от 0 до 9
            for _ in range(num_samples_per_class):
                img = create_digit_image(digit, img_size)
                images.append(img)
                labels.append(digit)

        return images, labels

    def visualize_dataset(images, labels, num_samples_per_class=5):
        """
        Визуализирует примеры изображений для каждого класса

        Parameters:
        -----------
        images : list
            Список изображений
        labels : list
            Список меток
        num_samples_per_class : int
            Сколько примеров показать для каждого класса
        """
        # Создаем словарь с изображениями для каждого класса
        class_images = {i: [] for i in range(10)}

        for img, label in zip(images, labels):
            class_images[label].append(img)

        # Выбираем случайные примеры для каждого класса
        fig, axes = plt.subplots(10, num_samples_per_class, figsize=(10, 10))

        for digit in range(10):
            # Выбираем случайные индексы
            if len(class_images[digit]) >= num_samples_per_class:
                selected_indices = random.sample(range(len(class_images[digit])), num_samples_per_class)
            else:
                selected_indices = list(range(len(class_images[digit])))

            for idx, sample_idx in enumerate(selected_indices):
                ax = axes[digit, idx] if num_samples_per_class > 1 else axes[digit]
                ax.imshow(class_images[digit][sample_idx], cmap='gray')
                ax.axis('off')

                # Добавляем метку только для первого столбца
                if idx == 0:
                    ax.set_ylabel(f'Цифра {digit}', fontsize=9)

        plt.suptitle('Примеры изображений для каждого класса (0-9)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    # ==================== ВЫПОЛНЕНИЕ ====================

    print("Создание набора данных...")
    # Создаем dataset: 100 изображений для каждой цифры
    images, labels = create_dataset(num_samples_per_class=100, img_size=(16, 16))

    print(f"Всего создано изображений: {len(images)}")
    print(f"Количество классов: {len(set(labels))}")
    print(f"Размер каждого изображения: 16x16 пикселей")

    # Проверяем распределение классов
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nРаспределение по классам: {dict(sorted(label_counts.items()))}")

    print("\nВизуализация примеров (5 случайных изображений для каждого класса):")
    # Визуализируем 5 случайных примеров для каждого класса
    visualize_dataset(images, labels, num_samples_per_class=5)

    print("\nГотово! Набор данных создан и визуализирован.")


def retake_exam2():
    # Импорт необходимых библиотек
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')

    # =============================================================================
    # ЧАСТЬ 1: ПРЕДОБРАБОТКА ДАННЫХ
    # =============================================================================

    # Для демонстрации создадим пример данных (замените на свои данные!)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=256, n_classes=3,
                               n_informative=200, random_state=42)

    print("Исходная форма данных:", X.shape)
    print("Количество образцов:", X.shape[0])
    print("Количество признаков:", X.shape[1])

    # Преобразование каждого изображения в одномерный массив (если нужно)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)  # Flatten всех изображений

    print("Форма после преобразования:", X.shape)

    # Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Разделение на обучающую и тестовую выборки (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nРазмер обучающей выборки:", X_train.shape[0])
    print("Размер тестовой выборки:", X_test.shape[0])

    # =============================================================================
    # ЧАСТЬ 2: ОБУЧЕНИЕ И ОЦЕНКА ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ
    # =============================================================================

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 60)

    # ИСПРАВЛЕНИЕ: убран устаревший параметр multi_class
    log_reg = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',  # Поддерживает многоклассовую классификацию
        random_state=42,
        n_jobs=-1  # Используем все ядра процессора
    )

    # Обучение модели
    log_reg.fit(X_train, y_train)
    print("Модель успешно обучена!")

    # Предсказание
    y_pred = log_reg.predict(X_test)

    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность на тестовой выборке: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # =============================================================================
    # ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    # =============================================================================

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)

    # Создание figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Матрица ошибок
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Матрица ошибок', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Предсказанный класс')
    axes[0].set_ylabel('Истинный класс')

    # 2. Нормализованная матрица ошибок
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Greens', ax=axes[1])
    axes[1].set_title('Нормализованная матрица ошибок (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')

    plt.tight_layout()
    plt.show()

    # Отчет о классификации
    print("\n" + "=" * 60)
    print("ОТЧЕТ О КЛАССИФИКАЦИИ")
    print("=" * 60)
    report = classification_report(y_test, y_pred)
    print(report)

    print("\n" + "=" * 60)
    print("ЗАДАНИЕ ВЫПОЛНЕНО УСПЕШНО!")
    print("=" * 60)


def retake_exam_full():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import seaborn as sns
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # ==========================================
    # Часть А: Создание набора данных
    # ==========================================

    def generate_synthetic_data(num_samples=1000, img_size=(16, 16)):
        X = []
        y = []

        for i in range(num_samples):
            digit = i % 10  # Равномерное распределение цифр от 0 до 9
            # Создаем пустое серое изображение (0 - черный)
            img = Image.new('L', img_size, color=0)
            draw = ImageDraw.Draw(img)

            # Используем стандартный шрифт (может отличаться в разных ОС)
            try:
                font = ImageFont.load_default()
            except:
                font = None

            # Добавим небольшое случайное смещение для разнообразия
            offset = (random.randint(2, 5), random.randint(1, 3))
            draw.text(offset, str(digit), fill=255, font=font)

            X.append(np.array(img))
            y.append(digit)

        return np.array(X), np.array(y)

    # 1. Создаем набор
    X, y = generate_synthetic_data()

    # 2. Визуализация примеров
    def visualize_classes(X, y):
        plt.figure(figsize=(12, 8))
        for digit in range(10):
            # Находим индексы всех изображений данного класса
            indices = np.where(y == digit)[0]
            # Выбираем 5 случайных
            selected_indices = np.random.choice(indices, 5, replace=False)

            for i, idx in enumerate(selected_indices):
                plt.subplot(5, 10, i * 10 + digit + 1)
                plt.imshow(X[idx], cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title(f"L: {digit}")
        plt.tight_layout()
        plt.show()

    visualize_classes(X, y)

    # ==========================================
    # Часть В: Логистическая регрессия
    # ==========================================

    # 1. Предобработка
    # Превращаем 16x16 в одномерный массив 256
    X_flat = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

    # 2. Обучение и оценка
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    print("--- Отчет по Логистической регрессии ---")
    print(f"Точность (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Матрица ошибок (тепловая карта)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок: Логистическая регрессия')
    plt.xlabel('Предсказано')
    plt.ylabel('Реальность')
    plt.show()

    # ==========================================
    # Часть С: Сверточная нейронная сеть (CNN)
    # ==========================================

    # 1. Подготовка данных для CNN
    # Изменяем форму на (samples, 16, 16, 1) и нормализуем (0-1)
    X_cnn = X.reshape(-1, 16, 16, 1) / 255.0
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

    # 2. Построение модели
    model = models.Sequential([
        # Сверточный слой
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)),
        # Слой подвыборки (Pooling)
        layers.MaxPooling2D((2, 2)),
        # Преобразование в вектор
        layers.Flatten(),
        # Полносвязный выходной слой (10 нейронов для 10 цифр)
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Обучение
    history = model.fit(X_train_c, y_train_c, epochs=15,
                        validation_data=(X_test_c, y_test_c), verbose=1)

    # Оценка
    test_loss, test_acc = model.evaluate(X_test_c, y_test_c, verbose=0)
    print(f"\nТочность CNN на тестовой выборке: {test_acc:.4f}")

    # Графики точности
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.title('График точности CNN')
    plt.show()


def retake_exam():
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    # ============================================================
    # 1. ПОДГОТОВКА ДАННЫХ ДЛЯ CNN
    # ============================================================

    # Загрузка данных MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Преобразование изображений к виду (samples, 16, 16, 1)
    # ИСПРАВЛЕНИЕ: используем tf.image.resize вместо keras.preprocessing.image.resize
    x_train = tf.image.resize(x_train[..., np.newaxis], (16, 16)).numpy()
    x_test = tf.image.resize(x_test[..., np.newaxis], (16, 16)).numpy()

    # Теперь x_train имеет форму (samples, 16, 16, 1)
    print(f"Форма тренировочных данных: {x_train.shape}")
    print(f"Форма тестовых данных: {x_test.shape}")

    # Нормализация значений пикселей (от 0 до 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Преобразование меток в one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # ============================================================
    # 2. ПОСТРОЕНИЕ ПРОСТОЙ CNN
    # ============================================================

    model = keras.Sequential([
        # Свёрточный слой
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(16, 16, 1),
            padding='same'
        ),

        # Слой подвыборки (pooling)
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Ещё один свёрточный слой
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        ),

        # Ещё один pooling слой
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten слой
        layers.Flatten(),

        # Полносвязный слой
        layers.Dense(128, activation='relu'),

        # Выходной полносвязный слой
        layers.Dense(10, activation='softmax')
    ])

    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Вывод структуры модели
    model.summary()

    # ============================================================
    # 3. ОБУЧЕНИЕ CNN
    # ============================================================

    from sklearn.model_selection import train_test_split

    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    # Обучение модели
    history = model.fit(
        x_train_final, y_train_final,
        batch_size=128,
        epochs=10,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # ============================================================
    # 4. ОЦЕНКА ТОЧНОСТИ НА ТЕСТОВОЙ ВЫБОРКЕ
    # ============================================================

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nТочность на тестовой выборке: {test_accuracy:.4f}")
    print(f"Потери на тестовой выборке: {test_loss:.4f}")

    # ============================================================
    # 5. ПОСТРОЕНИЕ ГРАФИКОВ ТОЧНОСТИ
    # ============================================================

    plt.figure(figsize=(12, 4))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()