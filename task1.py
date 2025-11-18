"""Обучение и оценка нейронных сетей для классификации MNIST."""

import json
import os
import sys
import time
from typing import Optional, Tuple, List

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# Константы
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODELS_DIR = 'models'
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
TRAIN_SIZE = 60000
TEST_SIZE = 10000

# Параметры обучения
EPOCHS = 15
BATCH_SIZE_1 = 128
BATCH_SIZE_2 = 64
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Имена файлов моделей
BEST_MODEL_FILE = 'best_mnist_model.h5'
TRAINING_MODEL_FILE = 'best_training_model.h5'
ARCHITECTURE_FILE = 'model_architecture.json'
WEIGHTS_FILE = 'model_weights.h5'
METADATA_FILE = 'model_metadata.json'


def setup_models_directory() -> None:
    """Создает директорию для сохранения моделей, если её нет."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает и возвращает данные MNIST.
    
    Returns:
        Кортеж (train_images, train_labels, test_images, test_labels)
    """
    print("Загрузка данных MNIST")
    (train_images, train_labels), (test_images, test_labels) = (
        keras.datasets.mnist.load_data()
    )
    return train_images, train_labels, test_images, test_labels


def preprocess_data(
    train_images: np.ndarray,
    test_images: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Предобрабатывает данные для обучения.
    
    Args:
        train_images: Обучающие изображения
        test_images: Тестовые изображения
        
    Returns:
        Кортеж предобработанных изображений
    """
    train_images = train_images.reshape((TRAIN_SIZE, INPUT_SIZE)).astype('float32') / 255
    test_images = test_images.reshape((TEST_SIZE, INPUT_SIZE)).astype('float32') / 255
    return train_images, test_images


def check_saved_model() -> Optional[Sequential]:
    """
    Проверяет наличие сохраненной модели и загружает её.
    
    Returns:
        Загруженная модель или None, если модель не найдена
    """
    model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILE)
    if not os.path.exists(model_path):
        return None
    
    print("Найдена сохраненная модель")
    try:
        model = keras.models.load_model(model_path)
        print("Модель успешно загружена")
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None


def test_model_predictions(
    model: Sequential,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    num_samples: int = 5
) -> None:
    """
    Тестирует модель на нескольких примерах.
    
    Args:
        model: Обученная модель
        test_images: Тестовые изображения
        test_labels: Тестовые метки
        num_samples: Количество примеров для тестирования
    """
    print(f"\nТестирование сохраненной модели")
    test_digits = test_images[:num_samples]
    predictions = model.predict(test_digits, verbose=0)
    
    print("Примеры предсказаний сохраненной модели:")
    for i in range(num_samples):
        actual = test_labels[i]
        predicted = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        status = "OK" if actual == predicted else "ERROR"
        print(
            f"Изображение {i}: Реальное={actual}, "
            f"Предсказанное={predicted}, "
            f"Уверенность={confidence:.4f} {status}"
        )


def should_retrain_model() -> bool:
    """
    Спрашивает у пользователя, хочет ли он переобучить модель.
    
    Returns:
        True, если пользователь хочет обучить новую модель
    """
    user_input = input("\nХотите обучить новую модель? (y/n): ").lower().strip()
    return user_input == 'y'


def create_improved_model() -> Sequential:
    """
    Создает улучшенную модель с Dropout слоями.
    
    Returns:
        Скомпилированная модель
    """
    model = Sequential([
        layers.Dense(1024, activation="relu", input_shape=(INPUT_SIZE,)),
        layers.Dropout(0.3),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    return model


def create_alternative_model() -> Sequential:
    """
    Создает альтернативную модель с BatchNormalization.
    
    Returns:
        Скомпилированная модель
    """
    model = Sequential([
        layers.Dense(784, activation="elu", input_shape=(INPUT_SIZE,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(392, activation="elu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(196, activation="elu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    return model


def create_callbacks() -> Tuple[ReduceLROnPlateau, EarlyStopping, ModelCheckpoint]:
    """
    Создает callbacks для обучения моделей.
    
    Returns:
        Кортеж из lr_scheduler, early_stopping, model_checkpoint
    """
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, TRAINING_MODEL_FILE),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    return lr_scheduler, early_stopping, model_checkpoint


def train_model(
    model: Sequential,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    callbacks: List,
    batch_size: int,
    model_name: str
) -> Tuple[Sequential, keras.callbacks.History, float]:
    """
    Обучает модель и возвращает результаты.
    
    Args:
        model: Модель для обучения
        train_images: Обучающие изображения
        train_labels: Обучающие метки
        callbacks: Список callbacks для обучения
        batch_size: Размер батча
        model_name: Имя модели для вывода
        
    Returns:
        Кортеж (обученная модель, история обучения, время обучения)
    """
    print(f"\n{model_name}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    start_time = time.time()
    history = model.fit(
        train_images,
        train_labels,
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    return model, history, training_time


def evaluate_model(
    model: Sequential,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    history: keras.callbacks.History,
    train_time: float,
    model_name: str
) -> Tuple[float, float, float, float]:
    """
    Оценивает модель и выводит результаты.
    
    Args:
        model: Обученная модель
        test_images: Тестовые изображения
        test_labels: Тестовые метки
        history: История обучения
        train_time: Время обучения
        model_name: Имя модели
        
    Returns:
        Кортеж (train_acc, val_acc, test_acc, overfitting)
    """
    print(f"\n{model_name}")
    print(f"Время обучения: {train_time:.2f} секунд")
    
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    overfitting = train_acc - test_acc
    
    print(f"Точность на обучающих данных: {train_acc:.4f}")
    print(f"Точность на валидационных данных: {val_acc:.4f}")
    print(f"Точность на тестовых данных: {test_acc:.4f}")
    print(f"Переобучение: {overfitting:.4f}")
    
    return train_acc, val_acc, test_acc, overfitting


def save_model(
    model: Sequential,
    accuracy: float,
    model_name: str
) -> None:
    """
    Сохраняет модель в нескольких форматах.
    
    Args:
        model: Модель для сохранения
        accuracy: Точность модели
        model_name: Имя модели
    """
    # Сохраняем полную модель
    model.save(os.path.join(MODELS_DIR, BEST_MODEL_FILE))
    
    # Сохраняем архитектуру
    architecture_path = os.path.join(MODELS_DIR, ARCHITECTURE_FILE)
    with open(architecture_path, 'w', encoding='utf-8') as f:
        f.write(model.to_json())
    
    # Сохраняем веса
    model.save_weights(os.path.join(MODELS_DIR, WEIGHTS_FILE))
    
    # Сохраняем метаданные
    metadata = {
        'accuracy': float(accuracy),
        'model_name': model_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape)
    }
    
    metadata_path = os.path.join(MODELS_DIR, METADATA_FILE)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nМодель сохранена в папке {MODELS_DIR} в нескольких форматах:")
    print(f"  - {BEST_MODEL_FILE} (полная модель)")
    print(f"  - {ARCHITECTURE_FILE} (архитектура)")
    print(f"  - {WEIGHTS_FILE} (веса)")
    print(f"  - {METADATA_FILE} (метаданные)")


def load_and_test_model(
    test_images: np.ndarray,
    test_labels: np.ndarray
) -> Optional[Sequential]:
    """
    Загружает и тестирует сохраненную модель.
    
    Args:
        test_images: Тестовые изображения
        test_labels: Тестовые метки
        
    Returns:
        Загруженная модель или None при ошибке
    """
    try:
        model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILE)
        model = keras.models.load_model(model_path)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print(f"\nПроверка загруженной модели:")
        print(f"Точность: {test_acc:.4f}")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None


def demonstrate_predictions(
    model: Sequential,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    indices: List[int]
) -> None:
    """
    Демонстрирует предсказания модели на выбранных примерах.
    
    Args:
        model: Обученная модель
        test_images: Тестовые изображения
        test_labels: Тестовые метки
        indices: Индексы примеров для демонстрации
    """
    print("\nПримеры предсказаний:")
    test_samples = test_images[indices]
    true_labels = test_labels[indices]
    
    predictions = model.predict(test_samples, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    for idx, true_label, pred_label, conf in zip(
        indices, true_labels, predicted_labels, confidences
    ):
        status = "OK" if true_label == pred_label else "ERROR"
        print(
            f"Изображение {idx}: {true_label} -> {pred_label} "
            f"(уверенность: {conf:.4f}) {status}"
        )


def print_usage_instructions() -> None:
    """Выводит инструкции по использованию сохраненной модели."""
    print("\nИнструкция по использованию")
    print("Для использования сохраненной модели в будущем:")
    print("1. Запустите скрипт снова - модель загрузится автоматически")
    print("2. Или используйте в своем коде:")
    print(f"""
from tensorflow import keras
model = keras.models.load_model('{MODELS_DIR}/{BEST_MODEL_FILE}')
predictions = model.predict(your_data)
""")


def main() -> None:
    """Основная функция для обучения и оценки моделей."""
    setup_models_directory()
    
    # Загрузка и предобработка данных
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    train_images, test_images = preprocess_data(train_images, test_images)
    
    # Проверка сохраненной модели
    saved_model = check_saved_model()
    
    if saved_model is not None:
        print("\nИспользование сохраненной модели")
        test_loss, test_acc = saved_model.evaluate(
            test_images, test_labels, verbose=0
        )
        print(f"Точность сохраненной модели на тестовых данных: {test_acc:.4f}")
        
        test_model_predictions(saved_model, test_images, test_labels)
        
        if not should_retrain_model():
            print("Используется сохраненная модель. Выход.")
            sys.exit(0)
    
    # Обучение новых моделей
    print("\nНачало обучения новых моделей")
    
    # Создание моделей
    model1 = create_improved_model()
    model2 = create_alternative_model()
    
    # Создание callbacks
    lr_scheduler, early_stopping, model_checkpoint = create_callbacks()
    
    # Обучение первой модели
    model1.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model1, history1, training_time1 = train_model(
        model1,
        train_images,
        train_labels,
        [lr_scheduler, early_stopping, model_checkpoint],
        BATCH_SIZE_1,
        "Модель 1: Улучшенная архитектура с Dropout"
    )
    
    # Обучение второй модели
    model2.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model2, history2, training_time2 = train_model(
        model2,
        train_images,
        train_labels,
        [early_stopping],
        BATCH_SIZE_2,
        "Модель 2: Архитектура с BatchNormalization"
    )
    
    # Оценка моделей
    print("\nОценка результатов")
    
    models_data = [
        (model1, "Улучшенная с Dropout", history1, training_time1),
        (model2, "С BatchNormalization", history2, training_time2),
    ]
    
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""
    
    for model, name, history, train_time in models_data:
        train_acc, val_acc, test_acc, _ = evaluate_model(
            model, test_images, test_labels, history, train_time, name
        )
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_model_name = name
    
    print(f"\nЛучшая модель: {best_model_name}")
    print(f"Точность: {best_accuracy:.4f}")
    
    # Сохранение лучшей модели
    if best_model is not None:
        save_model(best_model, best_accuracy, best_model_name)
    
    # Демонстрация работы модели
    print("\nДемонстрация работы модели")
    loaded_model = load_and_test_model(test_images, test_labels)
    
    if loaded_model is not None:
        demonstrate_predictions(
            loaded_model, test_images, test_labels, [0, 1, 2, 8, 13]
        )
    
    print_usage_instructions()


if __name__ == "__main__":
    main()
