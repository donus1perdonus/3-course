"""
Лабораторная работа №4. 8.2. Обучение сверточной нейронной сети с нуля 
на небольшом наборе данных.

Реализация кода из раздела 8.2 книги Франсуа Шолле
«Глубокое обучение на Python» с дополнениями и улучшениями.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore

# Делаем доступными утилиты сохранения моделей
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    save_model,
    load_model,
    load_model_metadata,
    list_saved_models,
)

# Параметры
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
ORIGINAL_DIR = pathlib.Path("dogs-vs-cats/train")
NEW_BASE_DIR = pathlib.Path("dogs-vs-cats/cats_vs_dogs_small")


def prepare_data() -> None:
    """
    Подготовка данных: копирование изображений в обучающий, проверочный 
    и контрольный каталоги (листинг 8.6).
    """
    print("Подготовка данных...")
    
    def make_subset(subset_name: str, start_index: int, end_index: int) -> None:
        """Создание подмножества данных."""
        for category in ("cat", "dog"):
            dir_path = NEW_BASE_DIR / subset_name / category
            os.makedirs(dir_path, exist_ok=True)
            fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
            for fname in fnames:
                src = ORIGINAL_DIR / fname
                dst = dir_path / fname
                if src.exists():
                    shutil.copyfile(src=src, dst=dst)
    
    # Создаем подмножества, если они еще не существуют
    if not (NEW_BASE_DIR / "train" / "cat").exists():
        make_subset("train", start_index=0, end_index=1000)
        make_subset("validation", start_index=1000, end_index=1500)
        make_subset("test", start_index=1500, end_index=2500)
        print("Данные подготовлены успешно!")
    else:
        print("Данные уже подготовлены.")


def load_datasets() -> Tuple[Any, Any, Any]:
    """
    Загрузка наборов данных (листинг 8.9).
    
    Returns:
        Кортеж (train_dataset, validation_dataset, test_dataset)
    """
    print("Загрузка наборов данных...")
    
    train_dataset = image_dataset_from_directory(
        NEW_BASE_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    validation_dataset = image_dataset_from_directory(
        NEW_BASE_DIR / "validation",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    test_dataset = image_dataset_from_directory(
        NEW_BASE_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Вывод информации о данных (листинг 8.10)
    for data_batch, labels_batch in train_dataset:
        print(f"Размер батча данных: {data_batch.shape}")
        print(f"Размер батча меток: {labels_batch.shape}")
        break
    
    return train_dataset, validation_dataset, test_dataset


def create_basic_model() -> keras.Model:
    """
    Создание базовой сверточной нейронной сети (листинг 8.7).
    
    Returns:
        Скомпилированная модель
    """
    print("Создание базовой модели...")
    
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Компиляция модели (листинг 8.8)
    model.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )
    
    return model


def create_improved_model() -> keras.Model:
    """
    Создание улучшенной модели с обогащением данных, batch normalization,
    улучшенной архитектурой и регуляризацией.
    
    Returns:
        Скомпилированная улучшенная модель
    """
    print("Создание улучшенной модели...")
    
    # Обогащение данных (листинг 8.14)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),  # Добавлено: случайный сдвиг
            layers.RandomBrightness(0.1),  # Добавлено: случайная яркость
        ]
    )
    
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    
    # Первый блок свертки с batch normalization
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)  # ELU вместо ReLU
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Второй блок свертки
    x = layers.Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Третий блок свертки
    x = layers.Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Четвертый блок свертки
    x = layers.Conv2D(filters=256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Пятый блок свертки
    x = layers.Conv2D(filters=256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Полносвязные слои
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="elu")(x)  # Добавлен дополнительный слой
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="elu")(x)  # Добавлен еще один слой
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Компиляция с оптимизатором Adam и learning rate
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    
    return model


class SaveBestModel(keras.callbacks.Callback):
    """Кастомный callback для сохранения лучшей модели."""
    
    def __init__(self, filepath: str, monitor: str = "val_loss"):
        super().__init__()
        # Используем .h5 формат для совместимости
        if filepath.endswith(".keras"):
            self.filepath = filepath.replace(".keras", ".h5")
        else:
            self.filepath = filepath
        self.monitor = monitor
        self.best_value = float("inf")
        self.best_weights = None
        
    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
            
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_weights = self.model.get_weights()
            # Сохраняем модель в формате .h5
            self.model.save(self.filepath)
            print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current_value:.5f}, saving model to {self.filepath}")
        else:
            print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best_value:.5f}")


def train_model(
    model: keras.Model,
    train_dataset: Any,
    validation_dataset: Any,
    model_name: str,
    epochs: int = 30,
    use_early_stopping: bool = True,
    use_lr_reduction: bool = True,
) -> keras.callbacks.History:
    """
    Обучение модели (листинг 8.11, 8.17).
    
    Args:
        model: Модель для обучения
        train_dataset: Обучающий набор данных
        validation_dataset: Валидационный набор данных
        model_name: Имя модели для сохранения
        epochs: Количество эпох
        use_early_stopping: Использовать ли early stopping
        use_lr_reduction: Использовать ли уменьшение learning rate
    
    Returns:
        История обучения
    """
    print(f"Обучение модели: {model_name}")
    
    # Сохраняем в текущей директории
    model_path = f"{model_name}.keras"
    
    callbacks = [
        SaveBestModel(filepath=model_path, monitor="val_loss")
    ]
    
    if use_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            )
        )
    
    if use_lr_reduction:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            )
        )
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1,
    )
    
    return history


def plot_training_history(history: keras.callbacks.History, model_name: str) -> None:
    """
    Построение графиков изменения потерь и точности (листинг 8.12).
    
    Args:
        history: История обучения
        model_name: Имя модели для названия файлов
    """
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    
    # Создаем директорию для графиков
    plots_dir = Path(__file__).parent / "plots" / "lab4"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # График точности
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, "bo", label="Точность на этапе обучения")
    plt.plot(epochs, val_accuracy, "b", label="Точность на этапе проверки")
    plt.title(f"Точность на этапах обучения и проверки ({model_name})")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{model_name}_accuracy.png", dpi=150, bbox_inches="tight")
    print(f"График точности сохранен: {plots_dir / f'{model_name}_accuracy.png'}")
    
    # График потерь
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, "bo", label="Потери на этапе обучения")
    plt.plot(epochs, val_loss, "b", label="Потери на этапе проверки")
    plt.title(f"Потери на этапах обучения и проверки ({model_name})")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{model_name}_loss.png", dpi=150, bbox_inches="tight")
    print(f"График потерь сохранен: {plots_dir / f'{model_name}_loss.png'}")
    
    plt.close("all")


def evaluate_model(model: keras.Model, test_dataset: Any) -> dict:
    """
    Оценка модели на тестовом наборе.
    
    Args:
        model: Модель для оценки
        test_dataset: Тестовый набор данных
    
    Returns:
        Словарь с метриками модели
    """
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    return {
        "accuracy": float(test_acc),
        "loss": float(test_loss),
    }


def visualize_data_augmentation(train_dataset: Any) -> None:
    """
    Визуализация обогащения данных (листинг 8.15).
    
    Args:
        train_dataset: Обучающий набор данных
    """
    print("Визуализация обогащения данных...")
    
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
    )
    
    plt.figure(figsize=(10, 10))
    for images, _ in train_dataset.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    
    plots_dir = Path(__file__).parent / "plots" / "lab4"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "data_augmentation.png", dpi=150, bbox_inches="tight")
    print(f"График обогащения данных сохранен: {plots_dir / 'data_augmentation.png'}")
    plt.close()


def train_basic_model(
    train_dataset: Any, validation_dataset: Any, test_dataset: Any
) -> None:
    """Обучение базовой модели."""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ БАЗОВОЙ МОДЕЛИ")
    print("=" * 60)
    
    model = create_basic_model()
    model.summary()
    
    history = train_model(
        model,
        train_dataset,
        validation_dataset,
        "convnet_from_scratch",
        epochs=30,
        use_early_stopping=False,
        use_lr_reduction=False,
    )
    
    plot_training_history(history, "базовая_модель")
    
    # Оценка на тестовом наборе
    # Пробуем загрузить .h5 (используется callback), если не существует - пробуем .keras
    try:
        test_model = keras.models.load_model("convnet_from_scratch.h5")
    except (OSError, IOError, FileNotFoundError):
        test_model = keras.models.load_model("convnet_from_scratch.keras")
    test_results = evaluate_model(test_model, test_dataset)
    print(f"\nTest accuracy: {test_results['accuracy']:.3f}")
    print(f"Test loss: {test_results['loss']:.3f}")
    
    # Сохранение модели с метаданными
    save_model(
        test_model,
        "cats_dogs_basic",
        metadata={
            "model_name": "Базовая модель Cats vs Dogs",
            "description": "Модель из главы 8.2 без модификаций",
            "test_metrics": test_results,
            "epochs": len(history.epoch),
        },
    )
    
    # Удаляем временные файлы после сохранения
    for temp_file in ["convnet_from_scratch.h5", "convnet_from_scratch.keras"]:
        temp_path = Path(temp_file)
        if temp_path.exists():
            temp_path.unlink()
            print(f"Временный файл удален: {temp_file}")


def train_improved_model(
    train_dataset: Any, validation_dataset: Any, test_dataset: Any
) -> None:
    """Обучение улучшенной модели."""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ")
    print("=" * 60)
    
    model = create_improved_model()
    model.summary()
    
    history = train_model(
        model,
        train_dataset,
        validation_dataset,
        "convnet_from_scratch_with_augmentation",
        epochs=100,
        use_early_stopping=True,
        use_lr_reduction=True,
    )
    
    plot_training_history(history, "улучшенная_модель")
    
    # Оценка на тестовом наборе
    # Пробуем загрузить .h5 (используется callback), если не существует - пробуем .keras
    try:
        test_model = keras.models.load_model("convnet_from_scratch_with_augmentation.h5")
    except (OSError, IOError, FileNotFoundError):
        test_model = keras.models.load_model("convnet_from_scratch_with_augmentation.keras")
    test_results = evaluate_model(test_model, test_dataset)
    print(f"\nTest accuracy: {test_results['accuracy']:.3f}")
    print(f"Test loss: {test_results['loss']:.3f}")
    
    # Сохранение модели с метаданными
    save_model(
        test_model,
        "cats_dogs_improved",
        metadata={
            "model_name": "Улучшенная модель Cats vs Dogs",
            "description": "Модель с batch normalization, ELU, расширенным обогащением данных, dropout и early stopping",
            "test_metrics": test_results,
            "epochs": len(history.epoch),
        },
    )
    
    # Удаляем временные файлы после сохранения
    for temp_file in ["convnet_from_scratch_with_augmentation.h5", "convnet_from_scratch_with_augmentation.keras"]:
        temp_path = Path(temp_file)
        if temp_path.exists():
            temp_path.unlink()
            print(f"Временный файл удален: {temp_file}")


def show_menu() -> str:
    """Отображение главного меню выбора."""
    print("\n" + "=" * 60)
    print("ГЛАВНОЕ МЕНЮ")
    print("=" * 60)
    print("1. Обучить новую модель")
    print("2. Загрузить существующую модель")
    print("3. Выход")
    print("=" * 60)

    while True:
        choice = input("\nВыберите действие (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Неверный выбор. Пожалуйста, введите 1, 2 или 3.")


def show_saved_models_menu() -> Optional[str]:
    """Отображение меню выбора сохраненной модели."""
    saved_models = list_saved_models()
    # Фильтруем только Cats vs Dogs модели
    cats_dogs_models = [m for m in saved_models if "cats_dogs" in m.lower()]

    if not cats_dogs_models:
        print("\nСохраненных моделей Cats vs Dogs не найдено.")
        return None

    print("\n" + "=" * 60)
    print("ДОСТУПНЫЕ МОДЕЛИ CATS VS DOGS")
    print("=" * 60)

    for idx, model_name in enumerate(cats_dogs_models, 1):
        metadata = load_model_metadata(model_name)
        if metadata:
            test_metrics = metadata.get("test_metrics", {})
            accuracy = test_metrics.get("accuracy", "N/A")
            if isinstance(accuracy, (int, float)):
                print(f"{idx}. {model_name} (Точность: {accuracy:.4f} ({accuracy*100:.2f}%))")
            else:
                print(f"{idx}. {model_name}")
        else:
            print(f"{idx}. {model_name}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"\nВыберите модель (1-{len(cats_dogs_models)}) или '0' для отмены: ").strip()
            if choice == "0":
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(cats_dogs_models):
                return cats_dogs_models[choice_num - 1]
            print(f"Неверный выбор. Пожалуйста, введите число от 1 до {len(cats_dogs_models)} или 0 для отмены.")
        except ValueError:
            print("Неверный ввод. Пожалуйста, введите число.")


def demonstrate_predictions(
    model: keras.Model,
    test_dataset: Any,
    num_samples: int = 10,
) -> None:
    """Демонстрация предсказаний модели на примерах изображений."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ")
    print("=" * 60)

    # Получаем батч изображений
    images_batch = None
    labels_batch = None
    for images, labels in test_dataset.take(1):
        images_batch = images
        labels_batch = labels
        break

    if images_batch is None:
        print("Не удалось загрузить изображения для демонстрации.")
        return

    # Делаем предсказания
    predictions = model.predict(images_batch, verbose=0)
    predicted_probs = predictions.flatten()

    # Классы: 0 = cat, 1 = dog
    class_names = ["Cat", "Dog"]

    print(f"\n{'Индекс':<8} {'Предсказание':<20} {'Вероятность':<15} {'Реальная метка':<20} {'Результат':<10}")
    print("-" * 80)

    num_to_show = min(num_samples, len(images_batch))
    for i in range(num_to_show):
        prob = predicted_probs[i]
        predicted_class = 1 if prob > 0.5 else 0
        true_class = int(labels_batch[i])
        confidence = prob if predicted_class == 1 else 1 - prob

        predicted_label = class_names[predicted_class]
        true_label = class_names[true_class]
        correct = "✓" if predicted_class == true_class else "✗"

        print(f"{i:<8} {predicted_label:<20} {confidence:<14.4f} {true_label:<20} {correct:<10}")

    # Статистика
    correct_predictions = sum(
        1 for i in range(num_to_show)
        if (1 if predicted_probs[i] > 0.5 else 0) == int(labels_batch[i])
    )
    accuracy = correct_predictions / num_to_show

    print("\n" + "=" * 60)
    print("СТАТИСТИКА ПРЕДСКАЗАНИЙ")
    print("=" * 60)
    print(f"Правильных предсказаний: {correct_predictions}/{num_to_show}")
    print(f"Точность на примерах: {accuracy:.4f} ({accuracy*100:.2f}%)")


def load_and_use_model(test_dataset: Any) -> None:
    """Загрузка и использование существующей модели."""
    model_name = show_saved_models_menu()

    if model_name is None:
        print("Отмена загрузки модели.")
        return

    try:
        print(f"\nЗагрузка модели: {model_name}")
        model = load_model(model_name)

        # Показываем метаданные модели
        metadata = load_model_metadata(model_name)
        if metadata:
            print("\n" + "=" * 60)
            print("ИНФОРМАЦИЯ О МОДЕЛИ")
            print("=" * 60)
            print(f"Название: {metadata.get('model_name', 'N/A')}")
            print(f"Описание: {metadata.get('description', 'N/A')}")
            test_metrics = metadata.get("test_metrics", {})
            if test_metrics:
                print("\nМетрики на тестовом наборе:")
                for metric_name, metric_value in test_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name == "accuracy":
                            print(f"  Точность: {metric_value:.4f} ({metric_value*100:.2f}%)")
                        elif metric_name == "loss":
                            print(f"  Потери: {metric_value:.4f}")
                        else:
                            print(f"  {metric_name}: {metric_value:.4f}")
            if "epochs" in metadata:
                print(f"Эпох обучения: {metadata['epochs']}")

        # Показываем архитектуру модели
        print("\n" + "=" * 60)
        print("АРХИТЕКТУРА МОДЕЛИ")
        print("=" * 60)
        model.summary()

        # Оцениваем модель на тестовых данных
        print("\n" + "=" * 60)
        print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
        print("=" * 60)
        test_results = evaluate_model(model, test_dataset)
        for metric_name, metric_value in test_results.items():
            if metric_name == "accuracy":
                print(f"Точность на тестовом наборе: {metric_value:.4f} ({metric_value*100:.2f}%)")
            elif metric_name == "loss":
                print(f"Потери на тестовом наборе: {metric_value:.4f}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")

        # Демонстрация предсказаний
        demonstrate_predictions(model, test_dataset)

    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
    except Exception as e:
        print(f"\nОшибка при загрузке модели: {e}")


def train_new_models() -> None:
    """Обучение новых моделей (базовая и улучшенная)."""
    print("Лабораторная работа №4: Обучение сверточной нейронной сети с нуля")
    print("=" * 60)

    # Подготовка данных
    try:
        prepare_data()
    except Exception as e:
        print(f"Ошибка при подготовке данных: {e}")
        print("Продолжаем с существующими данными...")

    # Загрузка наборов данных
    train_dataset, validation_dataset, test_dataset = load_datasets()

    # Визуализация обогащения данных
    visualize_data_augmentation(train_dataset)

    # Обучение базовой модели
    train_basic_model(train_dataset, validation_dataset, test_dataset)

    # Обучение улучшенной модели
    train_improved_model(train_dataset, validation_dataset, test_dataset)

    # Показываем список сохраненных моделей
    print("\n" + "=" * 60)
    print("СПИСОК СОХРАНЕННЫХ МОДЕЛЕЙ")
    print("=" * 60)
    saved_models = list_saved_models()
    cats_dogs_models = [m for m in saved_models if "cats_dogs" in m.lower()]
    for model_name in cats_dogs_models:
        print(f"  - {model_name}")
        metadata = load_model_metadata(model_name)
        if metadata:
            test_metrics = metadata.get("test_metrics", {})
            accuracy = test_metrics.get("accuracy", "N/A")
            if isinstance(accuracy, (int, float)):
                print(f"    Test accuracy: {accuracy:.3f} ({accuracy*100:.2f}%)")


def main() -> None:
    """Основная функция для выполнения лабораторной работы."""
    print("Лабораторная работа №4: Обучение сверточной нейронной сети с нуля")
    print("=" * 60)

    # Загружаем тестовый набор данных для использования в load_and_use_model
    # (если данные еще не подготовлены, это может вызвать ошибку, но это нормально)
    test_dataset = None
    try:
        _, _, test_dataset = load_datasets()
    except Exception:
        pass  # Данные будут загружены при обучении

    while True:
        choice = show_menu()

        if choice == "1":
            # Обучить новую модель
            train_new_models()

        elif choice == "2":
            # Загрузить существующую модель
            if test_dataset is None:
                try:
                    _, _, test_dataset = load_datasets()
                except Exception as e:
                    print(f"Ошибка при загрузке данных: {e}")
                    print("Пожалуйста, сначала обучите модель (опция 1).")
                    continue
            load_and_use_model(test_dataset)

        elif choice == "3":
            # Выход
            print("\nДо свидания!")
            break

        # Спрашиваем, хочет ли пользователь продолжить
        if choice in ["1", "2"]:
            continue_choice = input("\nПродолжить работу? (y/n): ").strip().lower()
            if continue_choice not in ["y", "yes", "да", "д"]:
                print("\nДо свидания!")
                break


if __name__ == "__main__":
    main()
