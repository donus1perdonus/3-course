"""
Лабораторная работа №2. 4.1. Классификация отзывов к фильмам (IMDB).

Реализация кода из раздела 4.1 книги Франсуа Шолле
«Глубокое обучение на Python» с дополнениями и улучшениями.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# Делаем доступными утилиты сохранения моделей
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    save_model,
    load_model,
    load_model_metadata,
    list_saved_models,
)

NUM_WORDS = 10_000
VAL_SET_SIZE = 10_000
BATCH_SIZE = 512


def decode_review(encoded_review: Iterable[int]) -> str:
    """Декодирование отзыва в текст (листинг 4.2)."""
    word_index = keras.datasets.imdb.get_word_index()
    reverse_word_index = {value + 3: key for key, value in word_index.items()}
    reverse_word_index[0] = "<PAD>"
    reverse_word_index[1] = "<START>"
    reverse_word_index[2] = "<UNK>"
    reverse_word_index[3] = "<UNUSED>"
    return " ".join(reverse_word_index.get(i, "?") for i in encoded_review)


def vectorize_sequences(sequences: Iterable[Iterable[int]], dimension: int = NUM_WORDS) -> np.ndarray:
    """Кодирование последовательностей в бинарную матрицу (листинг 4.3)."""
    results = np.zeros((len(sequences), dimension), dtype="float32")
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


def load_and_prepare_data(
    num_words: int = NUM_WORDS,
    dataset: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Загрузка и подготовка данных IMDB (листинги 4.1 и 4.3)."""
    if dataset is None:
        dataset = keras.datasets.imdb.load_data(num_words=num_words)
    (train_data, train_labels), (test_data, test_labels) = dataset

    x_train = vectorize_sequences(train_data, dimension=num_words)
    x_test = vectorize_sequences(test_data, dimension=num_words)
    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")

    return x_train, y_train, x_test, y_test


def build_baseline_model(input_shape: Tuple[int, ...]) -> keras.Model:
    """Базовая модель из книги (листинг 4.4)."""
    model = keras.Sequential(
        [
            layers.Dense(16, activation="relu", input_shape=input_shape),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_improved_model(input_shape: Tuple[int, ...]) -> keras.Model:
    """Улучшенная модель с дополнительными слоями, dropout и L2-регуляризацией."""
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def plot_history(history: keras.callbacks.History, title_prefix: str, output_dir: Path) -> None:
    """Построение графиков потерь и точности (листинги 4.7 и 4.9)."""
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_dict["loss"], "bo", label="Потери на обучении")
    plt.plot(epochs, history_dict["val_loss"], "b", label="Потери на проверке")
    plt.title(f"{title_prefix}: потери")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()
    loss_path = output_dir / f"{title_prefix.lower().replace(' ', '_')}_loss.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    if "accuracy" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history_dict["accuracy"], "bo", label="Точность на обучении")
        plt.plot(epochs, history_dict["val_accuracy"], "b", label="Точность на проверке")
        plt.title(f"{title_prefix}: точность")
        plt.xlabel("Эпохи")
        plt.ylabel("Точность")
        plt.legend()
        acc_path = output_dir / f"{title_prefix.lower().replace(' ', '_')}_accuracy.png"
        plt.savefig(acc_path, dpi=150, bbox_inches="tight")
        plt.close()


def evaluate_model(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Оценка модели на тестовом наборе (листинг 4.10)."""
    results = model.evaluate(x_test, y_test, verbose=0)
    return {name: float(value) for name, value in zip(model.metrics_names, results)}


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
    # Фильтруем только IMDB модели
    imdb_models = [m for m in saved_models if "imdb" in m.lower()]

    if not imdb_models:
        print("\nСохраненных моделей IMDB не найдено.")
        return None

    print("\n" + "=" * 60)
    print("ДОСТУПНЫЕ МОДЕЛИ IMDB")
    print("=" * 60)

    for idx, model_name in enumerate(imdb_models, 1):
        metadata = load_model_metadata(model_name)
        if metadata:
            test_metrics = metadata.get("test_metrics", {})
            accuracy = test_metrics.get("accuracy", test_metrics.get("loss", "N/A"))
            if isinstance(accuracy, (int, float)):
                print(f"{idx}. {model_name} (Точность: {accuracy:.4f} ({accuracy*100:.2f}%))")
            else:
                print(f"{idx}. {model_name}")
        else:
            print(f"{idx}. {model_name}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"\nВыберите модель (1-{len(imdb_models)}) или '0' для отмены: ").strip()
            if choice == "0":
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(imdb_models):
                return imdb_models[choice_num - 1]
            print(f"Неверный выбор. Пожалуйста, введите число от 1 до {len(imdb_models)} или 0 для отмены.")
        except ValueError:
            print("Неверный ввод. Пожалуйста, введите число.")


def demonstrate_predictions(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_sequences: List[List[int]],
    num_samples: int = 10,
) -> None:
    """Демонстрация предсказаний модели на примерах отзывов."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ")
    print("=" * 60)

    # Выбираем случайные примеры
    indices = np.random.choice(len(x_test), size=min(num_samples, len(x_test)), replace=False)
    test_samples = x_test[indices]
    true_labels = y_test[indices]
    sequences_samples = [test_sequences[i] for i in indices]

    predictions = model.predict(test_samples, verbose=0)
    predicted_probs = predictions.flatten()

    for i, idx in enumerate(indices):
        prob = predicted_probs[i]
        predicted_label = 1 if prob > 0.5 else 0
        true_label = int(true_labels[i])
        confidence = prob if predicted_label == 1 else 1 - prob

        # Декодируем отзыв (первые 100 слов)
        review_text = decode_review(sequences_samples[i][:100])
        review_preview = review_text[:200] + "..." if len(review_text) > 200 else review_text

        sentiment = "Положительный" if predicted_label == 1 else "Отрицательный"
        true_sentiment = "Положительный" if true_label == 1 else "Отрицательный"
        correct = "✓" if predicted_label == true_label else "✗"

        print(f"\n--- Пример {i+1} (индекс {idx}) ---")
        print(f"Отзыв: {review_preview}")
        print(f"Предсказание: {sentiment} (вероятность: {prob:.4f}, уверенность: {confidence:.4f})")
        print(f"Реальная метка: {true_sentiment} {correct}")


def load_and_use_model(x_test: np.ndarray, y_test: np.ndarray, test_sequences: List[List[int]]) -> None:
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
            if "num_words" in metadata:
                print(f"Размер словаря: {metadata['num_words']}")

        # Показываем архитектуру модели
        print("\n" + "=" * 60)
        print("АРХИТЕКТУРА МОДЕЛИ")
        print("=" * 60)
        model.summary()

        # Оцениваем модель на тестовых данных
        print("\n" + "=" * 60)
        print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
        print("=" * 60)
        test_results = evaluate_model(model, x_test, y_test)
        for metric_name, metric_value in test_results.items():
            if metric_name == "accuracy":
                print(f"Точность на тестовом наборе: {metric_value:.4f} ({metric_value*100:.2f}%)")
            elif metric_name == "loss":
                print(f"Потери на тестовом наборе: {metric_value:.4f}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")

        # Демонстрация предсказаний
        demonstrate_predictions(model, x_test, y_test, test_sequences)

    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
    except Exception as e:
        print(f"\nОшибка при загрузке модели: {e}")


def train_new_models() -> None:
    """Обучение новых моделей (базовая и улучшенная)."""
    print("Лабораторная работа №2: Классификация отзывов IMDB")
    print("=" * 60)

    raw_dataset = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    x_train, y_train, x_test, y_test = load_and_prepare_data(num_words=NUM_WORDS, dataset=raw_dataset)
    x_val = x_train[:VAL_SET_SIZE]
    partial_x_train = x_train[VAL_SET_SIZE:]
    y_val = y_train[:VAL_SET_SIZE]
    partial_y_train = y_train[VAL_SET_SIZE:]

    print(f"Размер обучающей выборки: {partial_x_train.shape}")
    print(f"Размер проверочной выборки: {x_val.shape}")
    print(f"Размер тестовой выборки: {x_test.shape}")

    train_sequences = raw_dataset[0][0]
    decoded_sample = decode_review(train_sequences[0][:200])
    print("\nПример декодированного отзыва:\n", decoded_sample[:500], "...")

    baseline_model = build_baseline_model((NUM_WORDS,))
    print("\n" + "=" * 60)
    print("Обучение базовой модели из книги")
    print("=" * 60)
    baseline_history = baseline_model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=2,
    )

    plots_dir = Path(__file__).parent / "plots" / "lab2"
    plot_history(baseline_history, "Базовая модель", plots_dir)

    baseline_results = evaluate_model(baseline_model, x_test, y_test)
    print("\nРезультаты базовой модели:", baseline_results)

    improved_model = build_improved_model((NUM_WORDS,))
    print("\n" + "=" * 60)
    print("Обучение улучшенной модели")
    print("=" * 60)
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=3,
            min_delta=0.001,
            restore_best_weights=True,
            monitor="val_loss",
        )
    ]
    improved_history = improved_model.fit(
        partial_x_train,
        partial_y_train,
        epochs=30,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    plot_history(improved_history, "Улучшенная модель", plots_dir)
    improved_results = evaluate_model(improved_model, x_test, y_test)
    print("\nРезультаты улучшенной модели:", improved_results)

    print("\n" + "=" * 60)
    print("Сравнение моделей")
    print("=" * 60)
    for metric in improved_model.metrics_names:
        base_value = baseline_results.get(metric)
        improved_value = improved_results.get(metric)
        if base_value is not None and improved_value is not None:
            delta = improved_value - base_value
            print(f"{metric}: базовая = {base_value:.4f}, улучшенная = {improved_value:.4f}, Δ = {delta:+.4f}")

    save_model(
        baseline_model,
        "imdb_basic_model",
        metadata={
            "model_name": "Базовая модель IMDB",
            "description": "Модель из главы 4.1 без модификаций",
            "test_metrics": baseline_results,
            "epochs": len(baseline_history.epoch),
            "num_words": NUM_WORDS,
        },
    )

    save_model(
        improved_model,
        "imdb_improved_model",
        metadata={
            "model_name": "Улучшенная модель IMDB",
            "description": "Дополнительные слои, Dropout, L2, Adam и EarlyStopping",
            "test_metrics": improved_results,
            "epochs": len(improved_history.epoch),
            "num_words": NUM_WORDS,
        },
    )

    # Показываем список сохраненных моделей
    print("\n" + "=" * 60)
    print("СПИСОК СОХРАНЕННЫХ МОДЕЛЕЙ")
    print("=" * 60)
    saved_models = list_saved_models()
    imdb_models = [m for m in saved_models if "imdb" in m.lower()]
    for model_name in imdb_models:
        print(f"  - {model_name}")


def main() -> None:
    """Основная функция для выполнения лабораторной работы."""
    print("Лабораторная работа №2: Классификация отзывов к фильмам (IMDB)")
    print("=" * 60)

    while True:
        choice = show_menu()

        if choice == "1":
            # Обучить новую модель
            train_new_models()

        elif choice == "2":
            # Загрузить существующую модель
            # Загружаем только тестовые данные для оценки
            raw_dataset = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
            _, _, x_test, y_test = load_and_prepare_data(num_words=NUM_WORDS, dataset=raw_dataset)
            test_sequences = raw_dataset[1][0]  # Тестовые последовательности
            load_and_use_model(x_test, y_test, test_sequences)

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


