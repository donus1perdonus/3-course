"""
Лабораторная работа №3. 4.3. Предсказание цен на дома: пример регрессии.

Реализация кода из раздела 4.3 книги Франсуа Шолле
«Глубокое обучение на Python» с дополнениями и улучшениями.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

# Делаем доступными утилиты сохранения моделей
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    save_model,
    load_model,
    load_model_metadata,
    list_saved_models,
)

K_FOLDS = 4
BATCH_SIZE = 16


def load_and_normalize_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Загрузка и нормализация данных Boston Housing (листинги 4.23 и 4.24).
    
    Примечание: Boston Housing dataset был удален из TensorFlow 2.6+.
    Если возникнет ошибка, можно использовать альтернативные источники данных.
    """
    try:
        # Листинг 4.23. Загрузка набора данных для Бостона
        (train_data, train_targets), (test_data, test_targets) = keras.datasets.boston_housing.load_data()
    except AttributeError:
        # Альтернативный способ загрузки (если доступен)
        try:
            from tensorflow.keras.datasets import boston_housing # type: ignore
            (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
        except Exception as e:
            raise RuntimeError(
                "Boston Housing dataset недоступен в этой версии TensorFlow. "
                "Пожалуйста, обновите TensorFlow или используйте альтернативный источник данных."
            ) from e

    print(f"Размер обучающей выборки: {train_data.shape}")
    print(f"Размер тестовой выборки: {test_data.shape}")
    print(f"Количество признаков: {train_data.shape[1]}")

    # Листинг 4.24. Нормализация данных
    mean = train_data.mean(axis=0)
    train_data = train_data - mean
    std = train_data.std(axis=0)
    train_data = train_data / std

    # Применяем те же параметры нормализации к тестовым данным
    test_data = test_data - mean
    test_data = test_data / std

    return train_data, train_targets, test_data, test_targets, mean, std


def build_baseline_model(input_shape: Tuple[int, ...]) -> keras.Model:
    """Базовая модель из книги (листинг 4.25)."""
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


def build_improved_model(input_shape: Tuple[int, ...]) -> keras.Model:
    """Улучшенная модель с дополнительными слоями, dropout и регуляризацией."""
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def k_fold_cross_validation(
    train_data: np.ndarray,
    train_targets: np.ndarray,
    k: int,
    build_model_func,
    num_epochs: int,
    batch_size: int = BATCH_SIZE,
    verbose: int = 0,
) -> Tuple[List[float], List[List[float]]]:
    """
    K-fold кросс-валидация (листинг 4.26).
    
    Returns:
        Tuple of (all_scores, all_mae_histories)
    """
    num_val_samples = len(train_data) // k
    all_scores = []
    all_mae_histories = []

    for i in range(k):
        print(f"Processing fold #{i + 1}/{k}")
        val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples :]], axis=0
        )
        partial_train_targets = np.concatenate(
            [train_targets[: i * num_val_samples], train_targets[(i + 1) * num_val_samples :]], axis=0
        )

        model = build_model_func((train_data.shape[1],))
        history = model.fit(
            partial_train_data,
            partial_train_targets,
            validation_data=(val_data, val_targets),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        mae_history = history.history["val_mae"]
        all_mae_histories.append(mae_history)

    return all_scores, all_mae_histories


def plot_mae_history(all_mae_histories: List[List[float]], title_prefix: str, output_dir: Path, truncate: int = 0) -> None:
    """Построение графиков MAE (листинги 4.28, 4.29, 4.30)."""
    # Листинг 4.28. Создание истории последовательных средних оценок проверки
    num_epochs = len(all_mae_histories[0])
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    output_dir.mkdir(parents=True, exist_ok=True)

    # График полной истории
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel("Эпохи")
    plt.ylabel("Оценка MAE")
    plt.title(f"{title_prefix}: Средняя MAE по K-fold кросс-валидации")
    plt.grid(True)
    full_path = output_dir / f"{title_prefix.lower().replace(' ', '_')}_mae_full.png"
    plt.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close()

    # График без первых 10 точек (листинг 4.30)
    if truncate > 0:
        truncated_mae_history = average_mae_history[truncate:]
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
        plt.xlabel("Эпохи")
        plt.ylabel("Оценка MAE")
        plt.title(f"{title_prefix}: Средняя MAE (без первых {truncate} эпох)")
        plt.grid(True)
        truncated_path = output_dir / f"{title_prefix.lower().replace(' ', '_')}_mae_truncated.png"
        plt.savefig(truncated_path, dpi=150, bbox_inches="tight")
        plt.close()


def evaluate_model(model: keras.Model, test_data: np.ndarray, test_targets: np.ndarray) -> Dict[str, float]:
    """Оценка модели на тестовом наборе (листинг 4.31)."""
    results = model.evaluate(test_data, test_targets, verbose=0)
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
    # Фильтруем только Boston Housing модели
    boston_models = [m for m in saved_models if "boston" in m.lower()]

    if not boston_models:
        print("\nСохраненных моделей Boston Housing не найдено.")
        return None

    print("\n" + "=" * 60)
    print("ДОСТУПНЫЕ МОДЕЛИ BOSTON HOUSING")
    print("=" * 60)

    for idx, model_name in enumerate(boston_models, 1):
        metadata = load_model_metadata(model_name)
        if metadata:
            test_metrics = metadata.get("test_metrics", {})
            mae = test_metrics.get("mae", test_metrics.get("loss", "N/A"))
            if isinstance(mae, (int, float)):
                print(f"{idx}. {model_name} (MAE: {mae:.4f})")
            else:
                print(f"{idx}. {model_name}")
        else:
            print(f"{idx}. {model_name}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"\nВыберите модель (1-{len(boston_models)}) или '0' для отмены: ").strip()
            if choice == "0":
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(boston_models):
                return boston_models[choice_num - 1]
            print(f"Неверный выбор. Пожалуйста, введите число от 1 до {len(boston_models)} или 0 для отмены.")
        except ValueError:
            print("Неверный ввод. Пожалуйста, введите число.")


def demonstrate_predictions(
    model: keras.Model,
    test_data: np.ndarray,
    test_targets: np.ndarray,
    num_samples: int = 10,
) -> None:
    """Демонстрация предсказаний модели на примерах."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ")
    print("=" * 60)

    # Выбираем случайные примеры
    indices = np.random.choice(len(test_data), size=min(num_samples, len(test_data)), replace=False)
    test_samples = test_data[indices]
    true_prices = test_targets[indices]

    predictions = model.predict(test_samples, verbose=0)
    predicted_prices = predictions.flatten()

    print(f"\n{'Индекс':<8} {'Предсказание':<15} {'Реальная цена':<15} {'Ошибка':<12} {'Ошибка %':<12}")
    print("-" * 70)

    for i, idx in enumerate(indices):
        pred = predicted_prices[i]
        true = true_prices[i]
        error = abs(pred - true)
        error_pct = (error / true * 100) if true != 0 else 0

        print(f"{idx:<8} ${pred:<14.2f} ${true:<14.2f} ${error:<11.2f} {error_pct:<11.2f}%")

    # Статистика
    errors = np.abs(predicted_prices - true_prices)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)

    print("\n" + "=" * 60)
    print("СТАТИСТИКА ОШИБОК")
    print("=" * 60)
    print(f"Средняя абсолютная ошибка: ${mean_error:.2f}")
    print(f"Медианная абсолютная ошибка: ${median_error:.2f}")
    print(f"Максимальная абсолютная ошибка: ${max_error:.2f}")


def load_and_use_model(test_data: np.ndarray, test_targets: np.ndarray) -> None:
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
                        if metric_name == "mae":
                            print(f"  MAE (средняя абсолютная ошибка): {metric_value:.4f}")
                        elif metric_name == "loss" or metric_name == "mse":
                            print(f"  MSE (средняя квадратичная ошибка): {metric_value:.4f}")
                        else:
                            print(f"  {metric_name}: {metric_value:.4f}")
            if "epochs" in metadata:
                print(f"Эпох обучения: {metadata['epochs']}")
            if "k_folds" in metadata:
                print(f"K-fold кросс-валидация: {metadata['k_folds']}")

        # Показываем архитектуру модели
        print("\n" + "=" * 60)
        print("АРХИТЕКТУРА МОДЕЛИ")
        print("=" * 60)
        model.summary()

        # Оцениваем модель на тестовых данных
        print("\n" + "=" * 60)
        print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
        print("=" * 60)
        test_results = evaluate_model(model, test_data, test_targets)
        for metric_name, metric_value in test_results.items():
            if metric_name == "mae":
                print(f"MAE на тестовом наборе: {metric_value:.4f}")
            elif metric_name == "loss" or metric_name == "mse":
                print(f"MSE на тестовом наборе: {metric_value:.4f}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")

        # Демонстрация предсказаний
        demonstrate_predictions(model, test_data, test_targets)

    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
    except Exception as e:
        print(f"\nОшибка при загрузке модели: {e}")


def train_new_models() -> None:
    """Обучение новых моделей (базовая и улучшенная)."""
    print("Лабораторная работа №3: Предсказание цен на дома (Boston Housing)")
    print("=" * 60)

    train_data, train_targets, test_data, test_targets, mean, std = load_and_normalize_data()

    # Базовая модель из книги
    print("\n" + "=" * 60)
    print("1. БАЗОВАЯ МОДЕЛЬ ИЗ КНИГИ")
    print("=" * 60)

    # K-fold кросс-валидация с 100 эпохами для быстрой оценки
    print("\nK-fold кросс-валидация (k=4, epochs=100)...")
    baseline_scores, baseline_mae_histories = k_fold_cross_validation(
        train_data,
        train_targets,
        K_FOLDS,
        build_baseline_model,
        num_epochs=100,
        verbose=0,
    )

    print(f"\nРезультаты кросс-валидации базовой модели:")
    print(f"Средняя MAE: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
    print(f"Минимальная MAE: {np.min(baseline_scores):.4f}")
    print(f"Максимальная MAE: {np.max(baseline_scores):.4f}")

    plots_dir = Path(__file__).parent / "plots" / "lab3"
    plot_mae_history(baseline_mae_histories, "Базовая модель", plots_dir, truncate=10)

    # Обучение финальной модели на всех данных (листинг 4.31)
    print("\nОбучение финальной базовой модели на всех данных (130 эпох)...")
    baseline_final_model = build_baseline_model((train_data.shape[1],))
    baseline_final_model.fit(train_data, train_targets, epochs=130, batch_size=BATCH_SIZE, verbose=0)

    baseline_results = evaluate_model(baseline_final_model, test_data, test_targets)
    print("\nРезультаты базовой модели на тестовом наборе:", baseline_results)

    # Улучшенная модель
    print("\n" + "=" * 60)
    print("2. УЛУЧШЕННАЯ МОДЕЛЬ")
    print("=" * 60)

    # K-fold кросс-валидация с 150 эпохами
    print("\nK-fold кросс-валидация (k=4, epochs=150)...")
    improved_scores, improved_mae_histories = k_fold_cross_validation(
        train_data,
        train_targets,
        K_FOLDS,
        build_improved_model,
        num_epochs=150,
        verbose=0,
    )

    print(f"\nРезультаты кросс-валидации улучшенной модели:")
    print(f"Средняя MAE: {np.mean(improved_scores):.4f} ± {np.std(improved_scores):.4f}")
    print(f"Минимальная MAE: {np.min(improved_scores):.4f}")
    print(f"Максимальная MAE: {np.max(improved_scores):.4f}")

    plot_mae_history(improved_mae_histories, "Улучшенная модель", plots_dir, truncate=10)

    # Обучение финальной улучшенной модели
    print("\nОбучение финальной улучшенной модели на всех данных (150 эпох)...")
    improved_final_model = build_improved_model((train_data.shape[1],))
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=15,
            min_delta=0.1,
            restore_best_weights=True,
            monitor="loss",
        )
    ]
    improved_final_model.fit(
        train_data, train_targets, epochs=200, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0
    )

    improved_results = evaluate_model(improved_final_model, test_data, test_targets)
    print("\nРезультаты улучшенной модели на тестовом наборе:", improved_results)

    # Сравнение результатов
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    baseline_mae = baseline_results.get("mae", 0)
    improved_mae = improved_results.get("mae", 0)
    if baseline_mae and improved_mae:
        delta = improved_mae - baseline_mae
        print(f"MAE: базовая = {baseline_mae:.4f}, улучшенная = {improved_mae:.4f}, Δ = {delta:+.4f}")
        if delta < 0:
            improvement_pct = abs(delta / baseline_mae * 100)
            print(f"Улучшение: {improvement_pct:.2f}%")

    # Сохранение моделей
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    save_model(
        baseline_final_model,
        "boston_basic_model",
        metadata={
            "model_name": "Базовая модель Boston Housing",
            "description": "Модель из главы 4.3 без модификаций",
            "test_metrics": baseline_results,
            "epochs": 130,
            "k_folds": K_FOLDS,
            "cv_mean_mae": float(np.mean(baseline_scores)),
            "cv_std_mae": float(np.std(baseline_scores)),
        },
    )

    save_model(
        improved_final_model,
        "boston_improved_model",
        metadata={
            "model_name": "Улучшенная модель Boston Housing",
            "description": "Дополнительные слои, Dropout, L2-регуляризация, Adam и EarlyStopping",
            "test_metrics": improved_results,
            "epochs": 200,
            "k_folds": K_FOLDS,
            "cv_mean_mae": float(np.mean(improved_scores)),
            "cv_std_mae": float(np.std(improved_scores)),
        },
    )

    # Показываем список сохраненных моделей
    print("\n" + "=" * 60)
    print("СПИСОК СОХРАНЕННЫХ МОДЕЛЕЙ")
    print("=" * 60)
    saved_models = list_saved_models()
    boston_models = [m for m in saved_models if "boston" in m.lower()]
    for model_name in boston_models:
        print(f"  - {model_name}")


def main() -> None:
    """Основная функция для выполнения лабораторной работы."""
    print("Лабораторная работа №3: Предсказание цен на дома (Boston Housing)")
    print("=" * 60)

    while True:
        choice = show_menu()

        if choice == "1":
            # Обучить новую модель
            train_new_models()

        elif choice == "2":
            # Загрузить существующую модель
            # Загружаем только тестовые данные для оценки
            _, _, test_data, test_targets, _, _ = load_and_normalize_data()
            load_and_use_model(test_data, test_targets)

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

