"""
Лабораторная работа №1. 2.1. ПЕРВОЕ ЗНАКОМСТВО С НЕЙРОННОЙ СЕТЬЮ.

Реализация кода из раздела 2.1 книги Франсуа Шолле "Глубокое обучение на Python"
с улучшениями для повышения точности классификации MNIST.
"""

from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import sys
from pathlib import Path

# Добавляем путь к utils для импорта функций работы с моделями
sys.path.append(str(Path(__file__).parent.parent))
from utils import save_model, load_model, list_saved_models, load_model_metadata


def load_and_prepare_data():
    """Загрузка и предобработка данных MNIST."""
    # Листинг 2.1. Загрузка набора данных MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    print("Форма обучающих данных:", train_images.shape)
    print("Количество меток:", len(train_labels))
    print("Форма тестовых данных:", test_images.shape)
    print("Количество тестовых меток:", len(test_labels))
    
    # Листинг 2.4. Подготовка исходных данных
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    
    return (train_images, train_labels), (test_images, test_labels)


def create_basic_model():
    """Создание базовой модели из книги (Листинг 2.2)."""
    model = keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(28 * 28,)),
        layers.Dense(10, activation="softmax")
    ])
    
    # Листинг 2.3. Этап компиляции
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model


def create_improved_model():
    """Создание улучшенной модели с различными оптимизациями."""
    model = keras.Sequential([
        # Первый скрытый слой с большим количеством нейронов
        layers.Dense(512, activation="relu", input_shape=(28 * 28,)),
        # Добавляем dropout для предотвращения переобучения
        layers.Dropout(0.3),
        # Второй скрытый слой
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        # Третий скрытый слой
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        # Выходной слой
        layers.Dense(10, activation="softmax")
    ])
    
    # Используем Adam оптимизатор (обычно работает лучше чем RMSprop)
    # и добавляем learning rate decay
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model


def train_and_evaluate_model(model, train_images, train_labels, 
                             test_images, test_labels, model_name="model", epochs=5):
    """Обучение и оценка модели."""
    print(f"\n{'='*60}")
    print(f"Обучение модели: {model_name}")
    print(f"{'='*60}")
    
    # Листинг 2.5. Обучение модели
    history = model.fit(train_images, train_labels, 
                       epochs=epochs, 
                       batch_size=128,
                       validation_split=0.1,  # Используем 10% данных для валидации
                       verbose=1)
    
    # Листинг 2.7. Оценка качества модели на новых данных
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nТочность на тестовом наборе: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Потери на тестовом наборе: {test_loss:.4f}")
    
    return history, test_acc, test_loss


def demonstrate_predictions(model, test_images, test_labels):
    """Демонстрация предсказаний модели (Листинг 2.6)."""
    print("\n" + "="*60)
    print("Демонстрация предсказаний")
    print("="*60)
    
    test_digits = test_images[0:10]
    predictions = model.predict(test_digits, verbose=0)
    
    for i in range(10):
        predicted_class = predictions[i].argmax()
        confidence = predictions[i][predicted_class]
        actual_class = test_labels[i]
        
        print(f"Изображение {i}: Предсказано = {predicted_class} "
              f"(уверенность: {confidence:.4f}), "
              f"Реальное = {actual_class} "
              f"{'✓' if predicted_class == actual_class else '✗'}")


def show_menu():
    """Отображение главного меню выбора."""
    print("\n" + "="*60)
    print("ГЛАВНОЕ МЕНЮ")
    print("="*60)
    print("1. Обучить новую модель")
    print("2. Загрузить существующую модель")
    print("3. Выход")
    print("="*60)
    
    while True:
        choice = input("\nВыберите действие (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Неверный выбор. Пожалуйста, введите 1, 2 или 3.")


def show_saved_models_menu():
    """Отображение меню выбора сохраненной модели."""
    saved_models = list_saved_models()
    
    if not saved_models:
        print("\nСохраненных моделей не найдено.")
        return None
    
    print("\n" + "="*60)
    print("ДОСТУПНЫЕ МОДЕЛИ")
    print("="*60)
    
    for idx, model_name in enumerate(saved_models, 1):
        metadata = load_model_metadata(model_name)
        if metadata:
            accuracy = metadata.get('test_accuracy', 'N/A')
            if isinstance(accuracy, float):
                print(f"{idx}. {model_name} (Точность: {accuracy:.4f} ({accuracy*100:.2f}%))")
            else:
                print(f"{idx}. {model_name}")
        else:
            print(f"{idx}. {model_name}")
    
    print("="*60)
    
    while True:
        try:
            choice = input(f"\nВыберите модель (1-{len(saved_models)}) или '0' для отмены: ").strip()
            if choice == '0':
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(saved_models):
                return saved_models[choice_num - 1]
            print(f"Неверный выбор. Пожалуйста, введите число от 1 до {len(saved_models)} или 0 для отмены.")
        except ValueError:
            print("Неверный ввод. Пожалуйста, введите число.")


def load_and_use_model(test_images, test_labels):
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
            print("\n" + "="*60)
            print("ИНФОРМАЦИЯ О МОДЕЛИ")
            print("="*60)
            print(f"Название: {metadata.get('model_name', 'N/A')}")
            print(f"Описание: {metadata.get('description', 'N/A')}")
            if 'test_accuracy' in metadata:
                acc = metadata['test_accuracy']
                print(f"Точность: {acc:.4f} ({acc*100:.2f}%)")
            if 'test_loss' in metadata:
                print(f"Потери: {metadata['test_loss']:.4f}")
            if 'epochs' in metadata:
                print(f"Эпох обучения: {metadata['epochs']}")
            if 'architecture' in metadata:
                print(f"Архитектура: {metadata['architecture']}")
            if 'optimizer' in metadata:
                print(f"Оптимизатор: {metadata['optimizer']}")
        
        # Показываем архитектуру модели
        print("\n" + "="*60)
        print("АРХИТЕКТУРА МОДЕЛИ")
        print("="*60)
        model.summary()
        
        # Оцениваем модель на тестовых данных
        print("\n" + "="*60)
        print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
        print("="*60)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
        print(f"\nТочность на тестовом наборе: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Потери на тестовом наборе: {test_loss:.4f}")
        
        # Демонстрация предсказаний
        demonstrate_predictions(model, test_images, test_labels)
        
    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
    except Exception as e:
        print(f"\nОшибка при загрузке модели: {e}")


def train_new_models():
    """Обучение новых моделей (базовая и улучшенная)."""
    # Загрузка и подготовка данных
    (train_images, train_labels), (test_images, test_labels) = load_and_prepare_data()
    
    # 1. Базовая модель из книги
    print("\n" + "="*60)
    print("1. БАЗОВАЯ МОДЕЛЬ ИЗ КНИГИ")
    print("="*60)
    basic_model = create_basic_model()
    basic_model.summary()
    
    basic_history, basic_test_acc, basic_test_loss = train_and_evaluate_model(
        basic_model, train_images, train_labels, 
        test_images, test_labels, 
        model_name="Базовая модель", 
        epochs=5
    )
    
    demonstrate_predictions(basic_model, test_images, test_labels)
    
    # 2. Улучшенная модель
    print("\n" + "="*60)
    print("2. УЛУЧШЕННАЯ МОДЕЛЬ")
    print("="*60)
    improved_model = create_improved_model()
    improved_model.summary()
    
    improved_history, improved_test_acc, improved_test_loss = train_and_evaluate_model(
        improved_model, train_images, train_labels, 
        test_images, test_labels, 
        model_name="Улучшенная модель", 
        epochs=10  # Больше эпох для лучшего обучения
    )
    
    demonstrate_predictions(improved_model, test_images, test_labels)
    
    # Сравнение результатов
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    print(f"Базовая модель:     Точность = {basic_test_acc:.4f} ({basic_test_acc*100:.2f}%)")
    print(f"Улучшенная модель:  Точность = {improved_test_acc:.4f} ({improved_test_acc*100:.2f}%)")
    improvement = improved_test_acc - basic_test_acc
    print(f"Улучшение:          {improvement:.4f} ({improvement*100:.2f} процентных пункта)")
    
    # Сохранение обеих моделей с метаданными
    print("\n" + "="*60)
    print("СОХРАНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    # Сохраняем базовую модель
    save_model(
        basic_model,
        "mnist_basic_model",
        metadata={
            "model_name": "Базовая модель MNIST",
            "description": "Базовая модель из книги Франсуа Шолле (раздел 2.1)",
            "test_accuracy": float(basic_test_acc),
            "test_loss": float(basic_test_loss),
            "epochs": 5,
            "architecture": "Dense(512, relu) -> Dense(10, softmax)",
            "optimizer": "rmsprop"
        }
    )
    
    # Сохраняем улучшенную модель
    save_model(
        improved_model,
        "mnist_improved_model",
        metadata={
            "model_name": "Улучшенная модель MNIST",
            "description": "Улучшенная модель с дополнительными слоями и dropout",
            "test_accuracy": float(improved_test_acc),
            "test_loss": float(improved_test_loss),
            "epochs": 10,
            "architecture": "Dense(512, relu) -> Dropout(0.3) -> Dense(256, relu) -> Dropout(0.3) -> Dense(128, relu) -> Dropout(0.2) -> Dense(10, softmax)",
            "optimizer": "adam"
        }
    )
    
    # Сохраняем лучшую модель отдельно
    if improved_test_acc > basic_test_acc:
        save_model(
            improved_model,
            "mnist_best_model",
            metadata={
                "model_name": "Лучшая модель MNIST",
                "description": "Модель с наилучшей точностью",
                "test_accuracy": float(improved_test_acc),
                "test_loss": float(improved_test_loss),
                "epochs": 10
            }
        )
    
    # Показываем список сохраненных моделей
    print("\n" + "="*60)
    print("СПИСОК СОХРАНЕННЫХ МОДЕЛЕЙ")
    print("="*60)
    saved_models = list_saved_models()
    for model_name in saved_models:
        print(f"  - {model_name}")


def main():
    """Основная функция для выполнения лабораторной работы."""
    print("Лабораторная работа №1: Первое знакомство с нейронной сетью")
    print("="*60)
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # Обучить новую модель
            train_new_models()
            
        elif choice == '2':
            # Загрузить существующую модель
            # Загружаем только тестовые данные для оценки
            (_, _), (test_images, test_labels) = load_and_prepare_data()
            load_and_use_model(test_images, test_labels)
            
        elif choice == '3':
            # Выход
            print("\nДо свидания!")
            break
        
        # Спрашиваем, хочет ли пользователь продолжить
        if choice in ['1', '2']:
            continue_choice = input("\nПродолжить работу? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', 'да', 'д']:
                print("\nДо свидания!")
                break


if __name__ == "__main__":
    main()

