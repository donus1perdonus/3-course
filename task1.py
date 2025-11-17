import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Отключаем лишние сообщения TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import numpy as np

# Листинг 2.1. Загрузка набора данных MNIST в Keras
print("=== Загрузка данных MNIST ===")
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Исследование обучающих данных
print("\n=== Обучающие данные ===")
print("Форма train_images:", train_images.shape)
print("Длина train_labels:", len(train_labels))
print("Первые 10 меток train_labels:", train_labels[:10])

# Исследование контрольных данных
print("\n=== Контрольные данные ===")
print("Форма test_images:", test_images.shape)
print("Длина test_labels:", len(test_labels))
print("Первые 10 меток test_labels:", test_labels[:10])

# Листинг 2.2. Архитектура сети
print("\n=== Создание модели ===")
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Листинг 2.3. Этап компиляции
print("\n=== Компиляция модели ===")
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Листинг 2.4. Подготовка исходных данных
print("\n=== Предварительная обработка данных ===")
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

print("Форма train_images после преобразования:", train_images.shape)
print("Форма test_images после преобразования:", test_images.shape)

# Листинг 2.5. Обучение («адаптация») модели
print("\n=== Обучение модели ===")
history = model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Листинг 2.6. Использование модели для получения предсказаний
print("\n=== Получение предсказаний ===")
test_digits = test_images[0:10]
predictions = model.predict(test_digits)

print("Предсказания для первого изображения:")
for i, prob in enumerate(predictions[0]):
    print(f"Класс {i}: {prob:.8f}")

print(f"\nНаивысшая вероятность у класса: {predictions[0].argmax()}")
print(f"Значение вероятности: {predictions[0][7]:.8f}")
print(f"Реальная метка: {test_labels[0]}")

# Листинг 2.7. Оценка качества модели на новых данных
print("\n=== Оценка модели на тестовых данных ===")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Потери на тестовых данных: {test_loss:.4f}")
print(f"Точность на тестовых данных: {test_acc:.4f}")

# Дополнительная информация
print(f"\n=== Сводка ===")
print(f"Точность на обучающих данных: {history.history['accuracy'][-1]:.4f}")
print(f"Точность на тестовых данных: {test_acc:.4f}")
print(f"Разница (переобучение): {history.history['accuracy'][-1] - test_acc:.4f}")