"""
Утилиты для сохранения и загрузки обученных моделей Keras.
"""

import os
import json
from pathlib import Path
from tensorflow import keras
from typing import Optional, Dict, Any


def get_models_dir() -> Path:
    """Получить путь к директории с моделями."""
    # Получаем корневую директорию проекта (на уровень выше utils)
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    return models_dir


def save_model(model: keras.Model, 
               filename: str, 
               metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Сохранить модель Keras в файл.
    
    Args:
        model: Обученная модель Keras
        filename: Имя файла для сохранения (без расширения .h5)
        metadata: Дополнительные метаданные для сохранения (точность, потери и т.д.)
    
    Returns:
        Полный путь к сохраненному файлу модели
    """
    models_dir = get_models_dir()
    
    # Сохраняем модель
    model_path = models_dir / f"{filename}.h5"
    model.save(str(model_path))
    print(f"Модель сохранена: {model_path}")
    
    # Сохраняем метаданные, если они предоставлены
    if metadata:
        metadata_path = models_dir / f"{filename}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Метаданные сохранены: {metadata_path}")
    
    return str(model_path)


def load_model(filename: str) -> keras.Model:
    """
    Загрузить модель Keras из файла.
    
    Args:
        filename: Имя файла модели (с расширением .h5 или без него)
    
    Returns:
        Загруженная модель Keras
    
    Raises:
        FileNotFoundError: Если файл модели не найден
    """
    models_dir = get_models_dir()
    
    # Добавляем расширение, если его нет
    if not filename.endswith('.h5'):
        filename = f"{filename}.h5"
    
    model_path = models_dir / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    model = keras.models.load_model(str(model_path))
    print(f"Модель загружена: {model_path}")
    
    return model


def load_model_metadata(filename: str) -> Optional[Dict[str, Any]]:
    """
    Загрузить метаданные модели из JSON файла.
    
    Args:
        filename: Имя файла модели (без расширения)
    
    Returns:
        Словарь с метаданными или None, если файл не найден
    """
    models_dir = get_models_dir()
    metadata_path = models_dir / f"{filename}_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata


def list_saved_models() -> list:
    """
    Получить список всех сохраненных моделей.
    
    Returns:
        Список имен файлов моделей (без расширения .h5)
    """
    models_dir = get_models_dir()
    model_files = list(models_dir.glob("*.h5"))
    return [f.stem for f in model_files]


def model_exists(filename: str) -> bool:
    """
    Проверить, существует ли сохраненная модель.
    
    Args:
        filename: Имя файла модели (с расширением .h5 или без него)
    
    Returns:
        True, если модель существует, False в противном случае
    """
    models_dir = get_models_dir()
    
    if not filename.endswith('.h5'):
        filename = f"{filename}.h5"
    
    model_path = models_dir / filename
    return model_path.exists()

