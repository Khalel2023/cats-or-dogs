# Cats vs Dogs Classifier

Пайплайн для классификации изображений кошек и собак на PyTorch

## Особенности
- Датасет взят с Kaggle: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset 
- Подготовка данных (автоматическое разделение на train/test) скрипт split_data.py
- Модель CNN с BatchNorm и Dropout 
- Визуализация метрик
PetImages/
├── train/ # Обучающая выборка
│ ├── dog/ # Изображения собак (например dog001.jpg)
│ └── cat/ # Изображения кошек (например cat001.jpg)
└── test/ # Тестовая выборка
├── dog/ # Изображения собак для тестирования
└── cat/ # Изображения кошек для тестирования

## Установка
```bash
git clone https://github.com/Khalel2023/cats-or-dogs.git
pip install -r requirements.txt

## Запуск обучения и сохранение модели
python train.py

## Вывод метрик и тестирование на новых изображениях 
python test_and_visualise.py

