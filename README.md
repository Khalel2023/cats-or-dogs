# Cats vs Dogs Classifier

Пайплайн для классификации изображений кошек и собак на PyTorch

## Особенности
- Датасет взят с Kaggle: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset 
- Подготовка данных (автоматическое разделение на train/test) скрипт split_data.py
Структура данных после обработки


PetImages/
├── train/ # Обучающая выборка
│ ├── dog/ # Изображения собак (dog.1.jpg, dog.2.jpg)
│ └── cat/ # Изображения кошек (cat.1.jpg, cat.2.jpg)
└── test/ # Тестовая выборка
├── dog/ # Тестовые изображения собак
└── cat/ # Тестовые изображения кошек



- Так же модель CNN с BatchNorm и Dropout 
- Визуализация метрик
## Установка
```bash
git clone https://github.com/Khalel2023/cats-or-dogs.git
pip install -r requirements.txt

## Запуск обучения и сохранение модели
python train.py

## Вывод метрик и тестирование на новых изображениях 
python test_and_visualise.py

