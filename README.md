# Cats vs Dogs Classifier

Пайплайн для классификации изображений кошек и собак на PyTorch

## Особенности
- Датасет взят с Kaggle: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset 
- Подготовка данных (автоматическое разделение на train/test) скрипт split_data.py
- Модель CNN с BatchNorm и Dropout 
- Визуализация метрик
## Установка
```bash
git clone https://github.com/Khalel2023/cats-or-dogs.git
pip install -r requirements.txt

## Запуск обучения и сохранение модели
python train.py

## Вывод метрик и тестирование на новых изображениях 
python test_and_visualise.py

