import os
from pathlib import Path
import random
import shutil
# Скрипт для разделения набора изображений на обучающую и тестовую выборки
random.seed(42)
base_dir = Path('/PetImages')
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'
train_dir.mkdir(exist_ok=True,parents=True)
test_dir.mkdir(exist_ok=True,parents=True)
split_ratio = 0.7

for class_dir in base_dir.iterdir():
    if not class_dir.is_dir or class_dir.name in ['train' , 'test']:
        continue

    class_name = class_dir.name
    images = list(Path(class_dir).glob('*jpg'))
    random.shuffle(images)
    split_index = int(len(images)* split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]
    (train_dir / class_name).mkdir(parents=True,exist_ok=True)
    (test_dir / class_name).mkdir(parents=True, exist_ok=True)

    for img in train_images:
        shutil.move(img, train_dir / class_name / img.name)

    for img in test_images:
        shutil.move(img, test_dir / class_name / img.name)
