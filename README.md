# Cats vs Dogs Classifier

**Pipeline for classifying images of cats and dogs using PyTorch.**

---

## Features

- Dataset from Kaggle: [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
(you can see the examples bellow)
![Dog image example](screenshots/51.jpg)
![Cat image example](screenshots/2.jpg)  
- Data preparation (automatic train/test split) using `split_data.py`  
- CNN model with Batch Normalization and Dropout  
- Visualization of metrics (accuracy, loss, etc.)  

---

## Installation

```bash
git clone https://github.com/Khalel2023/cats-or-dogs.git
pip install -r requirements.txt

Usage
Training the model
python train.py

Testing and visualizing predictions
python test_and_visualise.py


