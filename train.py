import data_setup
import model_builder
import engine
import utils
import torch
from torchvision import transforms
import test_and_visualize
import os
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE =16
EPOCHS = 10
LEARNING_RATE = 0.001
train_dir = '/PetImages/train'
test_dir = '/PetImages/test'

data_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5])
])

train_dataloader,test_dataloader,class_names = data_setup.create_dataloader(train_dir=train_dir,
                                               test_dir=test_dir,
                                               transform=data_transforms,
                                               batch_size=BATCH_SIZE
                                               )
model = model_builder.Simple_CNN_Model(input_shape=3,output_shape=2).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

results = engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=EPOCHS,
             device=device)



utils.save_model(model=model,
                 base_dir= 'models',
                 model_name='Cats_vs_dogs_weights.pth'

                 )

test_and_visualize.plot_model_curves(results=results)