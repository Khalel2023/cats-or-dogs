from typing import List
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from pathlib import Path
import model_builder
import random

class_names = ['Cat', 'Dog']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_model_curves(results):
    plt.figure(figsize=(10,10))
    train_acc = results['train_acc']
    train_loss = results['train_loss']
    test_acc = results['test_acc']
    test_loss = results['test_loss']
    epochs = range(len(train_acc))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train loss')
    plt.plot(epochs, test_loss, label='Test loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs,train_acc, label ='Train accuracy')
    plt.plot(epochs,test_acc, label='Test accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()

def pred_and_plot_image(model:torch.nn.Module,
                        image_path:str,
                        class_names:List[str],
                        image_size=(64,64),
                        device:torch.device=device,
                        transform:torchvision.transforms=None):

    img = Image.open(image_path).convert('RGB')

    if transform is not None:
        image_transform = transform

    else:
        image_transform = transforms.Compose([
            transforms.Resize(size=(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    model.eval()
    with torch.inference_mode():
        transformed_img = image_transform(img).unsqueeze(dim=0)
        img_pred = model(transformed_img.to(device))
        targ_img_probs = torch.softmax(img_pred,dim=1)
        targ_img_label = torch.argmax(targ_img_probs,dim=1)
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.title(f'Class : {class_names[targ_img_label]} | Probs : {targ_img_probs.max()}')
        plt.axis(False)
        plt.show()


# initialize model weights and upload test image from Google.

model = model_builder.Simple_CNN_Model(input_shape=3, output_shape=2).to(device)
model.load_state_dict(torch.load('models/Cats_vs_dogs_weights.pth',map_location=device))

image_dir = Path('Tested_images')
image_dir.mkdir(parents=True,exist_ok=True)
base_dir = '/PetImages/test'

# Take from test part of dataset 5 random test images and test our model
random_images_item = 10
random_iamge_paths = random.sample(list(Path(base_dir).glob('*/*.jpg')), k=random_images_item)
for img_path in random_iamge_paths:
    pred_and_plot_image(model=model,
                    image_path=img_path,
                    class_names=class_names,
                    device=device)


