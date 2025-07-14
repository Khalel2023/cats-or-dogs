from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def create_dataloader(train_dir:str,test_dir:str, transform:transforms.Compose, batch_size:int):

    train_data = ImageFolder(root=train_dir, transform=transform)
    test_data = ImageFolder(root=test_dir, transform=transform)
    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data,
                                  shuffle=True,
                                  batch_size=batch_size)

    test_dataloader = DataLoader(dataset=test_data,
                                  shuffle=False,
                                  batch_size=batch_size)

    return train_dataloader,test_dataloader,class_names