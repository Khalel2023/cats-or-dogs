import torch
import torch.nn as nn
# creating a simple CNN  Model
class Simple_CNN_Model(torch.nn.Module):
    def __init__(self,input_shape,output_shape):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(kernel_size=2,stride=2)

        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(p=0.1),
                                        nn.Linear(in_features=16 * 16 * 16, out_features=output_shape)
                                        )

    def forward(self,x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))