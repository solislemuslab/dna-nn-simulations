import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils import data
from transformers import AutoModelForSequenceClassification


def dna_bert6():
    model_name = "zhihan1996/DNA_bert_6"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    return model


def cnn_nguyen_2_conv2d(x_shape, classes=2):
    class cnn_nguyen_2_conv2d(nn.module):
        def __init__(self, x_shape, classes=2):
            super(cnn_nguyen_2_conv2d, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(x_shape[0], 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(32 * (x_shape[1] // 4) * (x_shape[2] // 4), 32),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            if classes < 3:
                self.net.add_module('fc', nn.Linear(32, 1))
                self.activation = nn.Sigmoid()
            else:
                self.net.add_module('fc', nn.Linear(32, classes))
                self.activation = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.net(x)
            x = self.activation(x)
            return x

    # Creating an instance of the model
    model = cnn_nguyen_2_conv2d(x_shape, classes)

    return model