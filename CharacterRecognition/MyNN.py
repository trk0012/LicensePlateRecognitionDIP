import torch
import torch.nn as nn
import torch.nn.functional as fun

class myNN(nn.Module):    
    def __init__(self):
        super(myNN, self).__init__()
        self.forward1 = nn.Linear(784, 512)
        self.forward2 = nn.Linear(512, 256)
        self.forward3 = nn.Linear(256, 256)
        self.forward35 = nn.Linear(256, 128)
        self.forward4 = nn.Linear(128, 47)
        #sets a dropout rate for the individual nodes of 15%
        self.dropout = nn.Dropout(p=.15)

    def forward(self, element):
        element = element.view(-1, 784)
        element = self.dropout(fun.relu(self.forward1(element)))
        element = self.dropout(fun.relu(self.forward2(element)))
        element = self.dropout(fun.relu(self.forward3(element)))
        element = self.dropout(fun.relu(self.forward35(element)))
        element = self.forward4(element)
        return element