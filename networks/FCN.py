import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(FCN,self).__init__()
        self.fc1 = nn.Linear(input_size,64)
        self.fc2 = nn.Linear(64,num_classes)
    def forward(self,x):
        tmp = self.fc1(x)
        out = self.fc2(tmp)
        return out