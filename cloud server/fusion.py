import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from torchvision import transforms

print("This is the fusion AI architecture for sign language recognition, welcome!")

device = torch.device("cpu")


# load and normalize visual image
img_original = Image.open("test.png")
transforms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224,224)), 
    ])
img_original = transforms(img_original)
img_original = img_original.to(dtype=torch.float32).to(device)

def real_normalize(tensor,mean,std):
    tensor = (tensor-mean)/std
    return tensor

Rmean = torch.load("/root/pt_files/Rmean.pt"); Rstd = torch.load("/root/pt_files/Rstd.pt")
Gmean = torch.load("/root/pt_files/Gmean.pt"); Gstd = torch.load("/root/pt_files/Gstd.pt")
Bmean = torch.load("/root/pt_files/Bmean.pt"); Bstd = torch.load("/root/pt_files/Bstd.pt")

for i in range(224):
    for j in range(224):
        img_original[0][i][j] = (img_original[0][i][j]-Rmean)/Rstd
        
for i in range(224):
    for j in range(224):
        img_original[1][i][j] = (img_original[1][i][j]-Gmean)/Gstd

for i in range(224):
    for j in range(224):
        img_original[2][i][j] = (img_original[2][i][j]-Bmean)/Bstd

img_data = torch.reshape(img_original,(1,-1))

print("image loaded and normalized!")


# load and normalize sensor data
sensor_data = np.load("/root/sensor_data.npy")

mean = np.load("/root/npy_files/mean.npy")
mean = mean.astype("float32")
std = np.load("/root/npy_files/std.npy")
sensor_data = (sensor_data-mean)/std
sensor_data =torch.from_numpy(sensor_data).to(dtype=torch.float32).to(device)

print("sensor data loaded and normalized!")


# concatenate

feature = torch.concat((sensor_data,img_data),dim=-1)

print("sensor data and image concatenated!")

# model

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, identity_downsample=None, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
            
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # if identity_downsample is not None as default, then:
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x = x + identity
        x = self.relu(x)
        
        return x


class model_mix(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(model_mix, self).__init__()
        
        # self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # fc
        self.hidden1=nn.Sequential(
                nn.Linear(in_features=21,out_features=256,bias=True), # 20+5 / 20+11
                nn.Dropout(p = 0.2),
                nn.ReLU())
        self.hidden2=nn.Sequential(
                nn.Linear(in_features=256,out_features=128,bias=True),
                nn.ReLU())
        self.hidden3=nn.Sequential(
                nn.Linear(in_features=128,out_features=100,bias=True),
                nn.ReLU())
        self.hidden4=nn.Sequential(
                nn.Linear(in_features=100,out_features=10,bias=True),
                nn.ReLU())

        
    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )  
    
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None #默认是none,即identity-free shortcut
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        #对于64-128.128-256.256-512的第一个block,有stride=2,且outchannel=2*inchannel；
        #其他的block,64-64的全部2个,64-128的第2个，128-256的第2个，256-512的第2个，都是outchannel=inchannel
            
        return nn.Sequential(
            ResBlock(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            ResBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        x1 = x[:,11:] ; x2=x[:,0:11]
        x1 = x1.reshape((-1, 3, 224, 224))
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.shape[0], -1)
        x1 = self.fc(x1)
        
        x = torch.concat((x1,x2),dim=-1)
        
        x=self.hidden1(x)
        x=self.hidden2(x)
        x=self.hidden3(x)
        output=self.hidden4(x)
        
        return output

model = torch.load("/root/moving_fusion_architecture.pth",map_location=torch.device('cpu'))

print("model loaded!")

total = 0
correct = 0
with torch.no_grad():
    feature = feature.to(device)
    output = model(feature)
    _, predicted = torch.max(output.data, 1)
    print("the predicted result is: ", predicted)
    np.save("/root/predicted.npy",predicted)
    
