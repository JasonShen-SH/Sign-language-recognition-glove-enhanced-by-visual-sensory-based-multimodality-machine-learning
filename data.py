from torchvision import transforms # first, we need to preprocess(i.e.transform) the train/val sets
# then we need to load all the train / val sets
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
import torch
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader

rootDir = "/root/visual_sensory_moving2"
# ROOT_TRAIN = "../visual_sensory_moving2"
# ROOT_TEST = "../visual_sensory_moving2" # test is val in our case
transforms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224,224)), # 裁剪为224*224
    ])
labelName = ["bowl", "dog", "feel", "get", "I", "know", "must", "sick", "you", "zero"]
labelBias = [128, 51, 72, 91, 114, 152, 129, 110, 145, 132]
labelCount = [2666, 2736, 2702, 2681, 2605, 2535, 2637, 2647, 2621, 2617]
label = 0
rawTensorDataSets = []
trainingRatio = 0.8
batchSize = 64
device = torch.device("cuda")

for i in range(len(labelName)):
    name = labelName[i]
    bias = labelBias[i]
    count = labelCount[i]
    tensorImgSets = []
    for j in range(count):
        img = Image.open(rootDir+"/%s/%s%04d.png" %(name, name, bias+j))
        tensorImg = transforms(img)
        tensorImg = tensorImg.reshape((1,-1)) # must be 1 * (3 * 224 * 224)
        if j == 0:
            tensorImgSets = tensorImg
        else:
            tensorImgSets = torch.concat((tensorImgSets, tensorImg), dim=0)
    print(tensorImgSets.shape)
    imgLabels = torch.zeros((tensorImgSets.shape[0], 1), dtype=torch.uint8)
    imgLabels += label
    label += 1
    tensorImgSets = torch.concat((tensorImgSets, imgLabels), dim=-1)
    if i == 0:
        rawTensorDataSets = tensorImgSets
    else:
        rawTensorDataSets = torch.concat((rawTensorDataSets, tensorImgSets), dim=0) # must be M * (3 * 224 * 224 + 1)
# torch.save(rawTensorDataSets, "/root/rawDatasets.pt")
rawTensorDataSets = rawTensorDataSets[torch.randperm(rawTensorDataSets.size()[0])]
torch.save(rawTensorDataSets, "/root/rawDatasets.pt")
totalCount = rawTensorDataSets.shape[0] # must be 21447
trainingCount = int(totalCount * trainingRatio)

trainingData = rawTensorDataSets[0:trainingCount, 0:-1].to(dtype=torch.float32).to(device)
trainingLabel = rawTensorDataSets[0:trainingCount, -1].to(dtype=torch.long).to(device)
validatingData = rawTensorDataSets[trainingCount:, 0:-1].to(dtype=torch.float32).to(device)
validatingLabel = rawTensorDataSets[trainingCount:, -1].to(dtype=torch.long).to(device)

trainingDataSet = TensorDataset(trainingData, trainingLabel)
trainingDateLoader = DataLoader(trainingDataSet, batch_size=batchSize, shuffle=True)

# now begin to preprocess(transform)
exit(0)