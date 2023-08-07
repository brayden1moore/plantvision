import requests
from io import BytesIO
from PIL import Image, ImageOps
import torchvision.transforms as T
import torch
import gc
import pickle as pkl
from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()
import datetime as dt
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

visionTransformer = AutoModel.from_pretrained(r"google/vit-base-patch16-224-in21k")

class PlantVision(nn.Module):
    def __init__(self, num_classes):
        super(PlantVision, self).__init__()
        self.vit = visionTransformer
        count = 0
        for child in self.vit.children():
            count += 1
            if count < 4:
                for param in child.parameters():
                    param.requires_grad = False
        self.vitLayers = list(self.vit.children()) 
        self.vitTop = nn.Sequential(*self.vitLayers[:-2])
        self.vitNorm = list(self.vit.children())[2]
        self.vit = None
        gc.collect()
        self.vitFlatten = nn.Flatten()
        self.vitLinear = nn.Linear(151296,num_classes)
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, input):
        output = self.vitTop(input).last_hidden_state
        output = self.vitNorm(output)
        output = self.vitFlatten(output)
        output = F.relu(self.vitLinear(output))
        output = self.fc(output)
        return output

device = 'cpu' # ('cuda' if torch.cuda.is_available else 'cpu')

with open(fr'{THIS_FOLDER}/resources/flowerLabelSet.pkl', 'rb') as f:
        flowerLabelSet = pkl.load(f)

with open(fr'{THIS_FOLDER}/resources/leafLabelSet.pkl', 'rb') as f:
        leafLabelSet = pkl.load(f)

with open(fr'{THIS_FOLDER}/resources/fruitLabelSet.pkl', 'rb') as f:
        fruitLabelSet = pkl.load(f)

def loadModel(feature, labelSet):
    model = PlantVision(num_classes=len(labelSet))
    model.vitFlatten.load_state_dict(torch.load(BytesIO(requests.get(f"https://storage.googleapis.com/bmllc-plant-model-bucket/{feature}-vitFlatten-weights.pt").content), map_location=torch.device(device)), strict=False)
    model.vitLinear.load_state_dict(torch.load(BytesIO(requests.get(f"https://storage.googleapis.com/bmllc-plant-model-bucket/{feature}-vitLinear-weights.pt").content), map_location=torch.device(device)), strict=False)
    model.fc.load_state_dict(torch.load(BytesIO(requests.get(f"https://storage.googleapis.com/bmllc-plant-model-bucket/{feature}-fc-weights.pt").content), map_location=torch.device(device)), strict=False)
    model = model.half()
    return model

start = dt.datetime.now()
flower = loadModel('flower',flowerLabelSet)
leaf = loadModel('leaf',leafLabelSet)
fruit = loadModel('fruit',fruitLabelSet)
print(dt.datetime.now() - start)

def processImage(imagePath, feature):
    with open(fr'{THIS_FOLDER}/resources/{feature}MeansAndStds.pkl', 'rb') as f:
        meansAndStds = pkl.load(f)

    img = Image.open(imagePath).convert('RGB')
    cropped = ImageOps.fit(img, (224,224), Image.Resampling.LANCZOS)

    process = T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(
       mean=meansAndStds['mean'],
       std=meansAndStds['std'])
    ])

    return process(cropped)

def see(tensor,feature,k):

        if feature=='flower':
                model = flower.float()
                labelSet = flowerLabelSet
        
        elif feature=='leaf':
                model = leaf.float()
                labelSet = leafLabelSet
        
        elif feature=='fruit':
                model = fruit.float()
                labelSet = fruitLabelSet
        
        with torch.no_grad():        
                output = model(tensor.unsqueeze(0))
                top = torch.topk(output,k,dim=1)
                predictions = top.indices[0]
        
        predictedSpecies = []
        for i in predictions:
                predictedSpecies.append(labelSet[i])
        
        model = None
        gc.collect()
        return predictedSpecies
