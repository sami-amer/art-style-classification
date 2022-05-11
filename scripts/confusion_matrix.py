import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch, torchvision
from torchvision import transforms
from torchvision import datasets
import time
import copy
import sys
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchinfo import summary
from PIL import ImageFile,Image
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = {"dataset":sys.argv[1],"weights":sys.argv[2],"output_name":sys.argv[3]}
print(f"Arguments passed: dataset is {args['dataset']}, weights are {args['weights']}, output_name is : {args['output_name']}")

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

d_size = args['dataset']


data_dir = f'heirarchy_data/{d_size}/wikipaintings_'
image_datasets = {x: datasets.ImageFolder(data_dir+x,
                                          data_transforms[x])
                  for x in ['test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256,
                                             shuffle=False, num_workers=48,pin_memory=True)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
classes = image_datasets['test'].classes
print(classes)

model = models.resnext101_32x8d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
weights = torch.load(args['weights'], map_location='cpu')
model.load_state_dict(weights)

y_pred = []
y_true = []

model.eval()
model.to(device)

with torch.no_grad():
    loop = tqdm(dataloaders["test"])
    for idx,(data) in enumerate(loop):
        images, labels = data
        images = images.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        y_pred.extend(predictions.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

cf_matrix = confusion_matrix(y_pred,y_true)
print(cf_matrix)
print(np.sum(cf_matrix,axis=0))
df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=0)).T *100, index = [i for i in classes],
                                          columns = [i for i in classes])
print(df_cm)
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig(args['output_name'])
