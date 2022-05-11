from typing import List
import torch
import torch, torchvision

# from models.load_pretrained_models import load_model
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torchvision import datasets
import time
import sys
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import numpy as np
import torchvision.models as models
from torchinfo import summary


def k_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk=(1,)
) -> List[torch.FloatTensor]:
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(
            topk
        )  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = (
            y_pred.t()
        )  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(
            y_pred
        )  # [B] -> [B, 1] -> [maxk, B]
        target_reshaped = target_reshaped.to(device)
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (
            y_pred == target_reshaped
        )  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.reshape(-1).float()
            )  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=True
            )  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = {"dataset": sys.argv[1], "weights": sys.argv[2]}
print(f"Arguments passed: dataset is {args['dataset']}, weights are {args['weights']}")

data_transforms = {
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
}

d_size = args["dataset"]


data_dir = f"heirarchy_data/{d_size}/wikipaintings_"
image_datasets = {
    x: datasets.ImageFolder(data_dir + x, data_transforms[x]) for x in ["test"]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=32, shuffle=True, num_workers=12
    )
    for x in ["test"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["test"]}
classes = image_datasets["test"].classes
print(classes)

model = models.resnext101_32x8d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
weights = torch.load(args["weights"], map_location="cpu")
model.load_state_dict(weights)

model.eval()
model.to(device)

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
top_1_acc = 0
top_3_acc = 0
top_5_acc = 0
batch_count = 0

# again no gradients needed
with torch.no_grad():
    for data in dataloaders["test"]:
        # batch_count += 1
        images, labels = data
        images = images.to(device)
        outputs = model(images)
        for output, label in zip(outputs, labels):
            # print(output)
            # print(label)
            batch_count += 1
            top_k_accs = k_accuracy(outputs, labels, topk=(1, 3, 5))
            top_1_acc += top_k_accs[0][0]
            top_3_acc += top_k_accs[1][0]
            top_5_acc += top_k_accs[2][0]
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    # for batch in dataloaders["test"]:
    # image,label = batch
    # output=model(image)
    # # print(output)

total_correct = 0
pred_tot = 0
print(total_pred)
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    total_correct += correct_count
    pred_tot += total_pred[classname]
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

total_acc = 100 * total_correct / pred_tot
k_1_acc = 100 * top_1_acc / batch_count
k_3_acc = 100 * top_3_acc / batch_count
k_5_acc = 100 * top_5_acc / batch_count

print(f"Accuracy is {total_acc:.1f} %")

print(f"Accuracy for top 1: is {k_1_acc:.1f} %")
print(f"Accuracy for top 3: is {k_3_acc:.1f} %")
print(f"Accuracy for top 5: is {k_5_acc:.1f} %")
