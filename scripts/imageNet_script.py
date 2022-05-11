import torch, torchvision
import sys
from torchvision import transforms
from torchvision import datasets
import time
import os
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchinfo import summary
from PIL import ImageFile
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


args = {"dataset":sys.argv[1],"weights":sys.argv[2],"output_name":sys.argv[3]}
print(f"Arguments passed: dataset is {args['dataset']}, weights are {args['weights']}, output_name is : {args['output_name']}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark=True


batch_size = 256
num_workers = 64

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(),
            #transforms.RandomGrayscale(.7),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

d_size = args["dataset"]

print(f"Using dataset {d_size}")
data_dir = f"heirarchy_data/{d_size}/wikipaintings_"
image_datasets = {
    x: datasets.ImageFolder(data_dir + x, data_transforms[x])
    for x in ["train", "val", "test"]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True
    )
    for x in ["train", "val", "test"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
class_names = image_datasets["train"].classes

print(f"Found Train set with {dataset_sizes['train']} images and {len(image_datasets['train'].classes)} classes")
print(f"Found Validation set with {dataset_sizes['val']} images and {len(image_datasets['val'].classes)}")
print(f"Using classes {class_names}")

model = models.resnext101_32x8d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
weights = torch.load(args["weights"], map_location='cpu')
model.load_state_dict(weights)

for param in model.parameters():
    param.requires_grad = False

for layer_num,param in enumerate(model.parameters()):
    if layer_num > 200:
        param.requires_grad = True

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    last_loss = 100
    patience = 10
    trigger = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            loop = tqdm(dataloaders[phase])
            #for inputs, labels in dataloaders[phase]:
            for idx,(inputs,labels) in enumerate(loop):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #if phase == "train":
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "val":
                scheduler.step(epoch_loss)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == "val" and epoch_loss > last_loss:
                print("Model failed to improve, losing paitience!")
                trigger += 1
                print(f"Patience Triggered {trigger} consecutive times")

                if trigger > patience:
                    print("Out of patience, returning best model")
                    model.load_state_dict(best_model_wts)
                    return model

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                trigger = 0
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                last_loss = epoch_loss
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

criterion = FocalLoss() # nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_convnext = optim.AdamW(model.parameters(), lr=0.001*batch_size/256, )

exp_lr_scheduler_convnext = lr_scheduler.ReduceLROnPlateau(
    optimizer_convnext,patience = 4,verbose=True)

pytorch_total_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(f"Trainable Params: {pytorch_total_params}")
print(summary(model,(batch_size,3,224,224)))
print(f"batch size is {batch_size}")

model_ft = train_model(
    model,
    criterion,
    optimizer_convnext,
    exp_lr_scheduler_convnext,
    num_epochs=40,
)

torch.save(model_ft.module.state_dict(), args["output_name"]+"_weights")
torch.save(model_ft.module,args["output_name"])
