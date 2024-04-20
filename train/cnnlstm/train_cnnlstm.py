from ...CNNLSTM.model import CNNLSTM
from dataset import SmokeDataset
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from time import sleep
import warnings
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
warnings.filterwarnings("ignore")

device = ("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = SmokeDataset(r"data/dataset_v1/train",None,transform,15)
# test_set = SmokeDataset(r"data/dataset/test",15)
batch_size = 64
validation_split = .2
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
# train_loader = DataLoader(dataset=train_set, batch_size=64, pin_memory=True, shuffle=True)
# valid_loader = DataLoader(dataset=valid_set, batch_size=64, pin_memory=True, shuffle=True)

model = CNNLSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()

num_epochs = 100

best_val_acc = 0

for epoch in range(num_epochs):
    if epoch % 5 == 0:
        loop = tqdm(train_loader, total=len(train_loader) + len(valid_loader), position=0, leave=False)
    else:
        loop = tqdm(train_loader, total=len(train_loader), position=0, leave=False)

    for x, y in train_loader:
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Training")

        x = x.to(device=device)
        y = y.to(device=device)

        outputs = model(x)

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(train_loss=loss.item())
        loop.update(1)
        sleep(0.1)

    if epoch % 5 == 0:
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in valid_loader:
                loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Validing")
                x = x.to(device=device)
                y = y.to(device=device)

                pred = model(x)
                pred_idx = torch.argmax(pred, dim=1)

                num_correct += (pred_idx == y).sum().item()
                num_samples += y.size(0)

                val_acc = round(num_correct / num_samples, 3)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), "model/CNNLSTM/save/model_best.pth")

                loop.set_postfix(val_accuracy=val_acc)
                loop.update(1)
                sleep(1)

        model.train()
