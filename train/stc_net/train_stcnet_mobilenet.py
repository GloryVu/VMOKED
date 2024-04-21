from STCNet.src.STCNet_mobilenetv2.stcnet import STCNet
from dataset.dataset import STCNetDataset
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
model = STCNet().to(device)
random_transform=model.get_augmentation()
hard_transform =transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(model.input_mean, model.input_std)])
tran_set = STCNetDataset(r"dataset/wildfire_smoke_dataset/classification/train",random_transform,hard_transform, 8, alpha=10)
test_set = STCNetDataset(r"dataset/wildfire_smoke_dataset/classification/test",None,hard_transform,8,alpha=10)
batch_size = 4
random_seed= 42
train_loader = DataLoader(dataset=tran_set, batch_size=batch_size, pin_memory=True, shuffle=True)
valid_loader = DataLoader(dataset=test_set, batch_size=1, pin_memory=True, shuffle=True)



criterion = nn.CrossEntropyLoss()
optimizer_policies = model.get_optim_policies()
optimizer = torch.optim.SGD(model.get_optim_policies(), lr=0.001, momentum=0.9, weight_decay=0.0005)
model.train()

num_epochs = 20

best_val_acc = 0

for epoch in range(num_epochs):
    # if epoch % 5 == 0:
    loop = tqdm(train_loader, total=len(train_loader) + len(valid_loader), position=0, leave=False)
    # else:
    #     loop = tqdm(train_loader, total=len(train_loader), position=0, leave=False)

    for (frames,res_frames), y in train_loader:
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Training")
        frames = frames.to(device=device)
        res_frames = res_frames.to(device=device)
        y = y.to(device=device)

        outputs = model(frames,res_frames)

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(train_loss=loss.item())
        loop.update(1)

    if epoch % 1 == 0:
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for (frames,res_frames), y in valid_loader:
                loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Validing")
                frames = frames.to(device=device)
                res_frames = res_frames.to(device=device)
                y = y.to(device=device)

                pred = model(frames,res_frames)
                pred_idx = torch.argmax(pred, dim=1)

                num_correct += (pred_idx == y).sum().item()
                num_samples += y.size(0)

                val_acc = round(num_correct / num_samples, 4)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "STCNet/save/STCMobilenetv2_model_best.pth")

            loop.set_postfix(val_accuracy=val_acc)
            loop.update(1)

        model.train()
