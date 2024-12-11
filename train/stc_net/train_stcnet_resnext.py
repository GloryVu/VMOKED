import os
import random
import sys
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset.dataset import STCNetDataset
from STCNet.src.STCNet_se_resnext.stcnet import STCNet

warnings.filterwarnings("ignore")

def train_stc_resnext(data_dir="dataset/demo/classification", save_dir = "STCNet", num_epochs = 20, batch_size = 2):

    device =  ("cuda" if torch.cuda.is_available() else "cpu")
    model = STCNet().to(device)
    random_transform=model.get_augmentation()
    hard_transform =transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(model.input_mean, model.input_std)])
    tran_set = STCNetDataset(os.path.join(data_dir,"train"),random_transform,hard_transform, 8, alpha=10)
    test_set = STCNetDataset(os.path.join(data_dir,"test"),None,hard_transform,8,alpha=10)
    random_seed= 42
    random.seed(random_seed)
    train_loader = DataLoader(dataset=tran_set, batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer_policies = model.get_optim_policies()
    optimizer = torch.optim.SGD(model.get_optim_policies(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    # for policy in optimizer_policies:
    #     optimizer.add_param_group(policy)
    model.train()


    best_val_acc = 0

    for epoch in range(num_epochs):
        if epoch % 1 == 0:
            loop = tqdm(train_loader, total=len(train_loader) + len(valid_loader), position=0, leave=False)
        else:
            loop = tqdm(train_loader, total=len(train_loader), position=0, leave=False)

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

                    val_acc = round(num_correct / num_samples, 3)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(save_dir,"STCNet_se_resnext_model_best.pth"))

                loop.set_postfix(val_accuracy=val_acc)
                loop.update(1)
            model.train()

if __name__ == '__main__':
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    num_epoch = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    print(sys.argv)
    train_stc_resnext(data_dir,save_dir,num_epoch,batch_size)