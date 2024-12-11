import os
import sys
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from CNNLSTM.model import CNNLSTM

# from STCNet.src.STCNet_se_resnext.stcnet import STCNet
from dataset.dataset import SmokeDataset

# from STCNet.src.
from STCNet.src.STCNet_mobilenetv2.stcnet import STCNet

warnings.filterwarnings("ignore")

def train_cnnlstm(data_dir="dataset/demo/classification", save_dir = "CNNLSTM/save", num_epochs = 30, batch_size = 64):
    device =  ("cuda" if torch.cuda.is_available() else "cpu")
    model = STCNet()
    random_transform=model.get_augmentation()
    hard_transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = SmokeDataset(os.path.join(data_dir,"train"),random_transform,hard_transform,15)
    test_set = SmokeDataset(os.path.join(data_dir,"test"),None,hard_transform,15)
    

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True, shuffle=True)

    model = CNNLSTM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    model.train()

    

    best_val_acc = 0

    for epoch in range(num_epochs):
        # if epoch % 1 == 0:
        loop = tqdm(train_loader, total=len(train_loader) + len(valid_loader), position=0, leave=False)
        # else:
        #     loop = tqdm(train_loader, total=len(train_loader), position=0, leave=False)

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

        if epoch % 1 == 0:
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
                    torch.save(model.state_dict(), os.path.join(save_dir,"model_best.pth"))

                loop.set_postfix(val_accuracy=val_acc)
                loop.update(1)
                # sleep(1)

            model.train()

if __name__ == '__main__':
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    num_epoch = sys.argv[3]
    batch_size = sys.argv[4]
    print(sys.argv)
    train_cnnlstm(data_dir,save_dir,num_epoch,batch_size)