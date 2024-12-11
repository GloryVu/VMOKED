from CNNLSTM.model import CNNLSTM
from dataset.dataset import SmokeDataset
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from time import sleep
import warnings
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time
import shutil
warnings.filterwarnings("ignore")

device = ("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = SmokeDataset(r"dataset/wildfire_smoke_dataset/classification/test",None,transform,15)
batch_size = 1

# train_loader = DataLoader(dataset=train_set, batch_size=64, pin_memory=True, shuffle=True)
test_loader = DataLoader(dataset=dataset, batch_size=1, pin_memory=True, shuffle=True)

model = CNNLSTM()
model.load_state_dict(torch.load("CNNLSTM/save/model_best.pth"))
model.to(device)
model.eval()
# smoke: 1
# nonsmoke: 0
epsilon = 0.0000001
with torch.no_grad():
    latency = 0
    num_correct = 0 
    num_samples = 0
    loop = tqdm(test_loader, total=len(test_loader), position=0, leave=False)
    tp = tn = fp = fn = 0
    for x, y in test_loader:
        x = x.to(device=device)
        y = y.to(device=device)
        start_time = time.time()
        pred = model(x)
        
        end_time= time.time()
        pred_idx = torch.argmax(pred, dim=1)
        tp +=  torch.sum((pred_idx==torch.ones_like(pred_idx)) and (y == torch.ones_like(y)))
        tn += torch.sum((pred_idx==torch.zeros_like(pred_idx)) and (y == torch.zeros_like(y)))
        fp += torch.sum((pred_idx==torch.ones_like(pred_idx)) and (y == torch.zeros_like(y)))
        fn += torch.sum((pred_idx==torch.zeros_like(pred_idx)) and (y == torch.ones_like(y)))
        precision = tp/(tp+fp+epsilon)
        recall = tp/(tp+fn+epsilon)
        f1 = 2*precision*recall/(precision+recall+epsilon)
        _, predictions = torch.max(pred, 1)
        # print(predictions)
        num_correct += (pred_idx == y).sum().item()
        num_samples += y.size(0)

        val_acc = round(num_correct / num_samples, 4)
        latency_time = end_time - start_time
        latency+=latency_time
        loop.set_postfix(precision=f'{precision:.4f}', recall=f'{recall:.4f}', f1=f'{f1:.4f}',val_acc=f'{val_acc:.4f}')
        loop.update(1)
    print(f'accuracy: {round(num_correct / num_samples, 4)}')
    print(f'average latency: {latency*1000 / num_samples:.2f}')
model.train()
