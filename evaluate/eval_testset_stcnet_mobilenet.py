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
import time
warnings.filterwarnings("ignore")

device = ("cuda" if torch.cuda.is_available() else "cpu")
model = STCNet(dropout=0.0).to(device)
hard_transform =transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(model.input_mean, model.input_std)])
dataset = STCNetDataset(r"dataset/wildfire_smoke_dataset/classification/test",None,hard_transform, 8, alpha=10)
batch_size = 4

# train_loader = DataLoader(dataset=train_set, batch_size=64, pin_memory=True, shuffle=True)
test_loader = DataLoader(dataset=dataset, batch_size=1, pin_memory=True, shuffle=True)


model.load_state_dict(torch.load("./STCNet/save/STCMobilenetv2_model_best.pth"))
model.eval()
epsilon = 0.0000001
with torch.no_grad():
    latency = 0
    num_correct = 0 
    num_samples = 0
    loop = tqdm(test_loader, total=len(test_loader), position=0, leave=False)
    tp = tn = fp = fn = 0
    for (frames,res_frames), y in test_loader:
        frames = frames.to(device=device)
        res_frames = res_frames.to(device=device)
        y = y.to(device=device)

    
        start_time = time.time()
        pred = model(frames,res_frames)
        
        end_time= time.time()
        pred_idx = torch.argmax(pred, dim=1)
        print(pred_idx)
        num_correct += (pred_idx == y).sum().item()
        num_samples += y.size(0)
         # 0: nonsmoke
        # 1: smoke
        tp +=  torch.sum((pred_idx==torch.ones_like(pred_idx)) and (y == torch.ones_like(y)))
        tn += torch.sum((pred_idx==torch.zeros_like(pred_idx)) and (y == torch.zeros_like(y)))
        fp += torch.sum((pred_idx==torch.ones_like(pred_idx)) and (y == torch.zeros_like(y)))
        fn += torch.sum((pred_idx==torch.zeros_like(pred_idx)) and (y == torch.ones_like(y)))
        precision = tp/(tp+fp+epsilon)
        recall = tp/(tp+fn+epsilon)
        f1 = 2*precision*recall/(precision+recall+epsilon)
        val_acc = round(num_correct / num_samples, 4)
        latency_time = end_time - start_time
        latency+=latency_time
        loop.set_postfix(precision=f'{precision:.4f}', recall=f'{recall:.4f}', f1=f'{f1:.4f}',test_accuracy=val_acc)
        loop.update(1)
    print(f'accuracy: {round(num_correct / num_samples, 4)}')
    print(f'average latency: {latency*1000 / num_samples:.2f}')
model.train()
