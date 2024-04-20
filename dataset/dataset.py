import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageChops, ImageOps, ImageEnhance
import random
import numpy as np
import json
import xmltodict
import cv2
class RandomChoice(torch.nn.Module):
    def __init__(self, transforms, hardtransforms):
       super().__init__()
       self.transforms = transforms.transforms if transforms != None else None
       self.hardtransforms = hardtransforms if hardtransforms != None else None

    def __call__(self, imgs):
        
        if self.transforms != None:
            t = random.choice(self.transforms)
            imgs = [t(img) for img in imgs]
        if self.hardtransforms != None:
            imgs = [self.hardtransforms(img) for img in imgs]
        return imgs
class SmokeDataset(Dataset):
    def __init__(self, root,random_transforms = None, 
        hard_transforms=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), sequence_length=8):
        self.root = root
        self.all_class = sorted(os.listdir(root))
        self.transform = RandomChoice(random_transforms, hard_transforms)
        self.images_cache = {}
        self.x = []
        self.y = []
        print(self.all_class)
        # load images to ram
        for (aa, i) in enumerate(self.all_class):
            path = os.path.join(self.root, i)
            for vid_path in tqdm([os.path.join(path, vid) for vid in os.listdir(path)], f'load {i} images to cache'):
                
                all_images = os.listdir(vid_path)
                for image in all_images:
                    img = Image.open(os.path.join(vid_path, image)).convert('RGB').resize((224, 224))
                    self.images_cache[os.path.join(vid_path, image)] = img
        
        # load sequence samples
        for (aa, i) in enumerate(self.all_class):
            path = os.path.join(self.root, i)
            for vid_path in [os.path.join(path, vid) for vid in os.listdir(path)]:
                all_images = os.listdir(vid_path)
                all_images.sort()
                for j in range(0, len(all_images)-sequence_length):
                    image_paths = []
                    for k in all_images[j: j + sequence_length]:
                        image_paths.append(os.path.join(vid_path, k))
                    self.x.append(image_paths)
                    self.y.append(aa)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image_tensor = torch.tensor(())
        image_list =[]
        for image_path in self.x[idx]:
            image_list.append(self.images_cache[image_path])
        image_list = self.transform(image_list)
        for img in image_list:
            img = Variable(torch.unsqueeze(img, 0))
            image_tensor = torch.cat((image_tensor, img), 0)
        label = torch.tensor(self.y[idx])
        return image_tensor, label

class STCNetDataset(SmokeDataset):
    def __init__(self, root, random_transforms = None, hard_transforms=transforms.Compose(
        [transforms.ToTensor()]), sequence_length=8, alpha = 1.0):
        super(STCNetDataset, self).__init__(root, random_transforms, hard_transforms, sequence_length)
        self. alpha = alpha
    def _multiply_image(image, alpha):
        enhancer = ImageEnhance.Brightness(image)
        multiplied_image = enhancer.enhance(alpha)
        return multiplied_image
    def _res_frame(self,images):
        resframe_tensor = torch.tensor(())
        images.append(images[-1])
        for i in range(len(images)-1):
            
            # Take the absolute value of the subtracted images
            abs_subtracted = ImageChops.difference(images[i], images[i+1])

            # Apply maximum threshold
            thresholded = Image.eval(abs_subtracted, lambda x: 255 if x * self.alpha > 255 else int(x * self.alpha))
 
            resframe = Variable(torch.unsqueeze(transforms.ToTensor()(thresholded), 0))
            resframe_tensor = torch.cat((resframe_tensor, resframe), 0)
        return resframe_tensor
    
    def __getitem__(self,idx):
        image_tensor = torch.tensor(())
        image_list = []
        for image_path in self.x[idx]:
            image_list.append(self.images_cache[image_path])
        resframe = self._res_frame(image_list)
        image_list = self.transform(image_list)
        for img in image_list[:-1]:
            img = Variable(torch.unsqueeze(img, 0))
            image_tensor = torch.cat((image_tensor, img), 0)
        label = torch.tensor(self.y[idx])
        return (image_tensor, resframe), label
    
class SmokeDetectionDataset(Dataset):
    def __init__(self, root, label_dir,random_transforms = None, 
        hard_transforms=transforms.Compose(
        []), sequence_length=8):
        self.root = root
        self.all_class = sorted(os.listdir(root))
        self.label_dir = label_dir
        self.transform = RandomChoice(random_transforms, hard_transforms)
        self.images_cache = {}
        self.x = []
        self.y = []
        print(self.all_class)
        # load images to ram
        for (aa, i) in enumerate(self.all_class):
            path = os.path.join(self.root, i)
            for vid_path in tqdm([os.path.join(path, vid) for vid in os.listdir(path)], f'load {i} images to cache'):
                
                all_images = os.listdir(vid_path)
                for image in all_images:
                    img = cv2.imread(os.path.join(vid_path, image)) #Image.open(os.path.join(vid_path, image)).convert('RGB').resize((640, 640))
                    img = cv2.resize(img,(640,640))
                    # img = np.transpose(img, (2, 0, 1))
                    self.images_cache[os.path.join(vid_path, image)] = img
        
        # load sequence samples
        for (aa, i) in enumerate(self.all_class):
            path = os.path.join(self.root, i)
            for vid_path in [os.path.join(path, vid) for vid in os.listdir(path)]:
                all_images = os.listdir(vid_path)
                all_images.sort()
                for j in range(0, len(all_images)-sequence_length):
                    image_paths = []
                    all_images_seg =all_images[j: j + sequence_length]
                    all_images_seg.reverse()
                    for k in all_images_seg:
                        image_paths.append(os.path.join(vid_path, k))
                    self.x.append(image_paths)
                    self.y.append(self._load_boxes(image_paths[-1]) if i=='smoke' else [])
   
   
    def _load_boxes(self,image_path):
        image_name = os.path.basename(image_path)
        
        with open(os.path.join(self.label_dir,image_name.replace('.jpg','.txt'))) as xml_file:
            boxes =[]
            for label in [label for label in xml_file.read().split('\n') if len(label) !=0]:
                x,y,w,h = [float(num) for num in label.split()[1:]]
                xmin = x-w/2
                xmax = x+w/2
                ymin = y-h/2
                ymax = y+h/2
                boxes.append([0,[xmin,ymin,xmax,ymax]])
        return boxes      


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        image_tensor = torch.tensor(())
        image_list =[]
        for image_path in self.x[idx]:
            image_list.append(self.images_cache[image_path])
        # image_list = self.transform(image_list)
        # for img in image_list:
        #     img = Variable(torch.unsqueeze(img, 0))
        #     image_tensor = torch.cat((image_tensor, img), 0)
        # label = torch.tensor(self.y[idx])
        return image_list, self.y[idx]