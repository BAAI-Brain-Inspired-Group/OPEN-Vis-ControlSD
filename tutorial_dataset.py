import json
import cv2
import numpy as np
from pycocotools.coco import COCO

from torch.utils.data import Dataset
from torchvision import transforms
from annotator.util import resize_image, HWC3
from cv2 import Canny


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
class MyDatasetHC(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        targ = (target.astype(np.float32) / 127.5) - 1.0
        cond = target.astype(np.float32) / 255.0

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDatasetBihua(Dataset):
    def __init__(self):
        self.data = []
        with open('./bihua/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./bihua/' + source_filename)
        target = cv2.imread('./bihua/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target,(512,512))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        targ = (target.astype(np.float32) / 127.5) - 1.0
        cond = target.astype(np.float32) / 255.0

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDatasetShuimo(Dataset):
    def __init__(self):
        self.data = []
        with open('./shuimo/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./shuimo/' + source_filename)
        target = cv2.imread('./shuimo/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target,(512,512))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        targ = (target.astype(np.float32) / 127.5) - 1.0
        cond = target.astype(np.float32) / 255.0

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDatasetShuimoCanny(MyDatasetShuimo):
    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./shuimo/' + source_filename)
        target = cv2.imread('./shuimo/' + target_filename)
        
        # Do not forget that OpenCV read images in BGR order.
        detected_map = Canny(cv2.resize(target,(512,512)), 100, 200)
        cond = HWC3(detected_map).astype(np.float32)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        targ = cv2.resize(target,(512,512))
        # print('MyDatasetCOCO target:',target.shape)

        # Normalize target images to [-1, 1].
        targ = (targ / 127.5) - 1.0
        cond = cond / 255.

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDatasetBihuaCanny(MyDatasetBihua):
    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./bihua/' + source_filename)
        target = cv2.imread('./bihua/' + target_filename)
        
        # Do not forget that OpenCV read images in BGR order.
        detected_map = Canny(cv2.resize(target,(512,512)), 100, 200)
        cond = HWC3(detected_map).astype(np.float32)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        targ = cv2.resize(target,(512,512))
        # print('MyDatasetCOCO target:',target.shape)

        # Normalize target images to [-1, 1].
        targ = (targ / 127.5) - 1.0
        cond = cond / 255.

        return dict(jpg=targ, txt=prompt, hint=cond)

class MyDatasetCOCO(Dataset):
    def __init__(self):
        self.data = []
        path = '/home/chenzhiqiang/data/ms-coco/annotations/captions_train2017.json'
        self.dir_train = '/home/chenzhiqiang/data/ms-coco/train2017'
        self.dir_val = '/home/chenzhiqiang/data/ms-coco/val2017'
        self.coco = COCO(path)
        # self.data = self.coco.getImgIds()
        self.data = self.coco.loadImgs(self.coco.getImgIds())
        # self.anns = self.coco.loadAnns(self.coco.getAnnIds())
        # self.trans = transforms.Resize(size=(512,512))
        # print('MyDatasetCOCO data[0]:',self.data[0],'weight:')
        # img = self.coco.loadImgs(self.data[32])
        # print('img:',img[0]['file_name'],img[0])
        # annotation = coco.loadAnns(self.data[0])
        # print('annotation:',annotation[0]['caption'])
        # self.data = coco.getImgIds()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # print(item)

        target_filename = self.dir_train + '/' + item['file_name']
        target = cv2.imread(target_filename)
        prompt = self.coco.loadAnns(self.coco.getAnnIds(item['id']))[0]['caption']
        # prompt = self.anns[idx]['caption']

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        targ = cv2.resize(target,(512,512))
        cond = cv2.resize(target,(512,512))
        # print('MyDatasetCOCO target:',target.shape)

        # Normalize target images to [-1, 1].
        targ = (targ / 127.5) - 1.0
        cond = cond / 255.
        # print('MyDatasetCOCO target:',targ.shape,'cond:',cond.shape,'prompt:',prompt)

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDatasetCOCO_val(MyDatasetCOCO):
    def __init__(self):
        path = '/home/chenzhiqiang/data/ms-coco/annotations/captions_val2017.json'
        self.dir_train = '/home/chenzhiqiang/data/ms-coco/val2017'
        self.coco = COCO(path)
        # self.data = self.coco.getImgIds()
        self.data = self.coco.loadImgs(self.coco.getImgIds())

class MyDatasetCOCO_canny(MyDatasetCOCO):
    def __getitem__(self,idx):
        item = self.data[idx]
        # print(item)

        target_filename = self.dir_train + '/' + item['file_name']
        target = cv2.imread(target_filename)
        prompt = self.coco.loadAnns(self.coco.getAnnIds(item['id']))[0]['caption']
        # prompt = self.anns[idx]['caption']

        # Do not forget that OpenCV read images in BGR order.
        detected_map = Canny(cv2.resize(target,(512,512)), 100, 200)
        cond = HWC3(detected_map).astype(np.float32)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        targ = cv2.resize(target,(512,512))
        # print('MyDatasetCOCO target:',target.shape)

        # Normalize target images to [-1, 1].
        targ = (targ / 127.5) - 1.0
        cond = cond / 255.
        # print('MyDatasetCOCO target:',targ.shape,'cond:',cond.shape,'prompt:',prompt)

        return dict(jpg=targ, txt=prompt, hint=cond)
    
class MyDatasetCOCO_val_canny(MyDatasetCOCO_val):
    def __getitem__(self,idx):
        item = self.data[idx]
        # print(item)

        target_filename = self.dir_train + '/' + item['file_name']
        target = cv2.imread(target_filename)
        prompt = self.coco.loadAnns(self.coco.getAnnIds(item['id']))[0]['caption']
        # prompt = self.anns[idx]['caption']

        # Do not forget that OpenCV read images in BGR order.
        detected_map = Canny(cv2.resize(target,(512,512)), 100, 200)
        cond = HWC3(detected_map).astype(np.float32)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        targ = cv2.resize(target,(512,512))
        # print('MyDatasetCOCO target:',target.shape)

        # Normalize target images to [-1, 1].
        targ = (targ / 127.5) - 1.0
        cond = cond / 255.
        # print('MyDatasetCOCO target:',targ.shape,'cond:',cond.shape,'prompt:',prompt)

        return dict(jpg=targ, txt=prompt, hint=cond)
