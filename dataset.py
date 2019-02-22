
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *
import json

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=32, num_workers=4):
      self.isJson = True
      file = open(root, 'r')

      try:
        self.lines = json.load(file)
        self.isJson = True
      except:
        file = open(root, 'r')
        self.lines = file.readlines()
        self.isJson = False

      if train == True:
        if self.isJson:
          newlines = []
          for j in self.lines:
            for k in j:
              newlines.append(k)
          
          self.lines = newlines
          random.shuffle(self.lines)
        else:
          random.shuffle(self.lines)
        self.nSamples = len(self.lines)
        print ("nSamples:" + str(self.nSamples))

      else:
        if self.isJson:
          self.nSamples = len(self.lines[0])
        else:
          self.nSamples = len(self.lines)

      self.transform = transform
      self.target_transform = target_transform
      self.train = train
      self.shape = shape
      self.seen = seen
      self.batch_size = batch_size
      self.num_workers = num_workers
       
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        if self.train and index % 32== 0:
            if self.seen < 4000*32:
               #width = 13*32
               #height = 13*32
               width = 20*8
               height = 40*8
               self.shape = (height, width)
            elif self.seen < 8000*32:
               width = (random.randint(-3,3) + 20)*8
               height = (random.randint(-3,3) + 40)*8
               self.shape = (height, width)
            elif self.seen < 12000*32:
               width = (random.randint(-5,5) + 20)*8
               height = (random.randint(-5,5) + 40)*8
               self.shape = (height, width)
            elif self.seen < 16000*32:
               width = (random.randint(-7,7) + 20)*8
               height = (random.randint(-7,7) + 40)*8
               self.shape = (height, width)
            else: # self.seen < 20000*64:
               width = (random.randint(-9,9) + 20)*8
               height = (random.randint(-9,9) + 40)*8
               self.shape = (height, width)

        if self.train:

          jitter = 0.2
          hue = 0.1
          saturation = 1.5 
          exposure = 1.5

          if self.isJson:
            info = self.lines[index]
            imgpath = info[0]
            box = info[1]
            label = np.zeros(10)
            label[1:3] = box[2:4]
            label[3:5] = box[0:2]
            img, label = load_data_detection(imgpath, label, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)


          else:
            imgpath = self.lines[index].rstrip()
            label = np.zeros(10)
            img, label = load_data_detection_2(imgpath, label, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
            
        else:
            if self.isJson:
              info = self.lines[0][index]
              imgpath = info[0]
              box = info[1]
              img = Image.open(imgpath).convert('RGB')
              if self.shape:
                img = img.resize(self.shape)
    
              label = np.zeros(10)
              label[1:3] = box[2:4]
              label[3:5] = box[0:2]
              label = torch.from_numpy(label)

            else:
              imgpath = self.lines[index].rstrip()
              img = Image.open(imgpath).convert('RGB')
              if self.shape:
                img = img.resize(self.shape)
                labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
                #labpath= imgpath.replace('.jpg', '.txt')
                label = torch.zeros(50*5)
                #print(label)
                #if os.path.getsize(labpath):
                #tmp = torch.from_numpy(np.loadtxt(labpath))
                try:
                    tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
                except Exception:
                    tmp = torch.zeros(1,5)
                    tmp = torch.from_numpy(tmp)

                #tmp = torch.from_numpy(read_truths(labpath))
                tmp = tmp.view(-1)
                tsz = tmp.numel()
                #print('labpath = %s , tsz = %d' % (labpath, tsz))
                if tsz > 50*5:
                    label = tmp[0:50*5]
                elif tsz > 0:
                    label[0:tsz] = tmp

                


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.batch_size
        return (img, label)