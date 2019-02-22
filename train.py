from __future__ import print_function
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
import h5py
import dataset
import random
import math
from utils import *
from region_loss import RegionLoss
from models import *
#from model_candidate2 import *
import numpy as np
import json
import argparse
import bitstring



parser = argparse.ArgumentParser(description='PyTorch Training for UAV Object Detection')
parser.add_argument('--model', '-m', '-model',
                    help='the neural network model to be trained')
parser.add_argument('--train-list', '-train', '-trainl',
                    help='training data list (.txt)')
parser.add_argument('--test-list', '-test', '-testl',
                    help='testing data list (.txt) (default: \'test.txt (in UAV dataset)\')')
parser.add_argument('--pre-trained-weight', '-weight', '-w', type=str,
                    help='pre trained weight for selected network model')
parser.add_argument('--gpus', '-g', type=str, default='0',
                    help='GPUs (id) to be used (default: \'0\')')
parser.add_argument('--backupdir', '-backup', type=str, default='backup',
                    help='backup weights stored location (default: backup)')
parser.add_argument('--eval', '-eval', '-rt', action='store_true')
parser.add_argument('--fixed-point', '-fixed', type=str, default='',
                    help='fixed point format for weights, (Integer part, decimal part) (default: empty)')

global args
args = parser.parse_args()


modeltype = args.model
trainlist       = args.train_list
weightfile    = args.pre_trained_weight
gpus = args.gpus
testlist = args.test_list
backupdir = 'backup'
evaluate = args.eval

## get fixed point setting for weights
prune = False
int_n = 16
frac_n = 16
if( args.fixed_point != '' ):
    prune = True
    fixed_format = args.fixed_point.split(',')
    int_n = int(fixed_format[0])
    frac_n = int(fixed_format[1])


ngpus = len(gpus.split(','))
print (ngpus)

num_workers = 10
batch_size    = 32
learning_rate = 0.001
momentum      = 0.9
decay         = 0.0005
steps         = [-1,100,20000,30000,100000]
scales        = [.1,10,.1,.1,.1]

#Train parameters
max_epochs = 10000
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
save_interval = 1  # epoches


# Test parameters
best_iou = 0
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
model       = eval(modeltype+'()')
region_loss = model.loss
region_loss.seen  = model.seen
processed_batches = 0


# start
if( weightfile ):
    load_net(weightfile,model)

init_width        = model.width
init_height       = model.height
init_epoch=0
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],std = [ 0.25, 0.25, 0.25 ]),
                   ]), train=False),
    batch_size=batch_size, shuffle=True, **kwargs)
if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
        print ("using cuda")
params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]
        
optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

def adjust_learning_rate(optimizer, batch):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch):
    global processed_batches
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],std = [ 0.25, 0.25, 0.25 ]),
                       ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)
    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train()
    
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        
        if use_cuda:
            data = data.cuda() 
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss = region_loss(output, target)
        loss.backward()
        optimizer.step()
    t1 = time.time()    
    logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    if (epoch+1) % save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        model.seen = (epoch + 1) * len(train_loader.dataset)
        save_net('%s/temp%s%06d_%d_%d.weights' % (backupdir, modeltype, epoch + 1, int_n, frac_n), model)


# def convert_to_fixed(val):
#     global int_n, frac_n
#     val_abs = abs(val)
#     int_fix = int(val_abs)
#     if( int_fix > 2**(int_n-1) ):
#         int_fix = 2**(int_n-1)
#     frac_fix = val_abs - int(val_abs)
#     minimum = 1.0 / 2**(frac_n)
#     frac_fix = round(frac_fix / minimum) * minimum
#     return (int_fix + frac_fix) * np.sign(val)


def fixed_tensor(v, ilen, dlen):
    v.data = torch.clamp(v.data, min=-2**ilen, max=2**ilen).double()
    v.data = v.data / (2**-dlen)
    v.data = torch.round(v.data)
    v.data = v.data / (2**dlen)
    return v.data


def weight_to_fixed(int_n, frac_n):
    for k, v in model.state_dict().items():
        v.copy_(fixed_tensor(v.data, int_n, frac_n))
    

def test(epoch, b_iou):

    model.eval()

    if( prune ):
        weight_to_fixed(int_n, frac_n)
        print("weights are transformed to fixed point <%d, %d>" % (int_n, frac_n))
    
    anchors     = model.anchors
    num_anchors = model.num_anchors
    anchor_step = len(anchors)/num_anchors
    total       = 0.0
    proposals   = 0.0


    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data = torch.autograd.Variable(data)
        if use_cuda:
            data = data.cuda()

        output = model(data).data
   
        batch = output.size(0)
        h = output.size(2)
        w = output.size(3)
        output = output.view(batch*num_anchors, 5, h*w).transpose(0,1).contiguous().view(5, batch*num_anchors*h*w)
        grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[1]) + grid_y

        anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
        anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
        ws = torch.exp(output[2]) * anchor_w
        hs = torch.exp(output[3]) * anchor_h
        det_confs = torch.sigmoid(output[4])
        sz_hw = h*w
        sz_hwa = sz_hw*num_anchors
        det_confs = convert2cpu(det_confs)
        xs = convert2cpu(xs)
        ys = convert2cpu(ys)
        ws = convert2cpu(ws)
        hs = convert2cpu(hs)        
        
        for b in range(batch):
            det_confs_inb = det_confs[b*sz_hwa:(b+1)*sz_hwa].numpy()
            xs_inb = xs[b*sz_hwa:(b+1)*sz_hwa].numpy()
            ys_inb = ys[b*sz_hwa:(b+1)*sz_hwa].numpy()
            ws_inb = ws[b*sz_hwa:(b+1)*sz_hwa].numpy()
            hs_inb = hs[b*sz_hwa:(b+1)*sz_hwa].numpy()      
            ind = np.argmax(det_confs_inb)
            
            bcx = xs_inb[ind]
            bcy = ys_inb[ind]
            bw = ws_inb[ind]
            bh = hs_inb[ind]
            
            box = [bcx/w, bcy/h, bw/w, bh/h]

            print("\ndetected: ", box)
            print("targeted: ", target[b][1:5])

            iou = bbox_iou(box, target[b][1:5], x1y1x2y2=False)
            proposals = proposals + iou
            total = total+1
        

    avg_ious = proposals/total
    logging("iou: %f, best iou: %f" % (avg_ious,b_iou))
    if avg_ious > b_iou:
        b_iou = avg_ious
        if( not evaluate ):
            save_net('%s/best%s.weights' % (backupdir, modeltype), model)

    return b_iou
        
        
if evaluate:
    logging('evaluating ...')

    test(0, best_iou)
else:
    for epoch in range(init_epoch, max_epochs): 

        if( epoch % 10 == 0):
            weight_to_fixed(int_n, frac_n)

        train(epoch)

        if epoch % 1==0:
            best_iou = test(epoch,best_iou)
        


# for k, v in model.state_dict().items():
#     print(v)
