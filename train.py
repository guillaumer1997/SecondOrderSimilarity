import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
import numpy as np

import phototour
import math 

from config import get_config, print_usage, print_config
from tensorboardX import SummaryWriter

import sosnet_model



config, unparsed = get_config()
lib_train = phototour.PhotoTour('.','liberty', download=True, train=True, mode = 'triplets', augment = True, nsamples=409600)
yos_train = phototour.PhotoTour('.','yosemite', download=True, train=True, mode = 'triplets', augment = True)
nd_train = phototour.PhotoTour('.','notredame', download=True, train=True, mode = 'triplets', augment = True)
eval_db = phototour.PhotoTour('.','yosemite', download=True, train=False)
# train_db = torch.utils.data.ConcatDataset((lib_train, yos_train))
train_db = nd_train
train_name = 'notredame'
'''
EPOCHS = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.003
'''
TRAIN_DATA_PATH = "./train_cl/"
TEST_DATA_PATH = "./test_named_cl/"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])
data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1)
    ])
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])
ex_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

transform_1 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.CenterCrop(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])


'''
#train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data = torchvision.datasets.PhotoTour(root=TRAIN_DATA_PATH, name="liberty_harris", train=True, transform=transform_2, download=True)
train_data_loader = data.DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True,  num_workers=4)
#test_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
test_data = torchvision.datasets.PhotoTour(root=TRAIN_DATA_PATH, name="liberty_harris", train=False, transform=transform_2, download=True)
test_data_loader  = data.DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True, num_workers=4) 
'''
train_data = train_db
test_data = eval_db
train_data_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=300, shuffle=False,
                                             num_workers=30)

test_data_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1024, shuffle=False,
                                             num_workers=32)

def data_criterion(config):
    """Returns the loss object based on the commandline argument for the data term"""

    if config.loss_type == "cross_entropy":
        data_loss = nn.CrossEntropyLoss()
    elif config.loss_type == "svm":
        data_loss = nn.MultiMarginLoss()

    return data_loss


def model_criterion(config):
    """Loss function based on the commandline argument for the regularizer term"""

    def model_loss(model):
        loss = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                loss += torch.sum(param**2)

        return loss * config.l2_reg

    return model_loss


class CNN(nn.Module):
    # omitted...


    if __name__ == '__main__':
        # ----------------------------------------
        # Parse configuration
        config, unparsed = get_config()
        # If we have unparsed arguments, print usage and exit
        if len(unparsed) > 0:
            print_usage()
            exit(1)
        print_config(config)
        print("Number of train samples: ", len(train_data))
        print("Number of test samples: ", len(test_data))
        #print("Detected Classes are: ", train_data.class_to_idx) # classes are detected by folder structure

        model = sosnet_model.SOSNet32x32()
        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

        seed=42
        torch.manual_seed(seed)
        np.random.seed(seed)
         # Create loss objects
        data_loss = data_criterion(config)
        model_loss = model_criterion(config)
        #print("train_data: ", (train_data))
        print("train_data_loader:", train_data_loader)
        print("test_data_loader:", test_data_loader)


        # Training loop
    for epoch in range(config.num_epoch):
        # For each iteration
        prefix = "Training Epoch {:3d}: ".format(epoch)
        print("len(train_data_loader):", len(train_data_loader))
        for batch_idx, (data_a, data_p, data_n) in tqdm(enumerate(train_data_loader)):
            print("batch_idx:", batch_idx)
            print("len(train_data_loader):", len(train_data_loader))
            data_a = data_a.unsqueeze(1).float()
            data_p = data_p.unsqueeze(1).float()
            data_n = data_n.unsqueeze(1).float()
            print("data_a.shape:", data_a.shape)
            print("data_p.shape:", data_p.shape)
            print("data_n.shape:", data_n.shape)
            out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
            print("out_a:", out_a)
            print("out_p:", out_p)
            print("out_n:", out_n)
            loss = F.triplet_margin_loss(out_a, out_p, out_n, margin=2, swap=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        l = np.empty((0,))
        d = np.empty((0,))
        #evaluate the network after each epoch
        for batch_idx, (data_l, data_r, lbls) in enumerate(test_data_loader):
            data_l = data_l.unsqueeze(1).float()
            data_r = data_r.unsqueeze(1).float()
            out_l, out_r = model(data_l), model(data_r)
            dists = torch.norm(out_l - out_r, 2, 1).detach().cpu().numpy()
            l = np.hstack((l,lbls.numpy()))
            d = np.hstack((d,dists))
            
        # FPR95 code from Yurun Tian
        d = torch.from_numpy(d)
        l = torch.from_numpy(l)
        dist_pos = d[l==1]
        dist_neg = d[l!=1]
        dist_pos,indice = torch.sort(dist_pos)
        loc_thr = int(np.ceil(dist_pos.numel() * 0.95))
        thr = dist_pos[loc_thr]
        fpr95 = float(dist_neg.le(thr).sum())/dist_neg.numel()
        print(e,fpr95)
        #fpr_per_epoch.append([e,fpr95])
        #scheduler.step()
        #np.savetxt('fpr.txt', np.array(fpr_per_epoch), delimiter=',') 
        exit(0)
       