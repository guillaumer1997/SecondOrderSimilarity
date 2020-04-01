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


lib_train = phototour.PhotoTour('.','liberty', download=True, train=True, mode = 'triplets', augment = True, nsamples=409600)
yos_train = phototour.PhotoTour('.','yosemite', download=True, train=True, mode = 'triplets', augment = True)
nd_train = phototour.PhotoTour('.','notredame', download=True, train=True, mode = 'triplets', augment = True)
eval_db = phototour.PhotoTour('.','yosemite', download=True, train=False)
train_db = nd_train
train_name = 'notredame'
train_data = train_db
test_data = eval_db
train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=300, shuffle=False, num_workers=30)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=32)

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


def train():
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

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Initialize training
    iter_idx = -1  # make counter start at zero
    best_loss = -1  # to check if best loss
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")
    bestmodel_file = os.path.join(config.save_dir, "best_model.pth")

    model = sosnet_model.SOSNet32x32().cuda()
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

    fpr_per_epoch = []
    # Training loop
    for epoch in range(config.num_epoch):
        # For each iteration
        prefix = "Training Epoch {:3d}: ".format(epoch)
        print("len(train_data_loader):", len(train_data_loader))
        for batch_idx, (data_a, data_p, data_n) in tqdm(enumerate(train_data_loader)):
            print("batch_idx:", batch_idx)
            print("len(train_data_loader):", len(train_data_loader))
            data_a = data_a.unsqueeze(1).float().cuda()
            data_p = data_p.unsqueeze(1).float().cuda()
            data_n = data_n.unsqueeze(1).float().cuda()
            print("data_a.shape:", data_a.shape)
            print("data_p.shape:", data_p.shape)
            print("data_n.shape:", data_n.shape)
            out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
            print("out_a:", out_a)
            print("out_p:", out_p)
            print("out_n:", out_n)
            loss = F.triplet_margin_loss(out_a, out_p, out_n, margin=2, swap=True)
            if best_loss == -1:
                best_loss = loss
            if loss < best_loss:
                best_loss = loss
                # Save
                torch.save({
                    "iter_idx": iter_idx,
                    "best_loss": best_loss,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, bestmodel_file)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save
        torch.save({
            "iter_idx": iter_idx,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, checkpoint_file)

        model.eval()

        l = np.empty((0,))
        d = np.empty((0,))
        #evaluate the network after each epoch
        for batch_idx, (data_l, data_r, lbls) in enumerate(test_data_loader):
            data_l = data_l.unsqueeze(1).float().cuda()
            data_r = data_r.unsqueeze(1).float().cuda()
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
        print(epoch,fpr95)
        fpr_per_epoch.append([epoch,fpr95])
        scheduler.step()
        np.savetxt('fpr.txt', np.array(fpr_per_epoch), delimiter=',') 

       


def init():
     # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

if __name__ == "__main__":
    train()