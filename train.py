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

        for batch_idx, (data_a, data_p, data_n) in tqdm(enumerate(train_data_loader)):
            data_a = data_a.unsqueeze(1).float()
            data_p = data_p.unsqueeze(1).float()
            data_n = data_n.unsqueeze(1).float()
            print("data_a.shape:", data_a.shape)
            print("data_p.shape:", data_p.shape)
            print("data_n.shape:", data_n.shape)
            pred = model(data_a)
            print("pred:", pred)


            exit(0)
        '''

        for data in iter(train_data_loader):
            # Counter
            iter_idx += 1

            # Split the data
            x, y = data

            # Send data to GPU if we have one
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Apply the model to obtain scores (forward pass)
            logits = model.forward(x)
            # Compute the loss
            loss = data_loss(logits, y) + model_loss(model)
            # Compute gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            # Zero the parameter gradients in the optimizer
            optimizer.zero_grad()

            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:
                # Compute accuracy (No gradients required). We'll wrapp this
                # part so that we prevent torch from computing gradients.
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    acc = torch.mean(torch.eq(pred, y).float()) * 100.0
                # Write loss and accuracy to tensorboard, using keywords `loss`
                # and `accuracy`.
                tr_writer.add_scalar("loss", loss, global_step=iter_idx)
                tr_writer.add_scalar("accuracy", acc, global_step=iter_idx)
                # Save
                torch.save({
                    "iter_idx": iter_idx,
                    "best_va_acc": best_va_acc,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, checkpoint_file)

            # Validate results every validation interval
            if iter_idx % config.val_intv == 0:
                # List to contain all losses and accuracies for all the
                # training batches
                va_loss = []
                va_acc = []
                # Set model for evaluation
                model = model.eval()
                for data in va_data_loader:

                    # Split the data
                    x, y = data

                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()

                    # Apply forward pass to compute the losses
                    # and accuracies for each of the validation batches
                    with torch.no_grad():
                        # Compute logits
                        logits = model.forward(x)
                        # Compute loss and store as numpy
                        loss = data_loss(logits, y) + model_loss(model)
                        va_loss += [loss.cpu().numpy()]
                        # Compute accuracy and store as numpy
                        pred = torch.argmax(logits, dim=1)
                        acc = torch.mean(torch.eq(pred, y).float()) * 100.0
                        va_acc += [acc.cpu().numpy()]
                # Set model back for training
                model = model.train()
                # Take average
                va_loss = np.mean(va_loss)
                va_acc = np.mean(va_acc)

                # Write to tensorboard using `va_writer`
                va_writer.add_scalar("loss", va_loss, global_step=iter_idx)
                va_writer.add_scalar("accuracy", va_acc, global_step=iter_idx)
                # Check if best accuracy
                if va_acc > best_va_acc:
                    best_va_acc = va_acc
                    # Save best model using torch.save. Similar to previous
                    # save but at location defined by `bestmodel_file`
                    torch.save({
                        "iter_idx": iter_idx,
                        "best_va_acc": best_va_acc,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, bestmodel_file)
        '''