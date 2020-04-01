import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from config import get_config, print_usage, print_config
from tensorboardX import SummaryWriter
import train



def main(config):




if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    print_config(config)
    main(config)