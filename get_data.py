

import os

import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils import * 
from metrics import * 
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *
from SFM import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2,sdd')    
args = parser.parse_args()
print(args)
print('*'*30)
print("Data processing....")

obs_seq_len = 8
pred_seq_len =12
data_set = './datasets/'+args.dataset+'/'

dset_train = TrajectoryDataset(
        data_set+'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)
torch.save(dset_train,"./data/"+args.dataset+"_train.pt")



dset_val = TrajectoryDataset(
        data_set+'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)
torch.save(dset_val,"./data/"+args.dataset+"_val.pt")

#Defining the model 
dset_test = TrajectoryDataset(
        data_set+'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)
torch.save(dset_test,"./data/"+args.dataset+"_test.pt")
print(args.dataset+" process done")



