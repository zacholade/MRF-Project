import torch

from models import *
import argparse
from config_parser import Configuration
from mrf_trainer import get_network

#  R2Plus1D_epoch-42_optim-Adam_initial-lr-0.001_loss-MSELoss_batch-size-100.pt

parser = argparse.ArgumentParser()
network_choices = ['cohen', 'oksuz']
parser.add_argument('-network', '-n', dest='network', choices=network_choices, type=str.lower, required=True)
parser.add_argument('-path', required=True)  # Path to the model + filename
args = parser.parse_args()

config = Configuration(args.network, "config.ini", args.debug)
model, using_spatial, using_attention = get_network(args.network)

checkpoint = torch.load(args.path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

start, stop, step = 0, 5000, 2
t1 = list(range(start, stop + step, step))

start, stop, step = 0, 2000, 2
t2 = list(range(start, stop + step, step))
