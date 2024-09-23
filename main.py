import os
import math
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
# from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from model.config import get_parameters
from script import dataloader, earlystopping
from model import models
from model.train import train
from model.test import test


def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)
    model = models.STGAT(args, blocks, n_vertex).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return loss, es, model, optimizer, scheduler


if __name__ == "__main__":
    args, device, blocks = get_parameters()
    print('Training configs: {}'.format(args))

    n_vertex, zscore, train_iter, val_iter, test_iter, test_num = dataloader.data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)

    # print(summary(model, input_size=(3, 25, 111), batch_size=32))

    train(n_vertex, loss, args, optimizer, scheduler, es, model, train_iter, val_iter, device)
    test(zscore, loss, model, test_iter, args, test_num, args.n_features, n_vertex, device)
