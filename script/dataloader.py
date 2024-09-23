import os
import numpy as np
import math
from sklearn import preprocessing
import torch.utils as utils
from script import utility
import pandas as pd
import scipy.sparse as sp
import torch


def load_data(dataset_name, len_train, len_val, features_file):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    # vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))
    vel = np.load(os.path.join(dataset_path, features_file))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test

def seperate_edge(len_train, len_val, edge, n_his, n_pred, device):
    train = edge[: len_train]
    val = edge[len_train: len_train + len_val]
    test = edge[len_train + len_val:]
    train = edge_transform(train, n_his, n_pred).to(device)
    val = edge_transform(val, n_his, n_pred).to(device)
    test = edge_transform(test, n_his, n_pred).to(device)

    return train, val, test

def data_transform(data, n_his, n_pred, n_features, device):
    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred + 1
    if num == 0:
        num = 1

    x = np.zeros([num, n_features, n_his, n_vertex])
    y = np.zeros([num, n_features, n_vertex])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = np.transpose(data[head: tail, :], (2, 0, 1))
        if num != 1:
            y[i, :, :] = np.transpose(data[tail + n_pred - 1, :], (1, 0))

    return x, y

def edge_transform(edge, n_his, n_pred):
    len_record = len(edge)
    num = len_record - n_his - n_pred + 1
    original_shape = edge.shape
    edge_output_shape = (num, n_his) + original_shape[1:]
    edge_output = torch.zeros(edge_output_shape)

    for i in range(num):
        head = i
        tail = i + n_his
        # edge_output[i, :, :, :] = np.transpose(edge[head: tail, :], (2, 0, 1))
        edge_output[i, :, :, :] = edge[head: tail, :]

    return edge_output


def data_preparate(args, device):
    dataset_path = './data'
    features_file = args.features_file
    dataset_path = os.path.join(dataset_path, args.dataset)

    data_col = np.load(os.path.join(dataset_path, features_file)).shape[0]
    val_and_test_rate = 0.15

    args.len_val = int(math.floor(data_col * val_and_test_rate))
    args.len_test = int(math.floor(data_col * val_and_test_rate))
    args.len_train = int(data_col - args.len_val - args.len_test)

    # load the first edge feature
    distance = np.load(os.path.join(dataset_path, 'distance_value.npy'))

    # the number of nodes
    n_vertex = distance.shape[1]
#######################
    distance_temp = np.zeros((distance.shape[0], n_vertex, n_vertex))
    distance = 1 - distance / np.max(distance)
    for i in range(distance.shape[0]):
        distance_temp[i, :, :] = utility.calc_gso(distance[i].squeeze(), args.gso_type).toarray()
    distance = distance_temp.astype(dtype=np.float32)

    args.distance = torch.from_numpy(distance).to(device)


    # load the second edge
    shortpath = np.load(os.path.join(dataset_path, 'path_length.npy'))
    shortpath = 1 - shortpath / np.max(shortpath)
    shortpath_temp = np.zeros((shortpath.shape[0], n_vertex, n_vertex))
    for i in range(shortpath.shape[0]):
        shortpath_temp[i] = utility.calc_gso(shortpath[i].squeeze(), args.gso_type).toarray()
    shortpath = shortpath_temp.astype(dtype=np.float32)

    args.shortpath = torch.from_numpy(shortpath).to(device)


    # load the third edge
    quater = np.load(os.path.join(dataset_path, 'quater_number.npy'))
    quater = quater.astype(dtype=np.float32)

    args.quater = torch.from_numpy(quater).to(device)

    # load the forth edge
    movement = np.load(os.path.join(dataset_path, 'movement_vector.npy'))
    movement = movement.astype(dtype=np.float32)

    args.movement = torch.from_numpy(movement).to(device)

##########################
    train, val, test = load_data(args.dataset, args.len_train, args.len_val, features_file)
    zscore = preprocessing.StandardScaler()

    # precessing now
    train = zscore.fit_transform(train.reshape(-1, args.n_features)).reshape(train.shape)
    val = zscore.transform(val.reshape(-1, args.n_features)).reshape(val.shape)
    test = zscore.transform(test.reshape(-1, args.n_features)).reshape(test.shape)

    x_train, y_train = data_transform(train, args.n_his, args.n_pred, args.n_features, device)
    x_val, y_val = data_transform(val, args.n_his, args.n_pred, args.n_features, device)
    x_test, y_test = data_transform(test, args.n_his, args.n_pred, args.n_features, device)
    x_train = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    x_val = torch.Tensor(x_val).to(device)
    y_val = torch.Tensor(y_val).to(device)
    x_test = torch.Tensor(x_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    test_num = y_test.shape[0]
    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)

    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter, test_num

