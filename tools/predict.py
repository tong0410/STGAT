import numpy as np
import torch
from model import models
from model.config import get_parameters
import os
from script import utility, dataloader
from sklearn import preprocessing


args, device, blocks = get_parameters()
device = torch.device('cpu')
dataset_path = '../data'
dataset_name = 'RS1'
dataset_path = os.path.join(dataset_path, dataset_name)

distance = np.load(os.path.join(dataset_path, 'distance_value.npy'))
# the number of nodes
n_vertex = distance.shape[1]
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
# shortpath = shortpath_temp.toarray()
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


model = models.STGAT(args, blocks, n_vertex).to(device)
model.load_state_dict(torch.load('../%s/test/final_model_50_1_5.pth' % (dataset_name)))
model.eval()


x_data = np.load(os.path.join(dataset_path, 'values_6.npy'))
# predict step length
prediction_steps = 2950
initial_input = x_data[22000:22050]
true = x_data[22050 : 22050 + prediction_steps]

zscore = preprocessing.StandardScaler()
x_data = zscore.fit_transform(x_data.reshape(-1, args.n_features)).reshape(x_data.shape)
initial_input = zscore.transform(initial_input.reshape(-1, args.n_features)).reshape(initial_input.shape)
true = zscore.transform(true.reshape(-1, args.n_features)).reshape(true.shape)

initial_input,_ = dataloader.data_transform(initial_input, args.n_his, args.n_pred, args.n_features, device)
initial_input = torch.Tensor(initial_input).to(device)
initial_input = initial_input.clone()

predictions = []

for step in range(prediction_steps):
    with torch.no_grad():
        distance_train = torch.zeros((1, args.n_his, n_vertex, n_vertex), device=device)
        shortpath_train = torch.zeros((1, args.n_his, n_vertex, n_vertex), device=device)
        quater_train = torch.zeros((1, args.n_his, n_vertex, n_vertex, 4), device=device)
        movement_train = torch.zeros((1, args.n_his, n_vertex, n_vertex, 3), device=device)
        distance_train[0] = args.distance[22000:22050]
        shortpath_train[0] = args.shortpath[22000:22050]
        quater_train[0] = args.quater[22000:22050]
        movement_train[0] = args.movement[22000:22050]
        pred = model(initial_input, distance_train, shortpath_train, quater_train, movement_train).squeeze(dim=2)

    predictions.append(pred.numpy())

    temp = initial_input
    initial_input[:, :, :-args.n_features, :] = temp[:, :, args.n_features:, :]
    initial_input[:, :, -args.n_features:, :] = pred
    if step % 50 == 0:
        print(step)

predictions = np.array(predictions)
pred_num, _, pred_features, pred_nodes = predictions.shape
predictions = predictions.squeeze().reshape(-1)

predictions = predictions.reshape(pred_num, pred_nodes, pred_features)
predictions = zscore.inverse_transform(predictions.reshape(-1, args.n_features)).reshape(predictions.shape)
true = zscore.inverse_transform(true.reshape(-1, args.n_features)).reshape(true.shape)


folder_path = '../%s/test/prediction/' %(dataset_name)
if not os.path.exists(folder_path):

    os.makedirs(folder_path)
    print(f"folder path: {folder_path} ")

np.save('%s/after_predictions_50_5_2950.npy' % (folder_path), predictions)
