import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch

def adjust_edge_features(edge_matrix):
    edge_matrix = torch.from_numpy(edge_matrix)
    max_distance = edge_matrix.max()
    adjusted_matrix = 1 - (edge_matrix / max_distance)
    adjusted_matrix = torch.clamp(adjusted_matrix, min=0, max=1)
    return adjusted_matrix

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]
    # dir_adj = dir_adj.numpy()

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def evaluate_model(args, model, loss, data_iter, count_temp, n_vertex, device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            # y_pred = model(x).view(len(x), -1)
            distance_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
            shortpath_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
            quater_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 4), device=device)
            movement_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 3), device=device)

            try:
                for i in range(args.batch_size):
                    distance_test[i] = args.distance[count_temp + i: count_temp + args.n_his + i]
                    shortpath_test[i] = args.shortpath[count_temp + i: count_temp + args.n_his + i]
                    quater_test[i] = args.quater[count_temp + i: count_temp + args.n_his + i]
                    movement_test[i] = args.movement[count_temp + i: count_temp + args.n_his + i]
                y_pred = model(x, distance_test, shortpath_test, quater_test, movement_test).squeeze(dim=2)
            except:
                break

            count_temp = count_temp + 1
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(args, model, n_features, data_iter, scaler, test_num, batch_size, n_vertex, count_temp, device):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []

        for x, y in data_iter:
            if test_num > batch_size:
                y_tru = np.zeros((batch_size, n_features))
                y_pre = np.zeros((batch_size, n_vertex, n_features))
                test_num = test_num - batch_size
            else:
                y_tru = np.zeros((test_num, n_vertex, n_features))
                y_pre = np.zeros((test_num, n_vertex, n_features))

            distance_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
            shortpath_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
            quater_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 4), device=device)
            movement_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 3), device=device)

            try:
                for i in range(args.batch_size):
                    distance_test[i] = args.distance[count_temp + i: count_temp + args.n_his + i]
                    shortpath_test[i] = args.shortpath[count_temp + i: count_temp + args.n_his + i]
                    quater_test[i] = args.quater[count_temp + i: count_temp + args.n_his + i]
                    movement_test[i] = args.movement[count_temp + i: count_temp + args.n_his + i]
                y_pred = model(x, distance_test, shortpath_test, quater_test, movement_test).squeeze(dim=2)
            except:
                break

            count_temp = count_temp + 1

            y_tru_num, y_tru_features, y_tru_nodes = y.shape
            y_pre_num, y_pre_features, y_pre_nodes = y_pred.shape
            y_tru = y.cpu().numpy().reshape(-1).reshape(y_tru_num, y_tru_nodes, y_tru_features)
            y_pre = y_pred.cpu().numpy().reshape(-1).reshape(y_pre_num, y_pre_nodes, y_pre_features)
            y_tru = scaler.inverse_transform(y_tru.reshape(-1, n_features)).reshape(y_tru.shape)
            y_pre = scaler.inverse_transform(y_pre.reshape(-1, n_features)).reshape(y_pre.shape)

            y_tru = y_tru.reshape(-1)
            y_pre = y_pre.reshape(-1)

            d = np.abs(y_tru - y_pre)
            mae += d.tolist()
            sum_y += y_tru.tolist()



            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        return MAE, RMSE, WMAPE
