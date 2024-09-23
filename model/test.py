import os

from script import utility
import torch
import numpy as np

@torch.no_grad()
def test(zscore, loss, model, test_iter, args, test_num, n_features, n_vertex, device):
    model.eval()
    count_test = args.len_train + args.len_val
    test_MSE = utility.evaluate_model(args, model, loss, test_iter, count_test, n_vertex, device)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(args, model, args.n_features, test_iter, zscore, test_num, args.batch_size, n_vertex, count_test, device)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    folder_path = './%s/test/'% (args.dataset)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the final model
    torch.save(model.state_dict(), './%s/test/final_model_%d_%d_%d.pth'% (args.dataset, args.n_his, args.n_pred, args.stblock_num))

    # Save each prediction and true value
    predictions = []
    true_values = []
    for x, y in test_iter:
        distance_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
        shortpath_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
        quater_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 4), device=device)
        movement_test = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 3), device=device)

        try:
            for i in range(args.batch_size):
                distance_test[i] = args.distance[count_test: count_test + args.n_his]
                shortpath_test[i] = args.shortpath[count_test: count_test + args.n_his]
                quater_test[i] = args.quater[count_test: count_test + args.n_his]
                movement_test[i] = args.movement[count_test: count_test + args.n_his]
            y_pred = model(x, distance_test, shortpath_test, quater_test, movement_test).squeeze(dim=2)
            count_test = count_test + 1
        except:
            break
        predictions.append(y_pred.cpu().numpy())
        true_values.append(y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)


    predictions = np.transpose(predictions, (0, 2, 1))
    true_values = np.transpose(true_values, (0, 2, 1))

    # precessing now
    predictions = zscore.inverse_transform(predictions.reshape(-1, n_features)).reshape(predictions.shape)
    true_values = zscore.inverse_transform(true_values.reshape(-1, n_features)).reshape(true_values.shape)

    np.save('./%s/test/predictions_%d_%d_%d.npy'% (args.dataset, args.n_his, args.n_pred, args.stblock_num), predictions)
    np.save('./%s/test/true_values_%d_%d_%d.npy' % (args.dataset, args.n_his, args.n_pred, args.stblock_num), true_values)

    print('Final model and predictions saved.')