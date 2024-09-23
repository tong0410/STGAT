import torch
import tqdm
from sklearn import preprocessing

def train(n_vertex, loss, args, optimizer, scheduler, es, model, train_iter, val_iter, device):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0
        count_train = 0  # use 'count' to find which of edges
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            distance_train = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
            shortpath_train = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
            quater_train = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 4), device=device)
            movement_train = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 3), device=device)

            try:
                for i in range(args.batch_size):
                    distance_train[i] = args.distance[count_train + i: count_train + args.n_his + i]
                    shortpath_train[i] = args.shortpath[count_train + i: count_train + args.n_his + i]
                    quater_train[i] = args.quater[count_train + i: count_train + args.n_his + i]
                    movement_train[i] = args.movement[count_train + i: count_train + args.n_his + i]

                y_pred = model(x, distance_train, shortpath_train, quater_train, movement_train).squeeze(dim=2)
            except:
                break

            count_train = count_train + 1
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(n_vertex, loss, model, val_iter, args, device)

        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        if es.step(val_loss):
            print('Early stopping.')
            break

@torch.no_grad()
def val(n_vertex, loss, model, val_iter, args, device):
    model.eval()
    l_sum, n = 0.0, 0
    count_val = args.len_train
    for x, y in val_iter:
        distance_val = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
        shortpath_val = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex), device=device)
        quater_val = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 4), device=device)
        movement_val = torch.zeros((args.batch_size, args.n_his, n_vertex, n_vertex, 3), device=device)

        try:
            for i in range(args.batch_size):
                distance_val[i] = args.distance[count_val + i: count_val + args.n_his + i]
                shortpath_val[i] = args.shortpath[count_val + i: count_val + args.n_his + i]
                quater_val[i] = args.quater[count_val + i: count_val + args.n_his + i]
                movement_val[i] = args.movement[count_val + i: count_val + args.n_his + i]

            y_pred = model(x, distance_val, shortpath_val, quater_val, movement_val).squeeze(dim=2)
        except:
            break

        count_val = count_val + 1
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)
