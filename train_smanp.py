# -*- coding : utf-8 -*-

from dataloader_smanp import DatasetGP, DatasetGP_test
from model_smanp import SpatialNeuralProcess, Criterion
import torch as torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from train_configs import train_runner, val_runner
from math import sqrt
import random
import time




start = time.perf_counter()
time.sleep(2)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



def main():
    set_seed(42)
    n_epoches = 300
    n_tasks =30
    batch_size = 1
    x_size = 6
    y_size = 1
    start_lr = 0.001
    num_hidden = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for xuhao in range(10):
        trainset = DatasetGP(n_tasks=n_tasks, xuhao=xuhao, batch_size=batch_size)
        testset = DatasetGP_test(n_tasks=n_tasks, xuhao=xuhao, batch_size=batch_size)
        model = SpatialNeuralProcess(x_size=x_size, y_size=y_size, num_hidden=num_hidden).to(device)
        criterion = Criterion()
        optimizer = optim.Adam(model.parameters(), lr=start_lr)
        model.train()
        lo=2.0

        for epoch in range(n_epoches):
            trainloader = DataLoader(trainset, shuffle=True)
            testloader = DataLoader(testset, shuffle=True)

            mean_y, var_y, target_id, target_y, context_id, loss, train_mse = train_runner(
                model, trainloader, criterion, optimizer)

            val_pred_y, val_var_y, val_target_id, val_target_y, val_loss, valid_mse = val_runner(
                model, testloader, criterion)

            if lo >= val_loss:
                lo = val_loss
                train_mse = (torch.sum((target_y - mean_y) ** 2)) / len(target_y)
                train_rmse = sqrt(train_mse)
                train_mae = (torch.sum(torch.absolute(target_y - mean_y))) / len(target_y)
                train_r2 = 1 - ((torch.sum((target_y - mean_y) ** 2)) / torch.sum((target_y - target_y.mean()) ** 2))
                valid_mse = (torch.sum((val_target_y - val_pred_y) ** 2)) / len(val_target_y)
                valid_rmse = sqrt(valid_mse)
                valid_mae = (torch.sum(torch.absolute(val_target_y - val_pred_y))) / len(val_target_y)

                valid_r2 = 1 - ((torch.sum((val_target_y - val_pred_y) ** 2)) / torch.sum((val_target_y - val_target_y.mean()) ** 2))
                print(
                    "ID: {} \t Train Epoch: {} \t Lr:{:.4f},train loss: {:.4f},train_mae: {:.4f},train_mse: {:.4f}, train_rmse: {:.4f},train_r2: {:.4f}".format(
                        xuhao, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], loss,     train_mae, train_mse, train_rmse, train_r2))
                print(
                    "ID: {} \t Valid Epoch: {} \t Lr:{:.4f},valid loss: {:.4f},valid_mae: {:.4f},valid_mse: {:.4f}, valid_rmse: {:.4f},valid_r2: {:.4f}".format(
                        xuhao, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], val_loss, valid_mae, valid_mse, valid_rmse, valid_r2))
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join('checkpoint_smanp', 'checkpoint_%d.pth.tar' % (xuhao)))
            # if epoch % 30 == 0:
            #     print(
            #         "ID: {} \t Train Epoch: {} \t Lr:{:.4f},train loss: {:.4f},train_mse: {:.4f}".format(
            #             xuhao, epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss, train_mse))
            #     print(
            #         "ID: {} \t Valid Epoch: {} \t Lr:{:.4f},valid loss: {:.4f},valid_mse: {:.4f}".format(
            #             xuhao, epoch, optimizer.state_dict()['param_groups'][0]['lr'], val_loss, valid_mse))

        prediction = pd.DataFrame(
            {"id_val": val_target_id.detach().cpu().numpy(), "valid": val_target_y.detach().cpu().numpy(), "val_pred": val_pred_y.detach().cpu().numpy(),
             "cha": val_target_y.detach().cpu().numpy() - val_pred_y.detach().cpu().numpy(), 'val_var_y': val_var_y.detach().cpu().numpy(), })
        prediction.to_csv('./prediction/prediction_{}.csv'.format(xuhao), index=False)

    end = time.perf_counter()
    print (str(end-start))
if __name__ == "__main__":
    main()
