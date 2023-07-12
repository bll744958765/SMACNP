import torchvision
# -*- coding : utf-8 -*-
import pandas as pd
from dataloader_smanp import DatasetGP_test, data_load
from model_smanp import SpatialNeuralProcess, Criterion
import torch as torch
from train_configs import val_runner
from torch.utils.data import DataLoader
import numpy as np
import time

start = time.perf_counter()
time.sleep(2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("validation set effect---------------------------------")
n_tasks = 1
batch_size = 1
x_size = 6
y_size = 1
z_size = 128
lr = 0.001
num_hidden = 128
list1 = []
list = []
model = SpatialNeuralProcess(x_size=x_size, y_size=y_size, num_hidden=num_hidden)
model = model.to(device)
for xuhao in range(10):
    dataset = DatasetGP_test(n_tasks=n_tasks, xuhao=xuhao, batch_size=batch_size)
    testloader = DataLoader(dataset, batch_size=1, shuffle=False)

    state_dict = torch.load('./checkpoint_smanp/checkpoint_{}.pth.tar'.format(xuhao))
    model.load_state_dict(state_dict=state_dict['model'])
    model.eval()
    criterion = Criterion()

    val_pred_y, val_var_y, val_target_id, val_target_y, val_loss, valid_mse = val_runner(model, testloader, criterion)

    val_target_y = val_target_y.cpu().detach().numpy()
    val_pred_y = val_pred_y.cpu().detach().numpy()
    val_var_y = val_var_y.cpu().detach().numpy()
    val_target_id = val_target_id.cpu().detach().numpy()
    valid_r2 = 1 - ((np.sum((val_target_y - val_pred_y) ** 2)) / np.sum((val_target_y - val_target_y.mean()) ** 2))
    valid_mse = (np.sum((val_target_y - val_pred_y) ** 2)) / len(val_target_y)
    valid_rmse = np.sqrt(valid_mse)
    valid_mae = (np.sum(np.absolute(val_target_y - val_pred_y))) / len(val_target_y)
    corr = np.corrcoef(val_target_y, val_pred_y)
    C = (2 * corr[0, 1] * np.std(val_pred_y) * np.std(val_target_y)) / (np.var(val_target_y) + np.var(val_pred_y) + (val_target_y.mean() - val_pred_y.mean()) ** 2)

    corr_t_p=np.corrcoef(val_target_y, val_pred_y)
    corr_t_v=np.corrcoef(val_target_y, val_var_y)
    corr_p_v = np.corrcoef(val_pred_y, val_var_y)



    prediction = pd.DataFrame(
        {"id": np.array(val_target_id), "true": np.array(val_target_y), "pred": np.array(val_pred_y),
         "cha": np.array(val_target_y) - np.array(val_pred_y), 'var_y': np.array(val_var_y)})
    prediction.to_csv('./prediction/prediction_val_{}.csv'.format(xuhao), index=False)

    print("ID:", xuhao, "valid_MAE:", round(valid_mae, 4), "valid_MSE:", round(valid_mse, 4), " valid_RMSE:", round(valid_rmse, 4),
          " valid_R-square:", round(valid_r2.item(), 4), "CCC:", round(C, 4), "average_var:", round(np.mean(val_var_y), 4),
          'corr_t_p:',round(corr_t_p[0,1], 4),"corr_t_v:",round(corr_t_v[0,1], 4),'corr_p_v:',round(corr_p_v[0,1], 4))

    list = [round(valid_mae, 4), round(valid_mse, 4), round(valid_rmse, 4), round(valid_r2.item(), 4), round(C, 4), round(np.mean(val_var_y), 4),
            round(corr_t_p[0,1], 4),round(corr_t_v[0,1], 4),round(corr_p_v[0,1], 4)]
    list1.append(list)


print(np.mean(list1, axis=0))
print(np.std(list1, axis=0))
end = time.perf_counter()
print(str(end - start))