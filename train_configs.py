import torch


def train_runner(model, trainloader, criterion, optimizer):
    model.train()
    for i, (cx, cy, tx, ty) in enumerate(trainloader):
        cx = torch.squeeze(cx, dim=0)  # (bs, n_context, x_size)
        cy = torch.squeeze(cy, dim=0)  # (bs, n_context)
        tx = torch.squeeze(tx, dim=0)  # (bs, n_target, x_size)
        ty = torch.squeeze(ty, dim=0)  # (bs, n_target)

        c_id = cy[..., 0]
        t_id = ty[..., 0]
        cy = cy[..., 1]  # (bs, n_context)
        ty = ty[..., 1]

        cy = cy.unsqueeze(dim=-1)
        mu, sigma = model(cx, cy, tx)  # (bs, n_target), (bs, n_target)

        loss = criterion(mu, sigma, ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    target_y = ty[0]
    mean_y = mu[0]
    var_y = sigma[0]
    target_id = t_id[0]
    context_id = c_id[0]

    index = target_id.argsort()
    target_id = target_id[index]
    target_y = target_y[index]
    mean_y = mean_y[index]
    var_y = var_y[index]

    train_mse = (torch.sum((target_y - mean_y) ** 2)) / len(target_y)

    return mean_y, var_y, target_id, target_y, context_id, loss.cpu().detach().numpy(), train_mse


def val_runner(model, testloader, criterion):
    model.eval()
    with torch.no_grad():
        for val_cx, val_cy, val_tx, val_ty in testloader:
            val_cx = torch.squeeze(val_cx, dim=0)  # (bs, n_context, x_size)
            val_cy = torch.squeeze(val_cy, dim=0)  # (bs, n_context)
            val_tx = torch.squeeze(val_tx, dim=0)  # (bs, n_target, x_size)
            val_ty = torch.squeeze(val_ty, dim=0)  # (bs, n_target)

            val_t_id = val_ty[..., 0]
            val_cy = val_cy[..., 1]
            val_ty = val_ty[..., 1]
            val_cy = val_cy.unsqueeze(dim=-1)

            val_pred_y, val_sigma_y = model(val_cx, val_cy, val_tx)  # Test
            val_loss = criterion(val_pred_y, val_sigma_y, val_ty)
            val_target_y = val_ty[0]
            val_pred_y = val_pred_y[0]
            val_var_y = val_sigma_y[0]
            val_target_id = val_t_id[0]

    val_index = val_target_id.argsort()
    val_target_id = val_target_id[val_index]
    val_target_y = val_target_y[val_index]
    val_pred_y = val_pred_y[val_index]
    val_var_y = val_var_y[val_index]


    valid_mse = (torch.sum((val_target_y - val_pred_y) ** 2)) / len(val_target_y)
    return val_pred_y, val_var_y, val_target_id, val_target_y, val_loss.cpu().detach().numpy(), valid_mse
