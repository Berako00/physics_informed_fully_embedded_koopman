import torch
import torch.nn.functional as F

def custom_loss(x_pred, x_target):
    total_custom_loss = torch.sum(torch.mean((x_pred - x_target) ** 2))
    return total_custom_loss

def loss_4(xuk, Num_meas, model):
    pred_4 = model.x_Koopman_op(model.x_Encoder(xuk[:, 0, :Num_meas])) + model.u_Koopman_op(model.u_Encoder(xuk[:, 0, :]))
    L_4 = F.mse_loss(pred_4, model.x_Encoder(xuk[:, 1, :Num_meas]), reduction='mean')
    return L_4, pred_4

def loss_6(xuk, Num_meas, Num_x_Obsv, T, L_4, pred_4, model):
    u = xuk[:, :, Num_meas:]
    total_6_loss = L_4
    y_k = pred_4
    x_k = pred_4[:,:Num_meas]

    for m in range(1, T-1):
        v = model.u_Encoder(torch.cat((x_k, u[:, m, :]), dim=1))
        y_k = model.x_Koopman_op(y_k) + model.u_Koopman_op(v)
        total_6_loss += F.mse_loss(y_k, model.x_Encoder(xuk[:, m+1, :Num_meas]), reduction='mean')
        x_k = y_k[:,:Num_meas]

    L_6 = total_6_loss / T
    return L_6

def total_loss(alpha, xuk, Num_meas, Num_x_Obsv, T, S_p, model):
    [L_4, pred_4]  = loss_4(xuk, Num_meas, model)
    L_6 = loss_6(xuk, Num_meas, Num_x_Obsv, T, L_4, pred_4, model)

    L_total = alpha[1]*L_4 + alpha[2]*L_6

    return L_total, L_4, L_6


def loss_4_uf(xuk, Num_meas, model):
    pred_4 = model.x_Koopman_op(model.x_Encoder(xuk[:, 0, :Num_meas]))
    L_4 = F.mse_loss(pred_4, model.x_Encoder(xuk[:, 1, :Num_meas]), reduction='mean')
    return L_4, pred_4


def loss_6_uf(xuk, Num_meas, Num_x_Obsv, T, L_4, pred_4, model):
    total_6_loss = L_4
    y_k = pred_4
    x_k = pred_4[:,:Num_meas]

    for m in range(1, T-1):
        y_k = model.x_Koopman_op(y_k)
        total_6_loss += F.mse_loss(y_k, model.x_Encoder(xuk[:, m+1, :Num_meas]), reduction='mean')

    L_6 = total_6_loss / T
    return L_6

def total_loss_unforced(alpha, xuk, Num_meas, Num_x_Obsv, T, S_p, model):

    [L_4, pred_4]  = loss_4_uf(xuk, Num_meas, model)
    L_6 = loss_6_uf(xuk, Num_meas, Num_x_Obsv, T, L_4, pred_4, model)

    L_total = alpha[0]*(L_gx) +  alpha[1]*(L_3 + L_4)+ alpha[2]*(L_5 + L_6)

    return L_total, L_4, L_6

def total_loss_forced(alpha, xuk, Num_meas, Num_x_Obsv, T, S_p, model):
    [L_4, pred_4]  = loss_4(xuk, Num_meas, model)
    L_6 = loss_6(xuk, Num_meas, Num_x_Obsv, T, L_4, pred_4, model)

    L_total = alpha[0]* L_gu +  alpha[1]*L_4 + alpha[2]*L_6

    return L_total, L_4, L_6
