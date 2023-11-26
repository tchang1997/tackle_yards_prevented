import torch
import torch.nn.functional as F

def nednet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, **kwargs):  # same func signature for backward compat
    loss_record = {}
    t_pred = (t_pred + 0.01) / 1.02
    loss_record["loss_t"] = torch.sum(F.binary_cross_entropy(t_pred, t_true, reduction='none'))
    loss_record["loss_total"] = loss_record["loss_t"]
    return loss_record

def dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0):
    loss_record = {}
    t_pred = (t_pred + 0.01) / 1.02
    loss_record["loss_t"] = torch.sum(F.binary_cross_entropy(t_pred, t_true, reduction='none'))

    loss_record["loss_y0"] = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss_record["loss_y1"] = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss_record["loss_y_overall"] = loss_record["loss_y0"] + loss_record["loss_y1"]

    loss_record["loss_total"] = loss_record["loss_y_overall"] + alpha * loss_record["loss_t"]

    return loss_record


def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0, beta=1.0):
    """
    Targeted regularisation loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    beta: float
        targeted regularization hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    loss_record = dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=alpha)
    t_pred = (t_pred + 0.01) / 1.02

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))

    y_pert = y_pred + eps * h
    targeted_regularization = torch.sum((y_true - y_pert)**2)

    # final
    loss_record["loss_total_tarreg"] = loss_record["loss_total"] + beta * targeted_regularization
    loss_record["tarreg"] = targeted_regularization
    return loss_record
