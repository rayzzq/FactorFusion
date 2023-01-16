import numpy as np
import torch
from torch import nn


def hamming_loss(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        # print('\nset_true: {0}'.format(set_true))
        # print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) /\
                float(len(set_true.union(set_pred)))
        # print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def huber_pinball_loss(input_data, target_data, alpha=0.2, quantile_levels=None):
    error_data = target_data.contiguous().reshape(-1, 1) - input_data
    if alpha == 0.0:
        loss_data = torch.max(quantile_levels * error_data, (quantile_levels - 1) * error_data)
        return loss_data.mean()

    loss_data = torch.where(torch.abs(error_data) < alpha,
                            0.5 * error_data * error_data,
                            alpha * (torch.abs(error_data) - 0.5 * alpha))
    loss_data /= alpha
    scale = torch.where(error_data >= 0,
                        torch.ones_like(error_data) * quantile_levels,
                        torch.ones_like(error_data) * (1 - quantile_levels))
    loss_data *= scale
    return loss_data.mean()


def margin_loss(input_data, margin_scale=0.0001, quantile_levels=None):
    # number of samples
    batch_size, num_quantiles = input_data.size()

    # compute margin loss (batch_size x num_net_outputs(above) x num_net_outputs(below))
    error_data = input_data.unsqueeze(1) - input_data.unsqueeze(2)

    # margin data (num_quantiles x num_quantiles)
    margin_data = quantile_levels.permute(1, 0) - quantile_levels
    margin_data = torch.tril(margin_data, -1) * margin_scale

    # compute accumulated margin
    loss_data = torch.tril(error_data + margin_data, diagonal=-1)
    loss_data = loss_data.relu()
    loss_data = loss_data.sum() / float(batch_size * (num_quantiles * num_quantiles - num_quantiles) * 0.5)
    return loss_data


def quantile_loss(predict_data, target_data, margin):
    if margin > 0.0:
        m_loss = margin_loss(predict_data)
    else:
        m_loss = 0.0
    h_loss = huber_pinball_loss(predict_data, target_data).mean()
    return h_loss + margin * m_loss


LOSS = {
    "BCELL": nn.BCEWithLogitsLoss(),
    "L1": nn.L1Loss(),
    "L2": nn.MSELoss(),
    "SL1": nn.SmoothL1Loss(),
    "huber_pinball_loss": huber_pinball_loss,
    "margin_loss": margin_loss,
    "quantile_loss": quantile_loss
}
