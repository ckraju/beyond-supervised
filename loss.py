import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

def cross_entropy2d(input, target, weights=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim = 1)
    loss = F.nll_loss(log_p, Variable(target.cuda()), weight=weights, size_average=size_average)

    return loss

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)



def performMetrics(epoch,iter_idx,iter_,hist,loss,is_train,write = False):
    
    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    mean_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * mean_iou[freq > 0]).sum()
    return 100*np.nanmean(mean_iou), 100*pixel_accuracy, 100*fwavacc

