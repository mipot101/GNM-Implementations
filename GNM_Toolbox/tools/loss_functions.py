import torch
import torch.nn.functional as F

def inverse_weighted_categorial_crossentropy_loss(pi, reduction='sum'):
    '''
    Returns the inverse weighted categorial crossentropy loss function.
    The returned object only accepts parameters of the same size
    as the vector pi.
    Args:
        pi (tensor): weighting vector
        reduction (string, optional): Either 'sum' (default) or 'mean'
    '''
    length = pi.shape[0]

    def loss_mean(out, target):
        tmp = torch.nn.functional.cross_entropy(out, target, reduction='none')
        return torch.mean((1./pi).view((length)) * tmp.view((length)))


    def loss_sum(out, target):
        tmp = torch.nn.functional.cross_entropy(out, target, reduction='none')
        return torch.sum((1/pi).view((length)) * tmp.view((length)))

    if reduction == 'mean':
        return loss_mean
    else:
        return loss_sum

def inverse_weighted_mean_squared_error(weight, reduction='sum'):
    '''
    Returns the inverse weighted mean squared error loss function.
    The returned object only accepts parameters of the same size
    as the vector pi.
    Args:
        pi (tensor): weighting vector
        reduction (string, optional): Either 'sum' (default) or 'mean'
    '''
    length = weight.shape[0]

    def loss_sum(out, target):
        assert (out.shape in [(length,), (length, 1), (1, length)]), "out should be a {}-dim vector but has shape {}".format(length, out.shape)
        assert (target.shape in [(length,), (length, 1), (1, length)]), "target should be a {}-dim vector but has shape {}".format(length, out.shape)
        return torch.sum((1/weight).view(length) * (out.view(length) - target.view(length)) ** 2, dim=0)

    def loss_mean(out, target):
        assert (out.shape in [(length,), (length, 1), (1, length)]), "out should be a {}-dim vector but has shape {}".format(length, out.shape)
        assert (target.shape in [(length,), (length, 1), (1, length)]), "target should be a {}-dim vector but has shape {}".format(length, out.shape)
        return torch.mean((1/weight).view(length) * (out.view(length) - target.view(length)) ** 2, dim=0)
        
    if reduction == 'mean':
        return loss_mean
    else:
        return loss_sum

def nll(x):
    return - torch.mean(torch.log(x))

def l2(piyi, piyib):
    """
    Args:
        piyi (1-dim Tensor): pi(y_i), für i mit r_i = 1
        piyib (2-dim Tensor (N_known x B)): pi(y_ib) für i mit r_i = 0
    """
    return torch.sum(-torch.log(piyi)) + torch.sum(-torch.log(1-piyib))

def evaluate_loss2_sampling(x, y, mask, modelR, YIB, num_classes, B):
    '''
    Evaluates the loss 2.
    The expectation in the second sum is calculated with samples.
    '''
    N = sum(mask.logical_not())

    # Berechne pi(y_i, h(x_i)) für bekannte i
    y0 = y[mask].view(sum(mask), )
    y_one_hot = F.one_hot(y0.type(torch.int64), num_classes=num_classes)
    out1 = modelR(x[mask], y_one_hot.type(x.dtype)) # output für bekannte i; TODO: Check typechange this is ok
    
    # Berechne expectation of pi(y, h(x_i)) für unbekannte i
    out2 = torch.zeros((N,B))
    for b in range(B):
        yb = YIB[:,b]
        yb = yb.view(yb.shape[0],)
        yb = F.one_hot(yb.type(torch.int64), num_classes)
        yb = yb.type(x.dtype) # TODO: Check if this is ok
        out2[:, b] = modelR(x[mask.logical_not()], yb).view(N)

    if len(out2.shape) == 1:
        out2 = out2.view((len(out2), 1))

    return l2(out1, out2)

def evaluate_loss2_exact(x, y, mask, modelR, y_dist, num_classes):
    '''
    Evaluates the loss 2.
    The expectation in the second sum is calculated exactly.
    '''
    N = sum(mask.logical_not())

    # Berechne pi(y_i, h(x_i)) für bekannte i
    y0 = y[mask].view(sum(mask), )
    y_one_hot = F.one_hot(y0.type(torch.int64), num_classes=num_classes)
    out1 = modelR(x[mask], y_one_hot.type(x.dtype)) # output für bekannte i; TODO: Check typechange this is ok
    
    # Berechne expectation of pi(y, h(x_i)) für unbekannte i
    yh = torch.zeros((N, num_classes))
    for k in range(num_classes):
        yb = torch.ones((N,)) * k
        yb = F.one_hot(yb.type(torch.int64), num_classes)
        yb = yb.type(x.dtype) # TODO: Check if this is ok
        yh[:, k] = modelR(x[mask.logical_not()], yb).view(N)
    out2 = torch.sum(yh * y_dist[mask.logical_not()], dim=1)

    if len(out2.shape) == 1:
        out2 = out2.view((len(out2), 1))

    return l2(out1, out2)

def evaluate_loss2_fast(x, y_est, mask, modelR, num_classes):
    '''
    Evaluates the loss 2.
    The expectation in the second sum is calculated with the expectation of y.
    '''
    pi_est = modelR(x, F.one_hot(y_est, num_classes).type(x.dtype))
    return l2(pi_est[mask], pi_est[mask.logical_not()])
