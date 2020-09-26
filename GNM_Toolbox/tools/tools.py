from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import time
import copy
import sys

'''
This file consists mainly of methods to assist gnm.py
'''

def my_log(string, ebene=0, maxpriority=0, priority=0):
    if priority < maxpriority:
        print("  "*ebene+string)

def prepare_modely(nety, num_node_features, num_classes, classification):
    modely = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if classification:
        modely = nety().to(device) if nety is not None else twoLayerConvolutionalNetwork(num_node_features, num_classes).to(device)
    else: 
        modely = nety().to(device) if nety is not None else twoLayerConvolutionalNetwork(num_node_features, 1, activation=lambda x: x).to(device)
    return modely

def prepare_modelr(x, h_out_features, neth, K):
    if neth is None:
        modelh = DefaultNetH(x.shape[1])
        h_out_features = 1
    else:
        modelh = neth()
    return NetR(modelh, h_out_features, K)

def estimateY(modely, data, classification, y_est, train_mask, N):
    y = data.y
    modely.eval()
    p = modely(data)
    y_est_old = y_est.clone()
    y_est = sample_from_distributions(p.detach().numpy(), 1) if classification else p.view((N, 1))
    tmp = y.type(y_est.dtype)[train_mask]
    y_est[train_mask] = tmp.view((tmp.shape[0], 1))
    return y_est, y_est_old

def eval_model_with_logging(model, data, logging, mask=None):
    if mask == None:
        mask = data.test_mask
    
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[mask].eq(data.y[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc

def eval_model(modelout, data, mask=None, loss_f=None, classification=True):
    if mask == None:
        mask = data.test_mask
    
    acc = 0
    if classification:
        _, pred = modelout.max(dim=1) # torch.max returned tuple with (values, indices)
        correct = int(pred[mask].eq(data.y[mask]).sum().item())
        acc = correct / int(mask.sum())
    else:
        # In this case the accuracy is not well defined. Thus we use the coefficient of determination (https://en.wikipedia.org/wiki/Coefficient_of_determination)
        mean = torch.mean(modelout)
        SS_tot = torch.sum((data.y-mean)**2)
        SS_res = torch.sum((modelout-mean)**2)
        acc = 1 - SS_res/SS_tot
    if loss_f is not None:
        l = loss_f(modelout[mask], data.y[mask])
        return acc, l
    else:
        return acc

def train_model(model, data, train_mask=None, loss_function=None, NB_IT=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    if loss_function is None:
        loss_function = F.nll_loss
    if train_mask is None:
        train_mask = data.train_mask

    model.train()
    for _ in range(NB_IT):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
    return model

def normalize(p):
    """
    Args:
        p (2 dim-array):
    """
    for i in range(len(p)):
        p[i] = p[i]/sum(p[i])
        p[i] = np.floor(p[i] * 10**5)/10**5 # Sehr sehr unschön. TODO: Was besseres finden

def sample_from_distributions(p, B):
    """
    Args:
        p (numpy array): matrix of shape [N, K] (N number of nodes, K number of classes) with probability distribution for each n in N
        B (int): Number of samples for each distribution
        returns [N, B] Matrix with samples 
    """
    N, K = p.shape
    normalize(p) # Noch einmal sichergehen, dass alle Zeilen sich zu knapp unter 1 aufaddieren
    samples = [np.random.multinomial(B, p[i]) for i in range(len(p))] # y_{ib} NxK Matrix bei K klassen TODO: Scheinbar klappt das gelegentlich immer noch nicht
    
    # Wandle samples von [N, K] Matrix zu [N, B] Matrix um (Laufzeit: O(N*K))
    YIB = torch.tensor([])
    for i in range(N):
        samp = torch.zeros((1, 0))
        for k in range(0, K):
            var = samples[i][k]
            samp = torch.cat((samp, k*torch.ones(var).view((1, var))), 1)
        if i == 0:
            YIB = samp
        else:
            YIB = torch.cat((YIB, samp), 0)
    return YIB

def evaluateR(modelR, pi, data):
    est_out = modelR(data.x, data.y.view(data.x.shape[0], 1).type(data.x.dtype))
    out = pi(data.x, data.y)
    return torch.nn.MSELoss()(est_out.view(est_out.shape[0]), out)

def count(array, k):
    a = 0
    for i in range(len(array)):
        if array[i] == k:
            a += 1
    return a

def choose(array, choosen, klasse):
    count = 0
    result = np.zeros_like(array)
    for i in range(len(array)):
        if array[i] == klasse:
            if count in choosen:
                result[i] = 1
            count += 1
    return result == 1

def time_to_string(i):
    # float i
    hours = int(i/3600)
    i -= hours * 3600
    minutes = int(i/60)
    i -= minutes * 60
    seconds = int(i)
    if hours > 0:
        return '{}h {}min {}sec'.format(hours, minutes, seconds)
    elif minutes > 0:
        return '{}min {}sec'.format(minutes, seconds)
    else:
        return '{:.2f}sec'.format(i)

def print_status(i, N, starttime = None):
    # Es wird angenommen, dass i von 0 bis N-1 läuft
    # Länge Ladebalken = 20
    l = 20
    done = int(i/(N-1) * l)
    counterstring = '({}/{})'.format(i+1, N)
    barstring = '|'+u'\u2588'*done + ' '*(l-done)+'|'
    if starttime is None:
        print('{} {}'.format(counterstring, barstring), end='\r')
    else:
        t_1 = time.time()
        time_spent = t_1 - starttime
        iterations_done = i
        iterations_to_go = N-i
        if iterations_done == 0:
            time_per_iteration = 0
        else:
            time_per_iteration = time_spent/iterations_done
        time_to_go = time_per_iteration * iterations_to_go
        
        time_string = '({}|{}|{})'.format(time_to_string(time_spent), time_to_string(time_to_go), time_to_string(time_per_iteration))
        
        print('{} {} {}                                 '.format(counterstring, barstring, time_string), end='\r')