import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import sys
import copy
import time

from .tools.tools import *
from .tools.networks import *
from .tools.loss_functions import *

def train_modelr(x, y, num_classes, known_mask, y_dist, NB_IT=100, B=1, 
    PATIENCE=15, logging=0, modelR = None, classification=True, 
    expectation='fast'):
    """
    Trains the model to estimate pi
    Args:
        x (tensor): x data to train on
        y (tensor): y data to train on
        num_classes (int): number of classes
        known_mask (boolean array): An element is True if the corresponding 
            node label is known
        y_dist (tensor): In case of classification NxK matrix which describes 
            the current estimate of the distribution of y. In case of 
            regression current estimate of y.
        NB_IT (int, optional): Maximum number of trainings iterations. 
            Default: 100
        B (int, optional): If expectation == 'sample' number of samples to take.
            Otherwise irrelevant. Default: 1
        PATIENCE (int, optional): Patience. If the validation loss doesnt 
            decrease for PATIENCE many training iterations training is 
            stopped. Default is 15
        logging (int, optional):  Describes the layer of logging that shall be 
            printed. Default is 0, which means nothing is printed
        modelR (torch.nn.module, optional): if no model is specified a default 
            multilayer perceptron will be created. Otherwise the potencial 
            pretrained model will be trained.
        classification (bool, optional): Tells whether the problem is a 
            classification or regression problem
        expectation (string, optional): Options are 'fast', 'exact' and 'sample'
            Decides which method to use for calculating the loss.
    """

    # Create model and its optimizer
    if modelR is None:
        modelR = prepare_modelr(x, 0, None, num_classes)
    optimizerR = torch.optim.Adam(modelR.parameters(), lr=0.01, weight_decay=5e-4)

    best_eval_loss = 999999
    wait = 0
    loss_list = list()
    YIB, y_est, loss, eval_loss = None, None, None, None
    if expectation == 'sample':
        YIB = sample_from_distributions(y_dist.detach().numpy(), B) if classification else y_dist.view((N, 1))
    if expectation == 'fast':
        y_est = torch.argmax(y_dist, dim=1) if classification else y_dist.view((N, 1)) # Get an estimation of y
        y_est[known_mask] = y[known_mask] # Set trainings data perfect

    # Train modelR
    for i in range(NB_IT):
        modelR.train()
        optimizerR.zero_grad()
        
        if expectation == 'sample':
            loss = evaluate_loss2_sampling(x, y, known_mask, modelR, YIB, num_classes, B)
        elif expectation == 'fast':
            loss = evaluate_loss2_fast(x, y_est, known_mask, modelR, num_classes)
        else:
            loss = evaluate_loss2_exact(x, y, known_mask, modelR, y_dist, num_classes)
        
        loss.backward(retain_graph=True)
        optimizerR.step()

        modelR.eval()
        if expectation == 'sample':
            eval_loss = evaluate_loss2_sampling(x, y, known_mask, modelR, YIB, num_classes, B)
        elif expectation == 'fast':
            eval_loss = evaluate_loss2_fast(x, y_est, known_mask, modelR, num_classes)
        else:
            eval_loss = evaluate_loss2_exact(x, y, known_mask, modelR, y_dist, num_classes)

        loss_list.append(eval_loss.detach().numpy())

        # Early Stopping
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            wait = 0
        else:
            if wait >= PATIENCE:
                my_log('Early Stopping in iteration {} while training modelr.'.format(i), 1, logging)
                break
            wait += 1 
    
    # Update pi
    y_one_hot = F.one_hot(y.type(torch.int64), num_classes)
    pi = modelR(x, y_one_hot.type(x.dtype)) 

    return (pi, loss_list, modelR)

def train_modelr_adapted(x, y, num_classes, known_mask, y_dist, NB_IT_1=30, 
    NB_IT_2=100, B=1, PATIENCE=15, logging=0, modelR = None, 
    classification=True, expectation='fast'):
    """
    Trains the model to estimate pi. Uses method proposed in chapter 6.3
    Args:
        x (tensor): x data to train on
        y (tensor): y data to train on
        num_classes (int): number of classes
        known_mask (boolean array): An element is True if the corresponding 
            node label is known
        y_dist (tensor): In case of classification NxK matrix which describes 
            the current estimate of the distribution of y. In case of 
            regression current estimate of y.
        NB_IT_1 (int, optional): Maximum number of trainings iterations for 
            pre training of model. Default: 30
        NB_IT_2 (int, optional): Maximum number of trainings iterations. 
            Default: 100
        B (int, optional): If expectation == 'sample' number of samples to take.
            Otherwise irrelevant. Default: 1
        PATIENCE (int, optional): Patience. If the validation loss doesnt 
            decrease for PATIENCE many training iterations training is 
            stopped. Default is 15
        logging (int, optional):  Describes the layer of logging that shall be 
            printed. Default is 0, which means nothing is printed
        modelR (torch.nn.module, optional): if no model is specified a default 
            multilayer perceptron will be created. Otherwise the potencial 
            pretrained model will be trained.
        classification (bool, optional): Tells whether the problem is a 
            classification or regression problem
        expectation (string, optional): Options are 'fast', 'exact' and 'sample'
            Decides which method to use for calculating the loss.
    """
    NB_MODELS_TRAINED = 7 # Depending on how much time you want to invest this number may be choosen differently

    # Train 7 models for a short amount of time and choose the best one
    best_modelR = None
    best_loss = 999999999
    for i in range(NB_MODELS_TRAINED):
        # Note that a copy of modelR will be trained, such that the original modelR stays untouched
        pi, loss_list, model = train_modelr(x, y, num_classes, known_mask, y_dist, NB_IT_1, B, PATIENCE, logging, copy.deepcopy(modelR), classification, expectation)
        if loss_list[-1] < best_loss:
            best_loss = loss_list[-1]
            best_modelR = model

    # Train the best of the 7 models for a longer amount of time
    pi, loss_list, modelR = train_modelr(x, y, num_classes, known_mask, y_dist, NB_IT_2, B, PATIENCE, logging, best_modelR, classification, expectation)

    return (pi, loss_list, modelR)

def train_modely(data, modely, pi, train_mask, val_mask, NB_IT, reduction, 
    classification=True, PATIENCE=15, logging=0):
    """
    Trains the model to estimate Y.
    Args:
        data (torch_geometric.data.data.Data): data to train on
        modely (torch.nn.module): model which will be trained
        pi (tensor): current estimation of pi
        train_mask (boolean array): An element is True if the corresponding 
            node label is included in the training set
        val_mask (boolean array): An element is True if the corresponding node 
            label is included in the validation set
        NB_IT (int): Maximum number of trainings iterations
        reduction (string): Options are 'mean' and 'sum'´
        classification (bool, optional): Tells whether the problem is a 
            classification or regression problem
        PATIENCE (int, optional): Patience. If the validation loss doesnt 
            decrease for PATIENCE many training iterations training is 
            stopped. Default is 15
        logging (int, optional):  Describes the layer of logging that shall be 
            printed. Default is 0, which means nothing is printed
    """
    # define loss 1
    l1 = inverse_weighted_categorial_crossentropy_loss(pi[train_mask], reduction=reduction) if classification else inverse_weighted_mean_squared_error(pi[train_mask], reduction=reduction)
    
    # Create optimizer for modely. It is import to be done seperatly, since this resets the learning rates for the parameters and thus later training epochs get a huger impact than otherwise
    optimizerY = torch.optim.Adam(modely.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_loss = 999999
    wait = 0

    # Train modely
    for i in range(NB_IT):
        modely.train()  
        optimizerY.zero_grad()
        out = modely(data)
        loss = l1(out[train_mask], data.y[train_mask])
        loss.backward(retain_graph=True)
        optimizerY.step()

        # Calc validation loss
        loss_f = inverse_weighted_categorial_crossentropy_loss(pi[val_mask], reduction=reduction) if classification else inverse_weighted_mean_squared_error(pi[val_mask], reduction=reduction)
        modely.eval()
        _, val_loss = eval_model(modely(data), data, mask = val_mask, loss_f = loss_f)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            if wait >= PATIENCE:
                my_log('Early Stopping in iteration {} while training modely.'.format(i), 1, logging)
                break
            wait += 1 

def train_net_with_gnm(data, train_mask, val_mask, classification=True,
    nety = None, neth = None, h_out_features = None, pi = None, 
    logging = 0, NB_IT_0 = 200, NB_IT_1 = 40, NB_EPOCHS = 5, PATIENCE = 15,
    epsilon = 0.005, B = 1, reduction='mean'):

    """
    This method is capable of training a neural network to solve a 
    semi-supervised classification or regression problem.
    It uses the algorithm propsed in "Graph-Based Semi-Supervised Learning with 
    Non-ignorable Non-response" written by Fan Zhou, Tengfei Li, Haibo Zhou,
    Hongtu Zhu and Ye Jieping (2019).
    If no other nety is specified the neural network is choosen to be a 
    2-layer GCN as known from "Semi-Supervised Classification with Graph 
    Convolutional Networks" written by Thomas N. Kipf und Max Welling (2016).

    Args:
        data (torch_geometric.data.data.Data): Data to train on; Should have 
            attribute num_node_features and num_classes
        train_mask (boolean array): An element is True if the corresponding 
            node label is included in the training set
        val_mask (boolean array): An element is True if the corresponding node 
            label is included in the validation set
        classification (boolean, optional): True if data.y describes classes, 
            False if data.y has continuous values. Default is True
        nety (module, optional): it is assumed, that the output is a 
            probability distribution, i.e. sum(out[i]) = 1.
            This can be reached by using the softmax layer as last layer. 
            Default: 2-layer GCN
        neth  (module, optional): Should take x_i as input and returns a 
            h_out_features-dim vector. Default: Multilayer Perceptron
        h_out_features (int, optional): dimension of output of neth
        pi (Tensor, optional): Initial pi setting
        logging (int, optional): Describes the layer of logging that shall be 
            printed. Default is 0, which means nothing is printed
        NB_IT_0 (int, optional): Maximum number of training iterations for main 
            model. Default is 200
        NB_IT_1 (int, optional): Maximum number of training iterations for 
            secondary model. Default is 40
        NB_EPOCHS (int, optional): Number of training epochs. Default is 5
        PATIENCE (int, optional): Patience. If in PATIENCE many training 
            iterations the validation loss doesnt decrease, training is 
            stopped. Default is 15
        epsilon (float, optiona):  Default is 0.005
        B (int, optional): Number of samples that shall be taken for the 
            approximation of the expectation of pi. Default is 1. If 
            classification is False, B doesn't matter
        reduction (string, optional): Decides which reduction at loss_functions 
            to take. Options are 'mean' and 'sum'. Default is 'mean'
    """
    # Fehlerabfragen
    assert (neth is None or h_out_features is not None), "The specification of h_out_features is essential when neth is given."

    x = data.x
    y = data.y
    # B is used to evaluate the expectation of pi, i.e. number of samples 
    # taken; Since we are not able to sample in Case of regression, we set B=1 
    # and take the expectation as only sample for a node
    if classification:
        B = 1
    N = x.shape[0]
    unknown = train_mask.logical_not()
    
    # Choose pi, where pi[i] is the probability, that x[i] is known, 
    # i.e. pi[i] = P(r[i] = 1 | y[i], x[i])
    if pi is None:
        pi = torch.ones(N)
        
    # Create models and its optimizers
    modelR = prepare_modelr(x, h_out_features, neth, data.num_classes)
    modely = prepare_modely(nety, data.num_node_features, data.num_classes, classification)
    
    # Create a few variables for main loop
    y_est, y_est_old = torch.zeros_like(y), torch.zeros_like(y)
    convergence_criterion = True
    unknown_acc, val_acc = list(), list()
    epoch = 0
    while epoch < NB_EPOCHS and convergence_criterion:
        epoch += 1
        my_log("Epoch {}:".format(epoch), 0, logging, 1)
        
        # Train Model R
        if epoch > 1:
            pi, _, _ = train_modelr(x, y, data.num_classes, train_mask+val_mask, modely(data), NB_IT_1, B, PATIENCE, logging, modelR, classification)
        train_modely(data, modely, pi, train_mask, val_mask, NB_IT_0, reduction, classification, PATIENCE, logging)

        # Estimate y
        y_est, y_est_old = estimateY(modely, data, classification, y_est, train_mask, N)

        # Update Convergence Criterion
        tmp = sum((y_est[unknown] == y_est_old[unknown]).logical_not()) / (sum(unknown) * 1.0)
        convergence_criterion = tmp > epsilon if epoch > 1 else True
        if not convergence_criterion:
            my_log('Das Konvergenzkriterium ist erreicht. Es haben sich nur {:.2f}% Werte geändert'.format(tmp*100), 0, logging)

        # Evaluate Model after i epochs
        acc = eval_model_with_logging(modely, data, logging, mask=unknown)
        acc_train = eval_model_with_logging(modely, data, logging, mask=train_mask)
        unknown_acc.append(eval_model(modely(data), data, (train_mask + val_mask).logical_not(), classification=classification))
        val_acc.append(eval_model(modely(data), data, val_mask, classification=classification))
        my_log('Accuracy on unknown data: {:.4f}'.format(acc), 1, logging, 1)
        my_log('Accuracy on trainings data: {:.4f}'.format(acc_train), 1, logging, 1)

    return (modely, val_acc, unknown_acc) # modely, val_acc, unknown_acc

def train_net_with_gnm_adapted(data, train_mask, val_mask, 
    classification=True, nety = None, neth = None, h_out_features = None, 
    logging = 0, NB_IT_0 = 100, NB_IT_1 = 30, NB_IT_2 = 100, NB_IT_3 = 150,
    NB_EPOCHS = 5, PATIENCE = 15, epsilon = 0.005, B = 1, reduction='mean',
    expectation='fast'):

    """
    This method is capable of training a neural network to solve a 
    semi-supervised classification or regression problem.
    It is a modification of an algorithm from the paper 
    "Graph-Based Semi-Supervised Learning with Non-ignorable Non-response" 
    written by Fan Zhou, Tengfei Li, Haibo Zhou, Hongtu Zhu and Ye Jieping 
    (2019).
    If no other nety is specified the neural network is choosen to be a 
    2-layer GCN as known from "Semi-Supervised Classification with Graph 
    Convolutional Networks" written by Thomas N. Kipf und Max Welling (2016).

    Args:
        data (torch_geometric.data.data.Data): Data to train on; Should have 
            attribute num_node_features and num_classes
        train_mask (boolean array): An element is True if the corresponding 
            node label is included in the training set
        val_mask (boolean array): An element is True if the corresponding node 
            label is included in the validation set
        classification (boolean, optional): True if data.y describes classes, 
            False if data.y has continuous values. Default is True
        nety (module, optional): it is assumed, that the output is a 
            probability distribution, i.e. sum(out[i]) = 1.
            This can be reached by using the softmax layer as last layer. 
            Default: 2-layer GCN
        neth  (module, optional): Should take x_i as input and returns a 
            h_out_features-dim vector. Default: Multilayer Perceptron
        h_out_features (int, optional): dimension of output of neth
        logging (int, optional): Describes the layer of logging that shall be 
            printed. Default is 0, which means nothing is printed
        NB_IT_0 (int, optional): Maximum number of training iterations for 
            first approximation of y. Default is 100
        NB_IT_1 (int, optional): Maximum number of training iterations for 
            first approximation of secondary model. Default is 30
        NB_IT_3 (int, optional): Maximum number of training iterations for main 
            approximation of secondary model. Default is 100
        NB_IT_4 (int, optional): Maximum number of training iterations for 
            main model. Default is 150
        NB_EPOCHS (int, optional): Number of training epochs. Default is 5
        PATIENCE (int, optional): Patience. If in PATIENCE many training 
            iterations the validation loss doesnt decrease, training is 
            stopped. Default is 15
        epsilon (float, optiona):  Default is 0.005
        B (int, optional): Number of samples that shall be taken for the 
            approximation of the expectation of pi. Default is 1. If 
            classification is False, B doesn't matter
        reduction (string, optional): Decides which reduction at loss_functions 
            to take. Options are 'mean' and 'sum'. Default is 'mean'
        expectation (string, optional): Chooses a method to calculate the
            expectation is Loss 2. Options are 'sample', 'exact' and 'fast'.
            Default is 'fast'
    """
    # Fehlerabfragen
    assert (neth is None or h_out_features is not None), "The specification of h_out_features is essential when neth is given."
    assert (expectation in ['sample', 'exact', 'fast']), "{} is not a valid value for parameter expectation. Valid values are 'sample', 'exact' and 'fast'".format(expectation)

    # B is used to evaluate the expectation of pi, i.e. number of samples 
    # taken; Since we are not able to sample in Case of regression, we set B=1 
    # and take the expectation as only sample for a node
    if classification:
        B = 1
        
    # Create models and its optimizers
    modelR = prepare_modelr(data.x, h_out_features, neth, data.num_classes)
    modely = prepare_modely(nety, data.num_node_features, data.num_classes, classification)
    
    # Train first modely
    train_modely(data, modely, torch.ones(data.x.shape[0]), train_mask, val_mask, NB_IT_0, reduction, classification, PATIENCE, logging)

    # Estimate pi
    modely.eval()
    y_est = modely(data)
    pi_est, _, _ = train_modelr_adapted(data.x, data.y, data.num_classes, train_mask+val_mask, y_est, NB_IT_1, NB_IT_2, B, PATIENCE, logging, modelR, classification, expectation)
    
    # Train final modely
    train_modely(data, modely, pi_est, train_mask, val_mask, NB_IT_0, reduction, classification, PATIENCE, logging)

    # Evaluate Model
    modely.eval()
    y_est = modely(data)
    val_acc = eval_model(y_est, data, val_mask, classification=classification)
    unknown_acc = eval_model(y_est, data, (train_mask+val_mask).logical_not(), classification=classification)

    return (modely, val_acc, unknown_acc)

def train_one_net(data, train_mask, val_mask, net = None, logging = 0,
    classification=True, loss_function= F.cross_entropy,
    val_loss_function=None, NB_IT = 200, PATIENCE = 15):
    '''
    This method is capable of training a neural network to solve a 
    semi-supervised classification or regression problem.
    If no other nety is specified the neural network is choosen to be a 
    2-layer GCN as known from "Semi-Supervised Classification with Graph 
    Convolutional Networks" written by Thomas N. Kipf und Max Welling (2016).
    If no other loss_function is specified the cross entropy loss will be used.
    '''
    
    last_iteration = NB_IT

    # Set up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if classification:
        model = net().to(device) if net is not None else twoLayerConvolutionalNetwork(data.num_node_features, data.num_classes).to(device)
    else: 
        model = net().to(device) if net is not None else twoLayerConvolutionalNetwork(data.num_node_features, 1, activation=lambda x: x).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Variables for loop
    best_val_loss = 999999
    wait = 0

    # Train model
    for i in range(NB_IT):
        # Fit
        model.train()
        optimizer.zero_grad()
        out = model(data)

        loss = loss_function(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        _, val_loss = eval_model(
            model(data), 
            data, 
            mask = val_mask, 
            loss_f = loss_function if val_loss_function is None else val_loss_function)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            if wait >= PATIENCE:
                my_log('Early Stopping in iteration {} while training model.'.format(i), 0, logging)
                last_iteration = i
                break
            wait += 1 
    # Evaluate Model
    val_acc = eval_model(model(data), data, val_mask, classification=classification) # Accuracy on validation data
    acc = eval_model(model(data), data, (train_mask + val_mask).logical_not()) # Accuracy on unkown data
    return (model, val_acc, acc, last_iteration)
