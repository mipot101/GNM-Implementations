# GNM-Implementations

This github repositiory comes hand in hand with my bachelor thesis.
For online access see: https://github.com/mipot101/BachelorarbeitPDF

In this repo you'll find jupyter notebooks which implement all the experiments done in my bachelor thesis.
Furthermore you'll find the GNM Toolbox which implements the algorithm proposed in "Graph-based Semi-Supervised Learning with Nonignorable Nonresponses" by Fan Zhou et al, 2019.

The toolbox is based on pytorch by Matthias Fey  and Jan E. Lenssen, 2019. (https://github.com/rusty1s/pytorch_geometric)

Included are the datasets 'Cora' and 'Citeseer' from the `"Revisiting Semi-Supervised Learning with Graph Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
Those are accessible at https://github.com/kimiyoung/planetoid/raw/master/data. I do not own any rights to those.

The architecture of Graph Convolution Networks (Thomas N. Kipf, Max Welling ICLR 2017) is used.

## Replicating experiments

To replicate an experiment it is necessary that the notebook and the GNM_Toolbox are in the same directory. Therefore you can either copy the experiment in the root folder or place a duplicate of the GNM_Toolbox in the directory of the notebook.

## The Toolbox
The Toolbox builds up around the module 'gnm.py'.
In this module you will find 3 methods which allow you to train a net.
* 'train_net_with_gnm': Trains a net based on the algorithm proposed by Fan Zhou et al, 2019
* 'train_net_with_gnm_adapted': Trains a net based on the algorithm I proposed in my bachelor thesis
* 'train_one_net': Usual way to train a net. Default Loss Function is NLL, but different ones can be choosen as parameter

All functions expect you to give the data, train_mask and val_mask as parameters.
All other parameters are optional.
