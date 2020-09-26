from torch_geometric.datasets import Planetoid
import pickle # Lokales Speichern von Objekten

'''
This file provides a useful way for quick access to the cora and citeseer
dataset.
Furthermore the methods pickle_read and pickle_write allow you to safe
stuff by using an one-liner.
'''

def load_dataset(dataset_name):
    """ 
    Args:
        dataset (string): Should be either 'Cora' or 'Citeseer'
    """
    PATH = 'GNM_Toolbox/data/datasets/{}_dataset.pkl'.format(dataset_name)
    dataset = None
    try:
        with open(PATH, 'rb') as data:
            dataset = pickle.load(data)
        print('Found dataset on harddrive.')
    except:
        print("Couldn't load dataset from harddrive.")
        dataset_pre_saved = Planetoid(root='/tmp/{}'.format(dataset_name), name=dataset_name)
        with open(PATH, 'wb') as output:
            pickle.dump(dataset_pre_saved, output, pickle.HIGHEST_PROTOCOL)
        dataset = dataset_pre_saved
        print('Loaded dataset from github.')
    
    return dataset

def pickle_read(PATH):
    '''
    Trys reading a file from path '/pickle_data/<PATH>'.
    '''
    data = None
    with open('pickle_data/{}'.format(PATH), 'rb') as instream:
        data = pickle.load(instream)
    return data

def pickle_write(PATH, data):
    '''
    Saves data to file at path '/pickle_data/<PATH>'.
    The folder '/pickle_data' must exist already.
    '''
    with open('pickle_data/{}'.format(PATH), 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)