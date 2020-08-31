import pickle
import numpy as np

# read data encoded with pickle
def unpickle(f):
    with open(f, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


# load both the samples & the names
def load_data(sample_file):
    '''
    @return data, label, label_names
    '''
    data_dict = unpickle(sample_file)
    data = data_dict[b'data']
    labels = data_dict[b'labels']

    return data, labels


def load():
    path = './cifar-10-batches-py/'

    '''Read all training data'''
    X_train = []
    y_train = []
    for i in range(1, 6):
        data, labels = load_data(path + 'data_batch_' + str(i))
        X_train.append(data) 
        y_train.append(labels)

    X_train = np.resize(X_train, [50000, 3, 32, 32])
    y_train = np.resize(y_train, [50000])
    
    '''Read test data'''
    data, labels = load_data(path + 'test_batch')
    X_test = np.resize(data, [10000, 3, 32, 32])
    y_test = np.resize(labels, [10000])

    label_names = unpickle(path + 'batches.meta')

    return X_train, y_train, X_test, y_test, label_names
