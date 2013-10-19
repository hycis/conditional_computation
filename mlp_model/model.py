    # -*- coding: utf-8 -*-
import theano.tensor as T
import theano
from theano import config
import numpy
import scipy
import cPickle
import sys
import copy
from math import sqrt
from jobman import flatten

from mlp_model.hps import *
from tools.default_config import model_config, layer_config

from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.svhn import SVHN
from pylearn2.datasets.cifar10 import CIFAR10
import os

from pylearn2_objects import *


def update_irange_in_layer(layer, prev_layer_dim):
    if layer.layer_class == 'tanh':
        # Case: tanh layer
        dim = layer.dim
        layer.irange = sqrt(6. / (prev_layer_dim + dim))
    elif layer.layer_class == 'sigmoid':
        # Case: sigmoid layer
        dim = layer.dim
        layer.irange = 4*sqrt(6. / (prev_layer_dim + dim))
        
def get_dim_input(state):

    if state.dataset == 'mnist':
        dataset = MNIST(which_set='test')
        dim = dataset.X.shape[1]
    elif state.dataset == 'svhn':
        dataset = SVHN(which_set='test')
        dim = dataset.X.shape[1]
    elif state.dataset == 'cifar10':
        dataset = CIFAR10(which_set='test')
        dim = dataset.X.shape[1]
    else:
        raise ValueError('only mnist, cifar10 and svhn are supported for now in get_dim_input')

    del dataset
    return dim


def update_default_layer_hyperparams(state):
    #state = sanity_checks(state)
    prev_layer_dim = get_dim_input(state)
    for key,layer in state.layers.iteritems():
        default_hyperparams = copy.deepcopy(layer_config)
        update_irange_in_layer(layer, prev_layer_dim)
        prev_layer_dim = layer.dim
        default_hyperparams.update(layer)
        state.layers[key] = default_hyperparams
    # TODO: find a better way to check for list of items in the hyperparams.
    # random_sampling should give a good string representation for list of values
    # in the hyperparams.
    flattened_state = flatten(state)
    for k,v in flattened_state.iteritems():
        if str(v)[0] == '[' and str(v)[-1] == ']' and type(v) != type([]):
            # It is a list of values but represented as a string.
            # Get just the content of the list.
            values = str(v)[1:-1]
            # Add the content in a proper list (not represented as a string).
            values = [val.strip() for val in values.split(',')]
            flattened_state[k] = values
    # TODO : why the modified state is not kept after returning from this function?
    # However, the updated state above is kept after returning from this function.
    state = expand(flattened_state)
    return state

def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


def experiment(state, channel):
    state = update_default_layer_hyperparams(state)
    hps = HPS(state=state)

    hps.run()

    # experiment's state.
    print 'We will save the experiment state'
    dump_pkl(state, 'state.pkl')
    return channel.COMPLETE
