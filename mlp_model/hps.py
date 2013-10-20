    # -*- coding: utf-8 -*-
import time, sys, cPickle, os, socket

from pylearn2.utils import serial
from itertools import izip
from pylearn2.utils import safe_zip
from collections import OrderedDict
from pylearn2.utils import safe_union

import numpy as np
import scipy.sparse as spp
import theano.sparse as S

from theano.gof.op import get_debug_values
from theano.printing import Print
from theano import function
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T
import theano

from pylearn2.linear.matrixmul import MatrixMul

from pylearn2.models.model import Model

from pylearn2.training_algorithms.sgd import SGD, MomentumAdjustor
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.costs.cost import SumOfCosts, Cost, MethodCost
from pylearn2.costs.mlp import WeightDecay, L1WeightDecay
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, \
    RectifiedLinear, Softmax, Sigmoid, Linear, Tanh, max_pool_c01b, \
    max_pool, Layer
from pylearn2.models.maxout import Maxout, MaxoutConvC01B
from pylearn2.monitor import Monitor
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import preprocessing as pp

from layers import NoisyRELU, GaussianRELU

from dataset import My_CIFAR10

from pylearn2_objects import *
#from load_model import compute_nll

from jobman import DD, expand

from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.svhn import SVHN

class HPS:
    def __init__(self,
                 state,
                 base_channel_names = ['train_objective'],
                 save_prefix = "model_",
                 cache_dataset = True):
        self.cache_dataset = cache_dataset
        self.dataset_cache = {}
        self.state = state
        self.mbsb_channel_name = self.state.term_array.early_stopping.save_best_channel
        self.base_channel_names = base_channel_names
        self.save_prefix = save_prefix
        # TODO store this in data for each experiment or dataset

    def run(self):
        (model, learner, algorithm) \
            = self.get_config()
        try:
            print 'learning'
          
            learner.main_loop()

        except Exception, e:
            print e

        print 'End of model training'

    def get_config(self):
        # dataset
        self.load_dataset()

        # model
        self.load_model()

        # monitor:
        self.setup_monitor()

        # training algorithm
        algorithm = self.get_train()

        # extensions
        extensions = self.get_extensions()

        # channels
        #self.setup_channels()

        # learner
        learner = Train(dataset=self.train_ddm,
                        model=self.model,
                        algorithm=algorithm,
                        extensions=extensions)

        return (self.model, learner, algorithm)

    def load_dataset(self):
        # TODO: we might need other variables for identifying what kind of
        # extra preprocessing was done such as features product and number
        # of features kept based on MI.
        #base_path = get_data_path(self.state)
        #self.base_path = base_path

        import pdb
        pdb.set_trace()
        
        if self.state.dataset == 'mnist':
            self.test_ddm = MNIST(which_set='test', one_hot=True)

            dataset = MNIST(which_set='train', shuffle=True, one_hot=True)
            train_X, valid_X = np.split(dataset.X, [50000])
            train_y, valid_y = np.split(dataset.y, [50000])
            self.train_ddm = DenseDesignMatrix(X=train_X, y=train_y)
            self.valid_ddm = DenseDesignMatrix(X=valid_X, y=valid_y)
            
        elif self.state.dataset == 'svhn':
            self.train_ddm = SVHN(which_set='splitted_train')
            self.test_ddm = SVHN(which_set='test')
            self.valid_ddm = SVHN(which_set='valid')

        elif self.state.dataset == 'cifar10':

            self.train_ddm = My_CIFAR10(which_set='train', one_hot=True)
            self.test_ddm = None
            self.valid_ddm = My_CIFAR10(which_set='test', one_hot=True)

        
        if self.train_ddm is not None:
            self.nvis = self.train_ddm.X.shape[1]
            self.nout = self.train_ddm.y.shape[1]
            print "nvis, nout :", self.nvis, self.nout
            self.ntrain = self.train_ddm.X.shape[0]
            print "ntrain :", self.ntrain
        
        if self.valid_ddm is not None:
            self.nvalid = self.valid_ddm.X.shape[0]
            print "nvalid :", self.nvalid
        
        if self.test_ddm is not None:
            self.ntest = self.test_ddm.X.shape[0]
            print "ntest :", self.ntest

    def load_model(self):
        model_class = self.state.model_class
        fn = getattr(self, 'get_model_'+model_class)
        self.model = fn()
        return self.model
 
    def get_model_mlp(self):
        self.dropout = False
        self.input_include_probs = {}
        self.input_scales = {}
        self.weight_decay = False
        self.weight_decays = {}
        self.l1_weight_decay = False
        self.l1_weight_decays = {}

        nnet_layers = self.state.layers
        input_space_id = self.state.input_space_id
        nvis = self.nvis
        self.batch_size = self.state.batch_size
        # TODO: add input_space as a config option.
        input_space = None
        # TODO: top_view always False for the moment.
        self.topo_view = False
        assert nvis is not None
        layers = []
        for i,layer in enumerate(nnet_layers.values()):
            layer = expand(layer)
            layer = self.get_layer(layer, i)
            layers.append(layer)
        # create MLP:
        model = MLP(layers=layers,input_space=input_space,nvis=nvis,
                    batch_size=self.batch_size)
        self.mlp = model
        return model

    def get_layer(self, layer, layer_id):
        layer_class = layer.layer_class
        layer_name = layer.layer_name
        dropout_scale = layer.dropout_scale
        dropout_prob = layer.dropout_probability
        weight_decay = layer.weight_decay
        l1_weight_decay = layer.l1_weight_decay
        fn = getattr(self, 'get_layer_'+layer_class)
        if layer_name is None:
            layer_name = layer_class+str(layer_id)
            layer.layer_name = layer_name
        layer = fn(layer)
        # per-layer cost function parameters:
        if (dropout_scale is not None):
            self.dropout = True
            self.input_scales[layer_name] = dropout_scale
        if (dropout_prob is not None):
            self.dropout = True
            self.input_include_probs[layer_name] = (1. - dropout_prob)
        if  (weight_decay is not None):
            self.weight_decay = False
            self.weight_decays[layer_name] = weight_decay
        if  (l1_weight_decay is not None):
            self.l1_weight_decay = False
            self.l1_weight_decays[layer_name] = l1_weight_decay
        return layer

    def get_layer_sigmoid(self, layer):
        return Sigmoid(layer_name=layer.layer_name,dim=layer.dim,irange=layer.irange,
                istdev=layer.istdev,sparse_init=layer.sparse_init,
                sparse_stdev=layer.sparse_stdev, include_prob=layer.include_prob,
                init_bias=layer.init_bias,W_lr_scale=layer.W_lr_scale,
                b_lr_scale=layer.b_lr_scale,max_col_norm=layer.max_col_norm,
                max_row_norm=layer.max_row_norm)

    def get_layer_tanh(self, layer):
        return Tanh(layer_name=layer.layer_name,dim=layer.dim,irange=layer.irange,
                istdev=layer.istdev,sparse_init=layer.sparse_init,
                sparse_stdev=layer.sparse_stdev, include_prob=layer.include_prob,
                init_bias=layer.init_bias,W_lr_scale=layer.W_lr_scale,
                b_lr_scale=layer.b_lr_scale,max_col_norm=layer.max_col_norm,
                max_row_norm=layer.max_row_norm)

    def get_layer_rectifiedlinear(self, layer):
        # TODO: left_slope is set to 0.0  It should be set by the user!
        layer.left_slope = 0.0
        return RectifiedLinear(layer_name=layer.layer_name,dim=layer.dim,irange=layer.irange,
                istdev=layer.istdev,sparse_init=layer.sparse_init,
                sparse_stdev=layer.sparse_stdev, include_prob=layer.include_prob,
                init_bias=layer.init_bias,W_lr_scale=layer.W_lr_scale,
                b_lr_scale=layer.b_lr_scale,max_col_norm=layer.max_col_norm,
                max_row_norm=layer.max_row_norm,
                left_slope=layer.left_slope,use_bias=layer.use_bias)
        
    def get_layer_softmax(self, layer):
        
        return Softmax(layer_name=layer.layer_name,n_classes=layer.dim,irange=layer.irange,
                istdev=layer.istdev,sparse_init=layer.sparse_init,
                init_bias_target_marginals=layer.init_bias, W_lr_scale=layer.W_lr_scale,
                b_lr_scale=layer.b_lr_scale, max_col_norm=layer.max_col_norm,
                max_row_norm=layer.max_row_norm)
        
    def get_layer_noisyRELU(self, layer):
        
        return NoisyRELU(
                        dim=layer.dim,
                        layer_name=layer.layer_name,
                        irange=layer.irange,
                        sparse_init=layer.sparse_init,
                        W_lr_scale=layer.W_lr_scale,
                        b_lr_scale=layer.b_lr_scale,
                        mask_weights = None,
                        max_row_norm=layer.max_row_norm,
                        max_col_norm=layer.max_col_norm,
                        use_bias=True,
                        noise_factor=layer.noise_factor,
                        desired_active_rate=layer.desired_active_rate,
                        adjust_threshold_factor=layer.adjust_threshold_factor
                        )
        
    def get_layer_gaussianRELU(self, layer):
        
        return GaussianRELU(
                        dim=layer.dim,
                        layer_name=layer.layer_name,
                        irange=layer.irange,
                        sparse_init=layer.sparse_init,
                        W_lr_scale=layer.W_lr_scale,
                        b_lr_scale=layer.b_lr_scale,
                        mask_weights = None,
                        max_row_norm=layer.max_row_norm,
                        max_col_norm=layer.max_col_norm,
                        use_bias=True,
                        desired_active_rate=layer.desired_active_rate,
                        adjust_threshold_factor=layer.adjust_threshold_factor,
                        noise_std=layer.noise_std
                        )

    def setup_monitor(self):
        if self.topo_view:
            print "topo view"
            self.minibatch = T.as_tensor_variable(
                        self.valid_ddm.get_batch_topo(self.batch_size),
                        name='minibatch'
                    )
        else:
            print "design view"
            batch = self.valid_ddm.get_batch_design(self.batch_size)
            if isinstance(batch, spp.csr_matrix):
                print "sparse2"
                self.minibatch = self.model.get_input_space().make_batch_theano()
                print type(self.minibatch)
            else:
                self.minibatch = T.as_tensor_variable(
                        self.valid_ddm.get_batch_design(self.batch_size),
                        name='minibatch'
                    )

        self.target = T.matrix('target')

        self.monitor = Monitor.get_monitor(self.model)
        self.log_channel_names = []
        self.log_channel_names.extend(self.base_channel_names)

#         self.monitor.add_dataset(self.valid_ddm, self.state.train_iteration_mode,
#                                     self.batch_size)
#         if self.test_ddm is not None:
#             self.monitor.add_dataset(self.test_ddm, self.state.train_iteration_mode,
#                                         self.batch_size)

    def get_train(self):
        train_class = self.state.train_class
        fn = getattr(self, 'get_train_'+train_class)
        return fn()

    def get_train_sgd(self):

        cost = MethodCost('cost_from_X')
        #cost = self.get_costs()
        num_train_batch = (self.ntrain/self.batch_size)
        print "num training batches:", num_train_batch

        termination_criterion = self.get_terminations()

        monitoring_dataset = {}
        for dataset_id in self.state.monitoring_dataset:
            if dataset_id == 'test' and self.test_ddm is not None:
                monitoring_dataset['test'] = self.test_ddm
            elif dataset_id == 'valid' and self.valid_ddm is not None:
                monitoring_dataset['valid'] = self.valid_ddm
            else:
                monitoring_dataset = None
            
        return SGD( learning_rate=self.state.learning_rate,
                    batch_size=self.state.batch_size,
                    cost=cost,
                    batches_per_iter=num_train_batch,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    init_momentum=self.state.init_momentum,
                    train_iteration_mode=self.state.train_iteration_mode)


    def get_terminations(self):
        if 'term_array' not in self.state:
            return None
        terminations = []

        for term_obj in self.state.term_array.values():
            fn = getattr(self, 'get_term_' + term_obj.term_class)
            terminations.append(fn(term_obj))
        if len(terminations) > 1:
            return And(terminations)
        return terminations[0]

    def get_term_epochcounter(self, term_obj):
        return EpochCounter(term_obj.max_epochs)

    def get_term_monitorbased(self, term_obj):
        print 'monitor_based'
        return MonitorBased(
                prop_decrease=term_obj.proportional_decrease,
                N=term_obj.max_epochs,
                channel_name=term_obj.channel_name
            )

    def get_extensions(self):
        if 'ext_array' not in self.state:
            return []
        extensions = []

        for ext_obj in self.state.ext_array.values():
            fn = getattr(self, 'get_ext_' + ext_obj.ext_class)
            extensions.append(fn(ext_obj))

        # monitor based save best
        print 'save best channel', self.mbsb_channel_name
        if self.mbsb_channel_name is not None:
            self.save_path = self.save_prefix + str(self.state.config_id) + "_optimum.pkl"
            extensions.append(MonitorBasedSaveBest(
                    channel_name = self.mbsb_channel_name,
                    save_path = self.save_path
                )
            )

        return extensions

    def get_ext_exponentialdecayoverepoch(self, ext_obj):
        return ExponentialDecayOverEpoch(
            decay_factor=ext_obj.decay_factor,
            min_lr_scale=ext_obj.min_lr_scale
        )

    def get_ext_momentumadjustor(self, ext_obj):
        return MomentumAdjustor(
            final_momentum=ext_obj.final_momentum,
            start=ext_obj.start_epoch,
            saturate=ext_obj.saturate_epoch
        )


'''
def get_data_path(state):
    # TODO: we might need other variables for identifying what kind of
    # extra preprocessing was done such as features product and number
    # of features kept based on MI.
    task = state.task
    pack = state.pack
    dataset = state.dataset
    import os
    os.environ['BASE_MQ_DATA_PATH'] = '/data/lisa/data/'
    # Get mq cached data path from environment variable.
    base_mq_data_path = os.environ.get('BASE_MQ_DATA_PATH')
    

    if base_mq_data_path is None:
        raise NotImplementedError('The environment variable BASE_MQ_DATA_PATH was not found.')

    if task == 'svhn':
	base_path = os.path.join(base_mq_data_path, '%s/' % SVHN)
    
    
    if task == 'fun':
        base_path = os.path.join(base_mq_data_path, "%s/cached/%s/%s"
                                 %(pack, task, dataset))
    elif task == 'diff':
        base_path = os.path.join(base_mq_data_path, "%s/cached/%s"
                                 %(pack, dataset))
    else:
        raise NotImplementedError('task=%s not supported yet!'%task)
    return base_path
'''
