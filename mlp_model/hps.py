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
from pylearn2.costs.cost import SumOfCosts, Cost
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
                 mbsb_channel_name = 'valid_hps_cost',
                 cache_dataset = True):
        self.cache_dataset = cache_dataset
        self.dataset_cache = {}
        self.state = state

        self.base_channel_names = base_channel_names
        self.save_prefix = save_prefix
        # TODO store this in data for each experiment or dataset
        self.mbsb_channel_name = mbsb_channel_name

    def run(self):
        (model, learner, algorithm) \
            = self.get_config()
        try:
            print 'learning'
            #import pdb
            #pdb.set_trace()            
            learner.main_loop()

        except Exception, e:
            print e

        # Compute the model NLL on train, valid and test.
#         compute_nll(exp_base_path=os.getcwd(), target_base_path=self.base_path,
#                     model_name=self.save_path,
#                     cache=(self.train_ddm, self.valid_ddm, self.test_ddm),
#                     state=self.state)
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
        self.setup_channels()

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
	
	'''
        task = self.state.task
        train_X = np.load(os.path.join(base_path, 'train_%s_cached_X.npy'%task))
        train_Y = np.load(os.path.join(base_path, 'train_%s_cached_Y.npy'%task))
        self.train_ddm = DenseDesignMatrix(X=train_X, y=train_Y)
        del train_X
        del train_Y

        valid_X = np.load(os.path.join(base_path, 'valid_%s_cached_X.npy'%task))
        valid_Y = np.load(os.path.join(base_path, 'valid_%s_cached_Y.npy'%task))
        self.valid_ddm = DenseDesignMatrix(X=valid_X, y=valid_Y)
        del valid_X
        del valid_Y
 
        test_X = np.load(os.path.join(base_path, 'test_%s_cached_X.npy'%task))
        test_Y = np.load(os.path.join(base_path, 'test_%s_cached_Y.npy'%task))
        self.test_ddm = DenseDesignMatrix(X=test_X, y=test_Y)
        del test_X
        del test_Y
	'''
        #import pdb
        #pdb.set_trace()
        if self.state.dataset == 'mnist':
            self.train_ddm = MNIST(which_set='train', one_hot=True)
            self.test_ddm = MNIST(which_set='test', one_hot=True)
            self.valid_ddm = MNIST(which_set='test', one_hot=True)

        if self.state.dataset == 'svhn':
            self.train_ddm = SVHN(which_set='splitted_train')
            self.test_ddm = SVHN(which_set='test')
            self.valid_ddm = SVHN(which_set='valid')
        
        #self.monitoring_dataset = {'valid': self.valid_ddm}          


        #self.nvis = self.train_ddm.get_design_matrix().shape[1]
        #self.nout = self.train_ddm.get_targets().shape[1]
        self.nvis = self.train_ddm.X.shape[1]
        self.nout = self.train_ddm.y.shape[1]
        
        #self.ntrain = self.train_ddm.get_design_matrix().shape[0]
        self.ntrain = self.train_ddm.X.shape[0]
        self.nvalid = self.valid_ddm.X.shape[0]
        self.ntest = self.test_ddm.X.shape[0]
        #self.nvalid = self.valid_ddm.get_design_matrix().shape[0]
        #self.ntest = self.test_ddm.get_design_matrix().shape[0]

        print "nvis, nout :", self.nvis, self.nout
        print "ntrain :", self.ntrain
        print "nvalid :", self.nvalid
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

        self.monitor.add_dataset(self.valid_ddm, 'sequential',
                                    self.batch_size)
        if self.test_ddm is not None:
            self.monitor.add_dataset(self.test_ddm, 'sequential',
                                        self.batch_size)

    def get_train(self):
        train_class = self.state.train_class
        fn = getattr(self, 'get_train_'+train_class)
        return fn()

    def get_train_sgd(self):
        # cost
        cost = self.get_costs()

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
		
        
        return SGD( learning_rate=self.state.learning_rate, cost=cost,
                    batch_size=self.state.batch_size,
                    batches_per_iter=num_train_batch,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    init_momentum=self.state.init_momentum,
                    train_iteration_mode='batchwise_shuffled_equential')

    def get_costs(self):
        costs = []
        for cost_type in self.state.cost_array:
            costs.extend(self.get_cost(cost_type))

        if len(costs) > 1:
            cost = SumOfCosts(costs)
        else:
            cost = costs[0]

        return cost

    def get_cost(self, cost_type):
        fn = getattr(self, 'get_cost_'+cost_type)
        return fn()

    def get_cost_mlp(self):
        raise NotImplementedError('get_cost_mlp not supported!')
        # TODO: add missing_target_value as a config option.
        missing_target_value = None
        # TODO: check if default_dropout_prob is the same as dropout_probability
        # and should it be taken from the last layer.
        default_dropout_prob = self.state.layers[-1]['dropout_probability']
        default_dropout_scale = self.state.layers[-1]['dropout_scale']
        mlp_cost = MLPCost(cost_type=self.state.cost_type,
                            missing_target_value=missing_target_value)
        # default monitor based save best channel:
        test_cost = mlp_cost.get_test_cost(self.model,
                                           self.minibatch,
                                           self.target)
        self.add_channel('cost',test_cost)

        if self.dropout:
            mlp_cost.setup_dropout(
                default_input_include_prob=(1.-default_dropout_prob),
                default_input_scale=default_dropout_scale,
                input_scales=self.input_scales,
                input_include_probs=self.input_include_probs)

        costs = [mlp_cost]
        if self.weight_decay:
            coeffs = []
            for layer in self.mlp.layers:
                coeffs.append(self.weight_decays[layer.layer_name])
            wd_cost = WeightDecay(coeffs)
            costs.append(wd_cost)
        if self.l1_weight_decay:
            coeffs = []
            for layer in self.mlp.layers:
                coeffs.append(self.l1_weight_decays[layer.layer_name])
            lwd_cost = L1WeightDecay(coeffs)
            costs.append(lwd_cost)
        return costs

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
        print 'self.mbsb_channel_name', self.mbsb_channel_name
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

    def setup_channels(self):
        if self.state.channel_array is None:
            return

        for channel_id in self.state.channel_array:
            fn = getattr(self, 'setup_channel_'+channel_id)
            fn(self.state.monitoring_dataset)

    def add_channel(self, channel_name, tensor_var,
                    dataset_names=['valid','test']):
        for dataset_name in dataset_names:
            if dataset_name == 'valid':
                ddm = self.valid_ddm
            elif dataset_name == 'test':
                if self.test_ddm is None:
                    continue
                ddm = self.test_ddm
            log_channel_name = dataset_name+'_hps_'+channel_name

            self.log_channel_names.append(log_channel_name)

            self.monitor.add_channel(log_channel_name,
                                (self.minibatch, self.target),
                                tensor_var, ddm)

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