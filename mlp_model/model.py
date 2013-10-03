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
import os



 
class HintLayer1(Linear):
    def fprop(self, state_below):
        p = self._linear_part(state_below)
        p = T.concatenate([T.nnet.sigmoid(p[:,0]).dimshuffle(0,'x'), p[:,1:]], axis=1)
        return p
 
    def cost(self, *args, **kwargs):
        raise NotImplementedError()
 
class HintCost1(MLPCost):
    supervised = True
    def __init__(self, name='cost'):
        self.name = name
        self.use_dropout = False
 
    def get_gradients(self, model, data, ** kwargs):
        """
        model: a pylearn2 Model instance
        X: a batch in model.get_input_space()
        Y: a batch in model.get_output_space()
 
        returns: gradients, updates
            gradients:
                a dictionary mapping from the model's parameters
                         to their gradients
                The default implementation is to compute the gradients
                using T.grad applied to the value returned by __call__.
                However, subclasses may return other values for the gradient.
                For example, an intractable cost may return a sampling-based
                approximation to its gradient.
            updates:
                a dictionary mapping shared variables to updates that must
                be applied to them each time these gradients are computed.
                This is to facilitate computation of sampling-based approximate
                gradients.
                The parameters should never appear in the updates dictionary.
                This would imply that computing their gradient changes
                their value, thus making the gradient value outdated.
        """
 
        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling "+str(type(self))+".__call__"
            print str(type(self))
            print e.message
            raise e
 
        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")
 
        params = list(model.get_params())
 
        grads = T.grad(cost, params, disconnected_inputs = 'raise')
 
        gradients = OrderedDict(izip(params, grads))
 
        updates = OrderedDict()
 
        return gradients, updates
 
    def expr(self, model, data, ** kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        if self.use_dropout:
            Y_hat = model.dropout_fprop(X, default_input_include_prob=self.default_input_include_prob,
                    input_include_probs=self.input_include_probs, default_input_scale=self.default_input_scale,
                    input_scales=self.input_scales
                    )
        else:
            Y_hat = model.fprop(X)
 
        cross_entropy = (-Y[:,0] * T.log(Y_hat[:,0]) - (1 - Y[:,0]) \
            * T.log(1 - Y_hat[:,0])).mean()
        mse = T.sqr(Y[:,1:] - Y_hat[:,1:]).mean()
        return mse + cross_entropy
 
    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)
 
    def get_test_cost(self, model, X, Y):
        Y_hat = model.fprop(X)
 
        cross_entropy = (-Y[:,0] * T.log(Y_hat[:,0]) - (1 - Y[:,0]) \
            * T.log(1 - Y_hat[:,0])).mean()
        return cross_entropy
 
 
class FunLayer1(Linear):
    def fprop(self, state_below):
        p = self._linear_part(state_below)
        p = T.nnet.sigmoid(p)
        return p
 
    def cost(self, *args, **kwargs):
        raise NotImplementedError()
 
class FunCost1(MLPCost):
    supervised = True
    def __init__(self, name='cost'):
        self.name = name
        self.use_dropout = False
 
    def get_gradients(self, model, data, ** kwargs):
        """
        model: a pylearn2 Model instance
        X: a batch in model.get_input_space()
        Y: a batch in model.get_output_space()
 
        returns: gradients, updates
            gradients:
                a dictionary mapping from the model's parameters
                         to their gradients
                The default implementation is to compute the gradients
                using T.grad applied to the value returned by __call__.
                However, subclasses may return other values for the gradient.
                For example, an intractable cost may return a sampling-based
                approximation to its gradient.
            updates:
                a dictionary mapping shared variables to updates that must
                be applied to them each time these gradients are computed.
                This is to facilitate computation of sampling-based approximate
                gradients.
                The parameters should never appear in the updates dictionary.
                This would imply that computing their gradient changes
                their value, thus making the gradient value outdated.
        """
 
        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling "+str(type(self))+".__call__"
            print str(type(self))
            print e.message
            raise e
 
        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")
 
        params = list(model.get_params())
 
        grads = T.grad(cost, params, disconnected_inputs = 'raise')
 
        gradients = OrderedDict(izip(params, grads))
 
        updates = OrderedDict()
 
        return gradients, updates
 
    def expr(self, model, data, ** kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        if self.use_dropout:
            Y_hat = model.dropout_fprop(X, default_input_include_prob=self.default_input_include_prob,
                    input_include_probs=self.input_include_probs, default_input_scale=self.default_input_scale,
                    input_scales=self.input_scales
                    )
        else:
            Y_hat = model.fprop(X)
 
        cross_entropy = (-Y * T.log(Y_hat) - (1 - Y) \
            * T.log(1 - Y_hat)).mean()
        return cross_entropy
 
    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)
 
    def get_test_cost(self, model, X, Y):
        Y_hat = model.fprop(X)
 
        cross_entropy = (-Y * T.log(Y_hat) - (1 - Y) \
            * T.log(1 - Y_hat)).mean()
 
        return cross_entropy

class MightyQuestHPS(HPS):
    pass
    def get_cost_hint1(self):
        mlp_cost = HintCost1()
        # default monitor based save best channel:
        test_cost = mlp_cost.get_test_cost(self.model,
                                            self.minibatch,
                                            self.target)
        self.add_channel('cost',test_cost)
 
        if self.dropout:
            mlp_cost.setup_dropout(
                default_input_include_prob=1,
                default_input_scale=1,
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
 
    def get_layer_hint1(self, layer):
        return HintLayer1(dim=layer['dim'], irange=layer['irange'],istdev=layer['istdev'],
                sparse_init=layer['sparse_init'],sparse_stdev=layer['sparse_stdev'],
                include_prob=layer['include_prob'],init_bias=layer['init_bias'],
                W_lr_scale=layer['W_lr_scale'],b_lr_scale=layer['b_lr_scale'],
                max_row_norm=layer['max_row_norm'],max_col_norm=layer['max_col_norm'],
                layer_name=layer['layer_name'],softmax_columns=layer['softmax_columns'])
 
    def get_cost_fun1(self):
        mlp_cost = FunCost1()
        # default monitor based save best channel:
        test_cost = mlp_cost.get_test_cost(self.model,
                                            self.minibatch,
                                            self.target)
        self.add_channel('cost',test_cost)
 
        if self.dropout:
            mlp_cost.setup_dropout(
                default_input_include_prob=1,
                default_input_scale=1,
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
 
    def get_layer_fun1(self, layer):
        return FunLayer1(dim=layer['dim'], irange=layer['irange'],istdev=layer['istdev'],
                sparse_init=layer['sparse_init'],sparse_stdev=layer['sparse_stdev'],
                include_prob=layer['include_prob'],init_bias=layer['init_bias'],
                W_lr_scale=layer['W_lr_scale'],b_lr_scale=layer['b_lr_scale'],
                max_row_norm=layer['max_row_norm'],max_col_norm=layer['max_col_norm'],
                layer_name=layer['layer_name'],softmax_columns=layer['softmax_columns'])
 
    def setup_channel_mca(self, monitoring_dataset):
        Y = self.model.fprop(self.minibatch)
        MCA = T.mean(T.eq(self.target,T.round(Y)), dtype=config.floatX)
        self.add_channel('mca',MCA,monitoring_dataset)
 
    def setup_channel_mca0(self, monitoring_dataset):
        Y = self.model.fprop(self.minibatch)
        MCA = T.mean(T.eq(self.target,T.round(Y)), dtype=config.floatX)
        self.add_channel('mca0',MCA,monitoring_dataset)
 
    def setup_channel_mse1(self, monitoring_dataset):
        Y = self.model.fprop(self.minibatch)[:,1]
        MSE = T.mean(T.sqr(self.target[:,1] - Y), dtype=config.floatX)
        self.add_channel('mse1',MSE,monitoring_dataset)
 
    def setup_channel_mse2(self, monitoring_dataset):
        Y = self.model.fprop(self.minibatch)[:,2]
        MSE = T.mean(T.sqr(self.target[:,2] - Y), dtype=config.floatX)
        self.add_channel('mse2',MSE,monitoring_dataset)


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
    #import pdb
    #pdb.set_trace()
    if state.dataset == 'mnist':
        dataset = MNIST(which_set='train')
        dim = dataset.X.shape[1]
    elif state.dataset == 'svhn':
        import pdb
        pdb.set_trace()
        dataset = SVHN(which_set='splitted_train')
        dim = dataset.X.shape[1]
    else:
        raise ValueError('only mnist and svhn are supported for now in get_dim_input')
    #base_path = get_data_path(state)
    #test_X = np.load(os.path.join(base_path, 'test_%s_cached_X.npy'%state.task))
    #dim = test_X.shape[1]
    #del test_X
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
# 
# 
# def sanity_checks(state):
#     # Sanity check for fun and diff (task, cost, output layer).
#     print 'Sanity check on hyperparameters ...'
#     # Good config on last layer.
#     last_layer_config = {'diff': {'layer_class': 'hint1', 'dim': 3},
#                          'fun': {'layer_class': 'fun1', 'dim': 1}}
# 
#     if state.task not in ['diff', 'fun']:
#         raise NotImplementedError('The task %s is not supported!'%state.task)
# 
#     task_config = last_layer_config[state.task]
# 
#     print 'The task is ', state.task
# 
#     if state.layers.output1.layer_class != task_config['layer_class']:
#         print 'layers.output1.layer_class: %s => %s'%(state.layers.output1.layer_class,
#                                                       task_config['layer_class'])
#         state.layers.output1.layer_class = task_config['layer_class']
# 
#     if state.layers.output1.dim != task_config['dim']:
#         print 'layers.output1.dim: %s => %s'%(state.layers.output1.dim,
#                                               task_config['dim'])
#         state.layers.output1.dim = task_config['dim']
# 
#     if state.cost_array != [task_config['layer_class']]:
#         print "cost_array: %s => ['%s']"%(state.cost_array,
#                                           task_config['layer_class'])
#         state.cost_array = [task_config['layer_class']]
# 
#     if state.dataset == 'test':
#         if state.pack != 'pack15':
#             print "pack: %s => ['%s']"%(state.pack,
#                                         'pack15')
#             state.dataset = 'default'
#         if state.batch_size != 8:
#             print "back_size: %s => ['%s']"%(state.batch_size,
#                                              8)
#             state.batch_size = 8
# 
#     return state


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
    #path = '/data/lisa/exp/wuzhen/code/'
    #sys.path.append(path)
    #os.environ['PYTHONPATH'] += ':' + path
    state = update_default_layer_hyperparams(state)
    #import pdb
    #pdb.set_trace()
    hps = MightyQuestHPS(state=state)
    #import pdb
    #pdb.set_trace()
    hps.run()
    # TODO: check why valid_nll and valid_rmse are not saved in the
    # experiment's state.
    print 'We will save the experiment state'
    dump_pkl(state, 'state.pkl')
    return channel.COMPLETE

# 
# if __name__ == '__main__':
#     model_config = model_config['mlp']
#     # Test hyperparameters
#     ## Dataset info ##
#     model_config.task = 'fun'
#     model_config.pack = 'pack15'
#     model_config.dataset = 'default+ratios'
#     model_config.batch_size = 32
# 
#     model_config.learning_rate = 0.01
#     model_config.init_momentum = 0.5
#     model_config.ext_array.exp_decay.decay_factor = 0.85
#     model_config.ext_array.exp_decay.min_lr_scale = 1e-2
#     model_config.ext_array.moment_adj.saturate_epoch = 20
#     model_config.term_array.epoch_count.max_epochs = 1000
#     model_config.term_array.early_stopping.proportional_decrease = 1e-5
# 
#     ## Layers ##
#     # First Hidden layer
#     model_config.layers.hidden1.layer_class = 'sigmoid'
#     model_config.layers.hidden1.dim = 100
#     # Second Hidden layer
#     model_config.layers.hidden2.layer_class = 'sigmoid'
#     model_config.layers.hidden2.dim = 100
#     # Third Hidden layer
#     # IMPORTANT:
#     # Since there are only 2 hidden layers in default_config.py, we have to
#     # create the third hidden layer.
#     model_config.layers['hidden3'] = DD()
#     model_config.layers.hidden3['layer_class'] = 'sigmoid'
#     model_config.layers.hidden3['dim'] = 100
# 
#     update_default_layer_hyperparams(model_config)
#     hps = MightyQuestHPS(state=model_config)
#     hps.run()
