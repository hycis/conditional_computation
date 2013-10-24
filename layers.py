from pylearn2.models.mlp import Linear, MLP, Softmax, Tanh
import numpy as np
import theano.tensor as T
from theano.compat.python2x import OrderedDict
#from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pylearn2.space import CompositeSpace

from theano import config

from pylearn2.models.mlp import MLP

class My_MLP(MLP):
    
    def test_fprop(self, state_below, return_all = False):

        rval = self.layers[0].test_fprop(state_below)

        rlist = [rval]

        for layer in self.layers[1:]:
            rval = layer.test_fprop(rval)
            rlist.append(rval)

        if return_all:
            return rlist
        return rval
    
    def get_monitoring_channels(self, data):
        """
        data is a flat tuple, and can contain features, targets, or both
        """
        X, Y = data
        state = X
        rval = OrderedDict()
        import pdb
        pdb.set_trace()
    
        for layer in self.layers:
            ch = layer.get_monitoring_channels()
            for key in ch:
                rval[layer.layer_name+'_'+key] = ch[key]
            state = layer.test_fprop(state)
            args = [state]
            if layer is self.layers[-1]:
                args.append(Y)
            ch = layer.get_monitoring_channels_from_state(*args)
            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                rval[layer.layer_name+'_'+key]  = ch[key]

        return rval


class GaussianRELU(Linear):

    def __init__(self, noise_std=1, desired_active_rate=0.1, adjust_threshold_factor=1, **kwargs):
        super(GaussianRELU, self).__init__(**kwargs)
        self.std = noise_std
        self.adjust_threshold_factor = adjust_threshold_factor
        self.desired_active_rate = desired_active_rate
        #self.threshold = theano.shared(np.zeros(shape=(self.dim,)))
        
        
        #self.threshold = T.zeros(shape=(self.dim,), dtype=config.floatX)
        #self.threshold = theano.sparse.basic.as_sparse_or_tensor_variable(np.zeros(shape=(self.dim,)))
        #self.active_rate = theano.tensor.zeros(shape=(self.dim,), dtype=config.floatX)
        
    def fprop(self, state_below):
        print "======fprop====="
        
        
        rng = RandomStreams(seed=234)

        #size = theano.tensor.as_tensor_variable((state_below.shape[0], self.dim))
        self.noise = rng.normal(size=(state_below.shape[0], self.dim), avg=0, std=self.std)
        #self.noise = T.log(un/(1-un))
        p = self._linear_part(state_below) + self.noise

        batch_size = (p.shape[0]).astype(config.floatX)
        self.active_rate = T.gt(p, self.threshold).sum(axis=0, dtype=config.floatX) / batch_size
        
        return T.gt(p, self.threshold) * p
        
    # fprop used by test set for monitoring data    
    def test_fprop(self, state_below):
        p = self._linear_part(state_below)
        return T.max(p, 0)
    
    def get_params(self):
        print "===get_params==="
        return super(GaussianRELU, self).get_params() + [self.threshold]


 
    def set_input_space(self, space):
        print "===set_input_space==="
        super(GaussianRELU, self).set_input_space(space)
        self.threshold = sharedX(np.zeros(shape=(self.dim,)), 'threshold')


#     
    def censor_updates(self, updates):
        print "===censor_updates==="
 
        super(GaussianRELU, self).censor_updates(updates)        
        renormalize = (T.gt(self.active_rate, self.desired_active_rate) - 0.5) * 2.
        updates[self.threshold] += renormalize * T.abs_(self.desired_active_rate - 
                    self.active_rate) * self.adjust_threshold_factor

#     def get_monitoring_channels_from_state(self, state, target=None):
#         
#         rval = super(GaussianRELU, self).get_monitoring_channels_from_state(state)
#         
#         print "===get_monitor_channels_from_state==="
        
#         active_rate = self.active_rate.astype(config.floatX)
# 
#         
#         max_active_rate = active_rate.max()
#         min_active_rate = active_rate.min()
#         mean_active_rate = active_rate.mean()
#         
#         max_threshold = self.threshold.max()
#         min_threshold = self.threshold.min()
#         mean_threshold = self.threshold.mean()
#         
#         
#         max_noise = self.noise.max()
#         min_noise = self.noise.min()
#         mean_noise = self.noise.mean()
#          
#         
#         
# #         num_row = self.active_rate.shape[0] * 1.
#         #num_col = self.active_rate.shape[1] * 1.
#         
#         
# #         renormalize = (T.gt(self.desired_active_rate, self.active_rate) - 0.5) * 2
# #         factor = renormalize * T.abs_(self.desired_active_rate - self.active_rate) * self.bias_factor
# #         
# #         rval['==factor mean=='] = T.mean(factor)
# #         rval['==factor shape=='] = factor.shape[0] * 1.
# #         rval['==factor max=='] = T.max(factor)
# #         rval['==factor min=='] = T.min(factor)
# #         
# #       
#         #rval["===p.shape[0]"] = state.shape[0] * 1. 
#         #rval['===p.shape[1]'] = state.shape[1] * 1. 
#         
#         rval['===max_active_rate===='] = max_active_rate
#         rval['===min_active_rate===='] = min_active_rate
#         rval['===mean_active_rate===='] = mean_active_rate
#         rval['===max_noise==='] = max_noise
#         rval['===min_noise==='] = min_noise
#         rval['===mean_noise==='] = mean_noise
#         
#         rval['===max_threshold==='] = max_threshold
#         rval['===min_threshold==='] = min_threshold
#         rval['===mean_threshold==='] = mean_threshold
#         
#         rval['===desired_active_rate==='] = self.desired_active_rate
#         
#         rval['active_rate_1'] = active_rate[1]
#         rval['active_rate_15'] = active_rate[15]
#         rval['active_rate_300'] = active_rate[30]
#         rval['active_rate_450'] = active_rate[45]
#         rval['active_rate_700'] = active_rate[50]
#         
#         
#         rval['===<active_rate_100>==='] = active_rate[99]
#         
#         rval['===<active_rate_100_threshold>'] = self.threshold[99]
#         
#         
#         rval['===max_active_rate_threshold>'] = self.threshold[self.active_rate.argmax()]
#         #rval['===min_active_rate_threshold>'] = self.threshold[self.active_rate.argmin()]
# 
#         
# #         rval['===num_row_active_rate===='] = num_row
# #         #rval['===num_col_active_rate===='] = num_col
#         
#



        return rval
    


class NoisyRELU(Linear):

    def __init__(self, noise_factor=1, desired_active_rate=0.1, adjust_threshold_factor=1, **kwargs):
        super(NoisyRELU, self).__init__(**kwargs)
        self.noise_factor = noise_factor
        self.adjust_threshold_factor = adjust_threshold_factor
        self.desired_active_rate = desired_active_rate
        #self.threshold = theano.shared(np.zeros(shape=(self.dim,)))
        
        
        #self.threshold = T.zeros(shape=(self.dim,), dtype=config.floatX)
        #self.threshold = theano.sparse.basic.as_sparse_or_tensor_variable(np.zeros(shape=(self.dim,)))
        #self.active_rate = theano.tensor.zeros(shape=(self.dim,), dtype=config.floatX)
        
    def fprop(self, state_below):
        
        print "======fprop====="
        
        rng = RandomStreams(seed=234)

        #size = theano.tensor.as_tensor_variable((state_below.shape[0], self.dim))
        un = rng.uniform(size=(state_below.shape[0], self.dim), low=0., high=1., dtype=config.floatX)
        self.noise = T.log(un/(1-un))
        p = self._linear_part(state_below) + self.noise * self.noise_factor

        batch_size = (p.shape[0]).astype(config.floatX)
        self.active_rate = T.gt(p, self.threshold).sum(axis=0, dtype=config.floatX) / batch_size
        
        return T.gt(p, self.threshold) * p
      
    # fprop used by test set for monitoring data    
    def test_fprop(self, state_below):
        p = self._linear_part(state_below)
        return T.max(p, 0)
    

    def get_params(self):
        print "===get_params==="
        return super(NoisyRELU, self).get_params() + [self.threshold]


 
    def set_input_space(self, space):
        print "===set_input_space==="
        super(NoisyRELU, self).set_input_space(space)
        self.threshold = sharedX(np.zeros(shape=(self.dim,)), 'threshold')


#     
    def censor_updates(self, updates):
        print "===censor_updates==="
 
        super(NoisyRELU, self).censor_updates(updates)        
        renormalize = (T.gt(self.active_rate, self.desired_active_rate) - 0.5) * 2.
        updates[self.threshold] += renormalize * T.abs_(self.desired_active_rate - 
                    self.active_rate) * self.adjust_threshold_factor

#     def get_monitoring_channels(self):
#         X, Y = 
# 
#     def get_monitoring_channels_from_state(self, state, target=None):
#         
#         rval = super(NoisyRELU, self).get_monitoring_channels_from_state(state)
        
#         print "===get_monitor_channels_from_state==="
#         
#         active_rate = self.active_rate.astype(config.floatX)
# 
#         
#         max_active_rate = active_rate.max()
#         min_active_rate = active_rate.min()
#         mean_active_rate = active_rate.mean()
#         
#         max_threshold = self.threshold.max()
#         min_threshold = self.threshold.min()
#         mean_threshold = self.threshold.mean()
#         
#         
#         max_noise = self.noise.max()
#         min_noise = self.noise.min()
#         mean_noise = self.noise.mean()
#          
#         
#         
# #         num_row = self.active_rate.shape[0] * 1.
#         #num_col = self.active_rate.shape[1] * 1.
#         
#         
# #         renormalize = (T.gt(self.desired_active_rate, self.active_rate) - 0.5) * 2
# #         factor = renormalize * T.abs_(self.desired_active_rate - self.active_rate) * self.bias_factor
# #         
# #         rval['==factor mean=='] = T.mean(factor)
# #         rval['==factor shape=='] = factor.shape[0] * 1.
# #         rval['==factor max=='] = T.max(factor)
# #         rval['==factor min=='] = T.min(factor)
# #         
# #       
#         #rval["===p.shape[0]"] = state.shape[0] * 1. 
#         #rval['===p.shape[1]'] = state.shape[1] * 1. 
#         
#         rval['===max_active_rate===='] = max_active_rate
#         rval['===min_active_rate===='] = min_active_rate
#         rval['===mean_active_rate===='] = mean_active_rate
#         rval['===max_noise==='] = max_noise
#         rval['===min_noise==='] = min_noise
#         rval['===mean_noise==='] = mean_noise
#         
#         rval['===max_threshold==='] = max_threshold
#         rval['===min_threshold==='] = min_threshold
#         rval['===mean_threshold==='] = mean_threshold
#         
#         rval['===desired_active_rate==='] = self.desired_active_rate
#         
#         rval['active_rate_1'] = active_rate[1]
#         rval['active_rate_15'] = active_rate[15]
#         rval['active_rate_300'] = active_rate[30]
#         rval['active_rate_450'] = active_rate[45]
#         rval['active_rate_700'] = active_rate[50]
#         
#         
#         rval['===<active_rate_100>==='] = active_rate[99]
#         
#         rval['===<active_rate_100_threshold>'] = self.threshold[99]
#         
#         
#         rval['===max_active_rate_threshold>'] = self.threshold[self.active_rate.argmax()]
        #rval['===min_active_rate_threshold>'] = self.threshold[self.active_rate.argmin()]

        
#         rval['===num_row_active_rate===='] = num_row
#         #rval['===num_col_active_rate===='] = num_col
#         
#

        return rval
    
    
class My_Softmax(Softmax):
    
    def test_fprop(self, state_below):
        return self.fprop(state_below)

class My_Tanh(Tanh):
    
    def test_fprop(self, state_below):
        return self.fprop(state_below)
    
#     def get_monitoring_channels(self):
# 
#         if self.no_affine:
#             return OrderedDict()
# 
#         W = self.W
# 
#         assert W.ndim == 2
# 
#         sq_W = T.sqr(W)
# 
#         row_norms = T.sqrt(sq_W.sum(axis=1))
#         col_norms = T.sqrt(sq_W.sum(axis=0))
# 
#         return OrderedDict([
#                             ('row_norms_min'  , row_norms.min()),
#                             ('row_norms_mean' , row_norms.mean()),
#                             ('row_norms_max'  , row_norms.max()),
#                             ('col_norms_min'  , col_norms.min()),
#                             ('col_norms_mean' , col_norms.mean()),
#                             ('col_norms_max'  , col_norms.max()),
#                             ])

#     def get_monitoring_channels_from_state(self, state, target=None):
# 
#         mx = state.max(axis=1)
# 
#         rval =  OrderedDict([
#                 ('mean_max_class' , mx.mean()),
#                 ('max_max_class' , mx.max()),
#                 ('min_max_class' , mx.min())
#         ])
# 
#         if target is not None:
#             y_hat = T.argmax(state, axis=1)
#             y = T.argmax(target, axis=1)
#             misclass = T.neq(y, y_hat).mean()
#             misclass = T.cast(misclass, config.floatX)
#             rval['misclass'] = misclass
#             rval['nll'] = self.cost(Y_hat=state, Y=target)
# 
#         return rval
    
    
    
    