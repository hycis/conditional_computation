from pylearn2.models.mlp import Linear, MLP
import numpy as np
import theano.tensor as T
from theano.compat.python2x import OrderedDict
#from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pylearn2.space import CompositeSpace


class NoisyRELU(Linear):

    def __init__(self, noise_factor=1, desired_active_rate=0.1, adjust_threshold_factor=1, **kwargs):
        super(NoisyRELU, self).__init__(**kwargs)
        self.noise_factor = noise_factor
        self.adjust_threshold_factor = adjust_threshold_factor
        self.desired_active_rate = desired_active_rate
        self.threshold = theano.shared(np.zeros(shape=(self.dim,)))
        
        #self.threshold = T.zeros(shape=(self.dim,), dtype=theano.config.floatX)
        #self.threshold = theano.sparse.basic.as_sparse_or_tensor_variable(np.zeros(shape=(self.dim,)))
        #self.active_rate = theano.tensor.zeros(shape=(self.dim,), dtype=theano.config.floatX)
        
    def fprop(self, state_below):
        print "======fprop====="
        
        rng = RandomStreams(seed=234)

        size = theano.tensor.as_tensor_variable((state_below.shape[0], self.dim))
        un = rng.uniform(size=size, low=0., high=1.)
        self.noise = T.log(un/(1-un))
        p = self._linear_part(state_below) + self.noise * self.noise_factor
        
        batch_size = p.shape[0]
        self.active_rate = (T.gt(p, self.threshold).sum(axis=0, dtype=theano.config.floatX) / batch_size).astype(theano.config.floatX)
                #import pdb
        #pdb.set_trace()
        
        #import traceback
        #trace = traceback.format_exc()
        
        rval = T.gt(p, self.threshold).astype(theano.config.floatX) * p
        return rval
        
        #batch_size = p.shape[0] 
        #self.active_rate = T.gt(p, 0).sum(axis=0, dtype=theano.config.floatX) / batch_size
        
        #factor = renormalize * T.abs_(self.desired_active_rate - self.active_rate) * self.adjust_threshold_factor
        
        #self.threshold += factor
        #p = T.maximum(0, p)


        #return p
        
#     def get_data_specs(self, model):
#         space = CompositeSpace([model.get_input_space(), model.get_output_space()])
#         sources = (model.get_input_source(), model.get_target_source())
#         return (space, sources)        
# 
# 
#     def get_monitoring_data_specs(self):
#         """
#         Return the (space, source) data_specs for self.get_monitoring_channels.
# 
#         In this case, we want the inputs and targets.
#         """
#         space = CompositeSpace((self.get_input_space(),
#                                 self.get_output_space()))
#         source = (self.get_input_source(), self.get_target_source())
#         return (space, source)


#     
    def censor_updates(self, updates):
        print "====censor_updates====="
 
        super(NoisyRELU, self).censor_updates(updates)

        
        renormalize = (T.gt(self.active_rate, self.desired_active_rate) - 0.5) * 2
        #self.threshold = self.active_rate
        #T.abs_(self.desired_active_rate - self.active_rate) * self.adjust_threshold_factor
        updates[self.threshold] += renormalize * T.abs_(self.desired_active_rate - self.active_rate) * self.adjust_threshold_factor

                   
    def cost(self, *args, **kwargs):
        raise NotImplementedError()
     
    def get_monitoring_channels(self):
 
        W ,= self.transformer.get_params()
 
        assert W.ndim == 2
 
        sq_W = T.sqr(W)
 
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
         
        print 'get_monitoring ===== '
        
        
         
        return OrderedDict([
#                             ('=====max_active_rate', max_active_rate),
#                             ('=====mean_active_rate', mean_active_rate),
#                             ('=====min_active_rate', min_active_rate),
#                              ('=====max_noise=====', max_noise),
#                              ('=====mean_noise=====', mean_noise),
#                              ('=====min_noise=====', min_noise),
#                             ('w_shape_0', W.shape[0] * 1.),
#                             ('w_shape_1', W.shape[1] * 1.),
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])
 
    def get_monitoring_channels_from_state(self, state, target=None):
        
        print "=========get monitor channels from state"
        rval =  OrderedDict()
 
        mx = state.max(axis=0)
        mean = state.mean(axis=0)
        mn = state.min(axis=0)
        rg = mx - mn
         
#         active_rate = []
#         for i in xrange(self.dim):
#             active_rate.append(T.sum(T.neq(state[:][i], 0), dtype=theano.config.floatX) / (state.shape[0]))
#  
        max_active_rate = self.active_rate.max()
        min_active_rate = self.active_rate.min()
        mean_active_rate = self.active_rate.mean()
        
        max_threshold = self.threshold.max()
        min_threshold = self.threshold.min()
        mean_threshold = self.threshold.mean()
        
        
        max_noise = self.noise.max()
        min_noise = self.noise.min()
        mean_noise = self.noise.mean()
         
        
#         num_row = self.active_rate.shape[0] * 1.
        #num_col = self.active_rate.shape[1] * 1.
        
        
#         renormalize = (T.gt(self.desired_active_rate, self.active_rate) - 0.5) * 2
#         factor = renormalize * T.abs_(self.desired_active_rate - self.active_rate) * self.bias_factor
#         
#         rval['==factor mean=='] = T.mean(factor)
#         rval['==factor shape=='] = factor.shape[0] * 1.
#         rval['==factor max=='] = T.max(factor)
#         rval['==factor min=='] = T.min(factor)
#         
#       
        #rval["===p.shape[0]"] = state.shape[0] * 1. 
        #rval['===p.shape[1]'] = state.shape[1] * 1. 
        rval['===max_active_rate===='] = max_active_rate
        rval['===min_active_rate===='] = min_active_rate
        rval['===mean_active_rate===='] = mean_active_rate
        rval['===max_noise==='] = max_noise
        rval['===min_noise==='] = min_noise
        rval['===mean_noise==='] = mean_noise
        
        rval['===max_threshold==='] = max_threshold
        rval['===min_threshold==='] = min_threshold
        rval['===mean_threshold==='] = mean_threshold
        
        rval['===desired_active_rate==='] = self.desired_active_rate
        
#         rval['===num_row_active_rate===='] = num_row
#         #rval['===num_col_active_rate===='] = num_col
#         
#         
#         rval['active_rate_cal_1'] = self.active_rate[1]
#         rval['active_rate_1'] = active_rate[1]
#         rval['active_rate_15'] = active_rate[15]
#         rval['active_rate_30'] = active_rate[30]
#         rval['active_rate_45'] = active_rate[45]
#         rval['active_rate_50'] = active_rate[50]
#         rval['active_rate_mean'] = T.sum(active_rate) / self.dim
#  
        #rval['active_shape_0'] = self.active_rate.shape[0] * 1.
#         rval['active_shape_1'] = self.active_rate.shape[1] * 1.
#          
        #self.noise -= 10
         
        #rval['state_shape[1]'] = float(state.shape[1])
         
        rval['range_x_max_u'] = rg.max()
        rval['range_x_mean_u'] = rg.mean()
        rval['range_x_min_u'] = rg.min()
 
        rval['max_x_max_u'] = mx.max()
        rval['max_x_mean_u'] = mx.mean()
        rval['max_x_min_u'] = mx.min()
 
        rval['mean_x_max_u'] = mean.max()
        rval['mean_x_mean_u'] = mean.mean()
        rval['mean_x_min_u'] = mean.min()
 
        rval['min_x_max_u'] = mn.max()
        rval['min_x_mean_u'] = mn.mean()
        rval['min_x_min_u'] = mn.min()
 
        return rval
    
    
    