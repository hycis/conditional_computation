from pylearn2.models.mlp import Linear, MLP
import numpy as np
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX
import theano

class My_MLP(MLP):
    def __init__(self, **kwargs):
        super(My_MLP, self).__init__(**kwargs)
    
    def censor_updates_inputs(self, updates, theano_args):
        for layer in self.layers:
            layer.censor_updates_inputs(self, updates, theano_args)


class NoisyRELU(Linear):

    def __init__(self, noise_factor=1, desired_active_rate=0.1, bias_factor=0.1, **kwargs):
        super(NoisyRELU, self).__init__(**kwargs)
        self.noise_factor = noise_factor
        self.bias_factor = bias_factor
        #self.activation_threshold = T.matrix(dtype=config.floatX)
        self.desired_active_rate = desired_active_rate
        #self.active_rate = T.vector()


    def fprop(self, state_below):
        print "======fprop====="
        
        #noise = sharedX(np.zeros(self.dim))
        rng = RandomStreams(seed=234)
        #print '=====self.dim====', self.dim
        #import pdb
        #pdb.set_trace()
        
        un = rng.uniform(size=(self.mlp.batch_size, self.dim), low=0., high=1.)
        #print '=====frop===='
        #print '=======statebelow=========', 
        #self.s = state_below.shape
        
        
        
        self.noise = T.log(un/(1-un))
        p = self._linear_part(state_below) + self.noise * self.noise_factor
        
        p = T.maximum(0., p)
        num_example = p.shape[1]
        self.active_rate, updates = theano.scan(fn=lambda example : T.gt(example,0).sum() * 1. / num_example,
                                     sequences = [p])
        #import pdb
        #pdb.set_trace()
        return p
        
    
    def censor_updates(self, updates):
        print "====censor_updates====="
        
        #import pdb
        #pdb.set_trace()
        W, b = self.get_params()

        if self.mask_weights is not None:
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_row_norm is not None:
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x')

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * desired_norms / (1e-7 + col_norms)
        
        def update_bias_elemwise(b_value_for_one_neuron, active_rate):
            if T.gt(active_rate, self.desired_active_rate):
                rval = b_value_for_one_neuron - (active_rate - self.desired_active_rate) * self.bias_factor
            else:
                rval = b_value_for_one_neuron + (self.desired_active_rate - active_rate) * self.bias_factor
            #import pdb
            #pdb.set_trace()
            return rval
        
        assert b in updates
        
        updates_b = updates[b]
        updates[b] = theano.scan(update_bias_elemwise, sequences=[updates_b, self.active_rate])[0]
        #import pdb
        #pdb.set_trace()
        #updates[b] = updates_b
            
        
        
#         assert b in updates
#         
#         if T.gt(self.desired_active_rate, self.active_rate):
#             assert b in updates
#             values, updates = scan(lambda : {example, (self.desired_active_rate - example) * self.bias_factor \
#                                              if T.gt(self.desired_active_rate, example) else })
#             updates[b] += (self.desired_active_rate - self.active_rate) * self.bias_factor
#         
#         elif T.lt(self.desired_active_rate, self.active_rates):
#             assert b in updates
#             updates[b] -= (self.desired_active_rate - self.active_rate) * self.bias_factor
            
            
    def cost(self, *args, **kwargs):
        raise NotImplementedError()
     
    def get_monitoring_channels(self):
 
        W ,= self.transformer.get_params()
 
        assert W.ndim == 2
 
        sq_W = T.sqr(W)
 
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
         
        max_noise = self.noise.max()
        min_noise = self.noise.min()
        mean_noise = self.noise.mean()
        max_active_rate = self.active_rate.max()
        min_active_rate = self.active_rate.min()
        mean_active_rate = self.active_rate.mean()
         
        print 'get_monitoring ===== '
         
        return OrderedDict([
                            #('=====max_active_rate', max_active_rate),
                            #('=====mean_active_rate', mean_active_rate),
                            #('=====min_active_rate', min_active_rate),
                            ('=====max_noise=====', max_noise),
                            ('=====mean_noise=====', mean_noise),
                            ('=====min_noise=====', min_noise),
                            ('w_shape_0', W.shape[0] * 1.),
                            ('w_shape_1', W.shape[1] * 1.),
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
         
        active_rate = []
        for i in xrange(self.dim):
            active_rate.append(T.sum(T.neq(state[:][i], 0), dtype=theano.config.floatX) / (state.shape[0]))
 
        max_active_rate = self.active_rate.max()
        min_active_rate = self.active_rate.min()
        mean_active_rate = self.active_rate.mean()
        num_row = self.active_rate.shape[0] * 1.
        #num_col = self.active_rate.shape[1] * 1.
        rval['===max_active_rate===='] = max_active_rate
        rval['===min_active_rate===='] = min_active_rate
        rval['===mean_active_rate===='] = mean_active_rate
        rval['===num_row_active_rate===='] = num_row
        #rval['===num_col_active_rate===='] = num_col
        
        rval['active_rate_cal_1'] = self.active_rate[1]
        rval['active_rate_1'] = active_rate[1]
        rval['active_rate_15'] = active_rate[15]
        rval['active_rate_30'] = active_rate[15]
 
 
        rval['state_shape_0'] = state.shape[0] * 1.
        rval['state_shape_1'] = state.shape[1] * 1.
         
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