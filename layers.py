from pylearn2.models.mlp import Linear
import numpy as np
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX
from theano import config


class NoisyRELU(Linear):

    def __init__(self, noise_factor=1, activation_rate=0.1, **kwargs):
        super(NoisyRELU, self).__init__(**kwargs)
        self.noise_factor = noise_factor
        self.activation_threshold = T.matrix(dtype=config.floatX)
        self.activation_rate = activation_rate


    def fprop(self, state_below):
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
        p = self._linear_part(state_below) + self.noise * self.noise_factor - self.activation_threshold
        #print 'fprop p', p.eval()
        p = T.maximum(0., p)
        
        for i in xrange(self.dim):
            active_rate = T.sum(T.neq(p[i][:], 0), dtype=config.floatX) / self.mlp.batch_size
            if active_rate < self.activation_rate:
                self.activation_threshold[:][i] -= T.abs_(self.activation_rate - active_rate) * 0.1
            else:
                self.activation_threshold[:][i] += T.abs_(active_rate - self.activation_rate) * 0.1
                
        return p

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
        
        
        return OrderedDict([
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
        rval =  OrderedDict()

        mx = state.max(axis=0)
        mean = state.mean(axis=0)
        mn = state.min(axis=0)
        rg = mx - mn
        
        active_rate = []
        for i in xrange(self.dim):
            active_rate.append(T.sum(T.neq(state[:][i], 0), dtype=config.floatX) / (state.shape[0]))

        
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