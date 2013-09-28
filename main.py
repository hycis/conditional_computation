


import pdb
import os
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.mlp import MLP, Softmax, RectifiedLinear
from pylearn2.space import VectorSpace
from pylearn2.costs.cost import SumOfCosts, MethodCost
from pylearn2.costs.mlp import WeightDecay
from pylearn2.training_algorithms.sgd import SGD, MomentumAdjustor
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.train import Train
from pylearn2.datasets.svhn import SVHN_On_Memory

from layers import NoisyRELU

def model1():
    #pdb.set_trace()
    # train set X has dim (60,000, 784), y has dim (60,000, 10)
    train_set = MNIST(which_set='train', one_hot=True)
    # test set X has dim (10,000, 784), y has dim (10,000, 10)
    valid_set = MNIST(which_set='test', one_hot=True)
    test_set = MNIST(which_set='test', one_hot=True)
    
    #import pdb
    #pdb.set_trace()
    #print train_set.X.shape[1]
    
    # =====<Create the MLP Model>=====

    h2_layer = NoisyRELU(layer_name='h1', sparse_init=50, dim=100, max_col_norm=100)
    #h2_layer = RectifiedLinear(layer_name='h2', dim=100, sparse_init=15, max_col_norm=1)
    #print h1_layer.get_params()
    #h2 = RectifiedLinear(layer_name='h2', dim=500, sparse_init=15, max_col_norm=1)
    y_layer = Softmax(layer_name='y', n_classes=10, irange=0., max_col_norm=1)
    
    mlp = MLP(batch_size = 100,
                input_space = VectorSpace(dim=train_set.X.shape[1]),
                layers = [h2_layer, y_layer])
    
    # =====<Create the SGD algorithm>=====
    sgd = SGD(init_momentum = 0.1, 
                    learning_rate = 0.01, 
                    monitoring_dataset = {'valid' : valid_set, 'test' : test_set},
                    cost = MethodCost('cost_from_X'),
                    termination_criterion = MonitorBased(channel_name='valid_y_misclass',
                                                        prop_decrease=0.001, N=50))
    #sgd.setup(model=mlp, dataset=train_set)
    
    # =====<Extensions>=====
    ext = [MomentumAdjustor(start=1, saturate=10, final_momentum=0.9)]
    
    # =====<Create Training Object>=====
    save_path = './mlp_model3.pkl'
    train_obj = Train(dataset=train_set, model=mlp, algorithm=sgd, 
                      extensions=ext, save_path=save_path, save_freq=10)
    #train_obj.setup_extensions()
    
    #import pdb
    #pdb.set_trace()
    train_obj.main_loop()
    
    # =====<Run the training>=====
    '''
    while True:
        rval = train_obj.algorithm.train(train_set)
        assert rval is None
        train_obj.run_callbacks_and_monitoring()
        
        if train_obj.save_freq > 0 and \
        train_obj.model._epochs_seen % train_obj.save_freq == 0:
            train_obj.save()
        
        continue_learning = train_obj.algorithm.continue_learning(train_obj.model)
        assert continue_learning in [True, False, 0, 1]
        
        if not continue_learning:
            break
    
    if train_obj.save_freq > 0:
        train_obj.save()
    '''

def model2():
    #pdb.set_trace()
    # train set X has dim (60,000, 784), y has dim (60,000, 10)
    train_set = MNIST(which_set='train', one_hot=True)
    # test set X has dim (10,000, 784), y has dim (10,000, 10)
    test_set = MNIST(which_set='test', one_hot=True)
    
    # =====<Create the MLP Model>=====

    h1_layer = RectifiedLinear(layer_name='h1', dim=1000, irange=0.5)
    #print h1_layer.get_params()
    h2_layer = RectifiedLinear(layer_name='h2', dim=1000, sparse_init=15, max_col_norm=1)
    y_layer = Softmax(layer_name='y', n_classes=train_set.y.shape[1], irange=0.5)
    
    mlp = MLP(batch_size = 100,
                input_space = VectorSpace(dim=train_set.X.shape[1]),
                layers = [h1_layer, h2_layer, y_layer])
    
    # =====<Create the SGD algorithm>=====
    sgd = SGD(batch_size = 100, init_momentum = 0.1, 
                    learning_rate = 0.01, 
                    monitoring_dataset = {'valid' : train_set, 'test' : test_set},
                    cost = SumOfCosts(costs=[MethodCost('cost_from_X'), 
                             WeightDecay(coeffs=[0.00005, 0.00005, 0.00005])]),
                    termination_criterion = MonitorBased(channel_name='valid_y_misclass',
                                                        prop_decrease=0.0001, N=5))
    #sgd.setup(model=mlp, dataset=train_set)
    
    # =====<Extensions>=====
    ext = [MomentumAdjustor(start=1, saturate=10, final_momentum=0.99)]
    
    # =====<Create Training Object>=====
    save_path = './mlp_model2.pkl'
    train_obj = Train(dataset=train_set, model=mlp, algorithm=sgd, 
                      extensions=ext, save_path=save_path, save_freq=0)
    #train_obj.setup_extensions()
    
    train_obj.main_loop()
    
def model3():
    #pdb.set_trace()
    # train set X has dim (60,000, 784), y has dim (60,000, 10)
    train_set = SVHN_On_Memory(which_set='train')
    # test set X has dim (10,000, 784), y has dim (10,000, 10)
    test_set = SVHN_On_Memory(which_set='test')
    
    # =====<Create the MLP Model>=====

    h1_layer = NoisyRELU(layer_name='h1', dim=100, threshold=5, sparse_init=15, max_col_norm=1)
    #print h1_layer.get_params()
    #h2_layer = NoisyRELU(layer_name='h2', dim=100, threshold=15, sparse_init=15, max_col_norm=1)
    
    y_layer = Softmax(layer_name='y', n_classes=train_set.y.shape[1], irange=0.5)
    
    mlp = MLP(batch_size = 64,
                input_space = VectorSpace(dim=train_set.X.shape[1]),
                layers = [h1_layer, y_layer])
    
    # =====<Create the SGD algorithm>=====
    sgd = SGD(batch_size = 64, init_momentum = 0.1, 
                    learning_rate = 0.01, 
                    monitoring_dataset = {'valid' : train_set, 'test' : test_set},
                    cost = MethodCost('cost_from_X'),
                    termination_criterion = MonitorBased(channel_name='valid_y_misclass',
                                                        prop_decrease=0.001, N=50))
    #sgd.setup(model=mlp, dataset=train_set)
    
    # =====<Extensions>=====
    ext = [MomentumAdjustor(start=1, saturate=10, final_momentum=0.9)]
    
    # =====<Create Training Object>=====
    save_path = './mlp_model.pkl'
    train_obj = Train(dataset=train_set, model=mlp, algorithm=sgd, 
                      extensions=ext, save_path=save_path, save_freq=10)
    #train_obj.setup_extensions()
    
    train_obj.main_loop()

if __name__ == '__main__':
    #os.environ['PYLEARN2_DATA_PATH'] = '/Users/zhenzhou/Desktop/pylearn2/data'
    #model2()
    model1()