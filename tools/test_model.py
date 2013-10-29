import theano
from theano import tensor as T, function
from pylearn2.utils import serial

from dataset import My_CIFAR10
import sys
import numpy as np
    
    
def test_model(dataset, model_path):
    
    data_X = dataset.X
    data_y = dataset.y
    import pdb
    pdb.set_trace()
    
    try:
        model = serial.load(model_path)
    except Exception, e:
        print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
        print e
        sys.exit()

    # Compile theano function for computing the model predictions.
    #X = model.get_input_space().make_batch_theano()
    X = T.dmatrix()
    Y = model.fprop(X)
    fprop = function([X], [Y])
    input_ndarray = np.asarray(data_X, dtype='float32')
    # predict outputs
    predictions = np.asarray(fprop(input_ndarray), dtype='float32').T
    predictions = predictions.reshape((len(predictions),))
    
    
    print predictions
    
if __name__ == '__main__':
    dataset = My_CIFAR10(which_set='test')
    model_path = '/data/lisa/exp/wuzhen/test/conditional_computation/mlp/jobman_20131026_182820_36582050/model_Noisy200-2kCifar200epochPreproc_optimum.pkl'
    test_model(dataset, model_path)