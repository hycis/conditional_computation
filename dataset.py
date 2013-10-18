from pylearn2.datasets.svhn import SVNH

import os
import gc
import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far SVHN is "
            "only supported with PyTables")
import numpy as np
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess


class My_SVHN(SVHN):
    
    mapper = {'train': 0, 'test': 1, 'extra': 2, 'train_all': 3,
                'splitted_train': 4, 'valid': 5}

    data_path = '${PYLEARN2_DATA_PATH}/SVHN/format2/'

    def __init__(self, which_set, path = None, center = False, scale = False,
            start = None, stop = None, axes = ('b', 0, 1, 'c'),
            preprocessor = None):
        """
        Only for faster access there is a copy of hdf5 file in
        PYLEARN2_DATA_PATH but it mean to be only readable.
        If you wish to modify the data, you should pass a local copy
        to the path argument.
        """

        assert which_set in self.mapper.keys()

        self.__dict__.update(locals())
        del self.self

        if path is None:
            path = '${PYLEARN2_DATA_PATH}/SVHN/format2/'
            mode = 'r'
        else:
            mode = 'r+'

        if mode == 'r' and (scale or center or (start != None) or
                        (stop != None)):
            raise ValueError("Only for speed there is a copy of hdf5 " +\
                    "file in PYLEARN2_DATA_PATH but it meant to be only " +\
                    "readable. If you wish to modify the data, you should " +\
                    "pass a local copy to the path argument.")

        # load data
        path = preprocess(path)
        file_n = "{}{}_32x32.h5".format(path + "h5/", which_set)
        if os.path.isfile(file_n):
            make_new = False
        else:
            make_new = True
            warnings.warn("Over riding existing file: {}".format(file_n))

        # if hdf5 file does not exist make them
        if make_new:
            self.make_data(which_set, path)

        self.h5file = tables.openFile(file_n, mode = mode)
        data = self.h5file.getNode('/', "Data")

        if start != None or stop != None:
            self.h5file, data = self.resize(self.h5file, start, stop)

        # rescale or center if permitted
        if center and scale:
            data.X[:] -= 127.5
            data.X[:] /= 127.5
        elif center:
            data.X[:] -= 127.5
        elif scale:
            data.X[:] /= 255.

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3), axes)
        
        
        if which_set is 'splitted_train':
            count = {}
            for ele in data.y:
                index = np.argmax(ele, axis=0)
                count[str(index)] += 1
            min = float('Inf')
            for val in count.itervalues():
                if val < min:
                    min = val
            
            
            
            def search(start_index, label):
                for i in xrange(start_index, len(data.y)):
                    if label == np.argmax(data.y[i]):
                        return i
            
            cutoff = min * 10
            
            for i in xrange(cutoff):
                
                label = np.argmax(data.y[i])
                index_label = i % 10
                
                if label != index_label:
                    index = search(i, label)
                    tmp = data.y[index]
                    data.y[index] = data.y[i]
                    data.y[i] = tmp
                    
                    tmp2 = data.X[index]
                    data.X[index] = data.X[i]
                    data.X[i] = tmp2
            data.X = data.X[:cutoff]
            data.y = data.y[:cutoff]
                
                    
        
        super(SVHN, self).__init__(X = data.X, y = data.y,
                                    view_converter = view_converter)

        if preprocessor:
            if which_set in ['train', 'train_all', 'splitted_train']:
                can_fit = True
            preprocessor.apply(self, can_fit)

        self.h5file.flush()
    
    