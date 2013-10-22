from pylearn2.datasets.svhn import SVHN
from pylearn2.expr.preprocessing import global_contrast_normalize
import cPickle, logging, os
_logger = logging.getLogger(__name__)

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
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.expr.preprocessing import global_contrast_normalize


class My_CIFAR10(dense_design_matrix.DenseDesignMatrix):
    
    def __init__(self, which_set, center = False, rescale = False, gcn = None,
            one_hot = False, start = None, stop = None, axes=('b', 0, 1, 'c'),
            toronto_prepro = False, preprocessor = None):


        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)

        self.axes = axes

        # we define here:
        dtype  = 'uint8'
        ntrain = 50000
        nvalid = 0  # artefact, we won't use it
        ntest  = 10000

        # we also expose the following details:
        self.img_shape = (3,32,32)
        self.img_size = np.prod(self.img_shape)
        self.n_classes = 10
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog','horse','ship','truck']

#         # prepare loading
#         fnames = ['data_batch_%i' % i for i in range(1,6)]
#         lenx = np.ceil((ntrain + nvalid) / 10000.)*10000
#         x = np.zeros((lenx,self.img_size), dtype=dtype)
#         y = np.zeros(lenx, dtype=dtype)
# 
#         # load train data
#         nloaded = 0
#         for i, fname in enumerate(fnames):
#             data = CIFAR10._unpickle(fname)
#             x[i*10000:(i+1)*10000, :] = data['data']
#             y[i*10000:(i+1)*10000] = data['labels']
#             nloaded += 10000
#             if nloaded >= ntrain + nvalid + ntest: break;
# 
#         # load test data
#         data = CIFAR10._unpickle('test_batch')
# 
#         # process this data
#         Xs = {
#                 'train' : x[0:ntrain],
#                 'test'  : data['data'][0:ntest]
#             }
# 
#         Ys = {
#                 'train' : y[0:ntrain],
#                 'test'  : data['labels'][0:ntest]
#             }

        if which_set == 'train':
            
#             pkl = self._unpickle(os.environ['PYLEARN2_DATA_PATH']+
#                                  'cifar10/pylearn2_gcn_whitened/train.pkl')
            pkl = self._unpickle(os.environ['PYLEARN2_DATA_PATH']+
                                 'cifar10/pylearn2_gcn_whitened/test.pkl')
            X = pkl.X
            y = pkl.y
        
        elif which_set == 'test':
            pkl = self._unpickle(os.environ['PYLEARN2_DATA_PATH']+
                                 'cifar10/pylearn2_gcn_whitened/test.pkl')
            X = pkl.X
            y = pkl.y
            
            
#         X = np.cast['float32'](Xs[which_set])
#         y = Ys[which_set]

        if isinstance(y,list):
            y = np.asarray(y)

        if which_set == 'test':
            assert y.shape[0] == 10000

        if center:
            X -= 127.5
        self.center = center

        if rescale:
            X /= 127.5
        self.rescale = rescale

        if toronto_prepro:
            assert not center
            assert not gcn
            X = X / 255.
            if which_set == 'test':
                other = CIFAR10(which_set='train')
                oX = other.X
                oX /= 255.
                X = X - oX.mean(axis=0)
            else:
                X = X - X.mean(axis=0)
        self.toronto_prepro = toronto_prepro

        self.gcn = gcn
        if gcn is not None:
            gcn = float(gcn)
            X = global_contrast_normalize(X, scale=gcn)

        if start is not None:
            # This needs to come after the prepro so that it doesn't change the pixel
            # means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop]
            assert X.shape[0] == y.shape[0]

        if which_set == 'test':
            assert X.shape[0] == 10000

        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3), axes)
        
        if which_set == 'train':
            length = X.shape[0]
            def search_right_label(desired_label, i):
                for idx in xrange(i, length):
                    if y[idx] == desired_label:
                        return idx
            
            def swap_ele(index, i):
                x_tmp = X[i]
                X[i] = X[index]
                X[index] = x_tmp
                
                y_tmp = y[i]
                y[i] = y[index]
                y[index] = y_tmp
                
            desired_label = 0
            for i in xrange(length):
                desired_label = i % 10
                if y[i] != desired_label:
                    index = search_right_label(desired_label, i)
                    swap_ele(index, i)
            
            for i in xrange(length-100, length):
                print y[i]
                        
        self.one_hot = one_hot
        if one_hot:
            one_hot = np.zeros((y.shape[0],10),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot
        
        super(My_CIFAR10,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not np.any(np.isnan(self.X))

        if preprocessor:
            preprocessor.apply(self)
    

    def adjust_for_viewer(self, X):
        #assumes no preprocessing. need to make preprocessors mark the new ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i,:] /= np.abs(rval[i,:]).max()
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval

    def adjust_to_be_viewed_with(self, X, orig, per_example = False):
        # if the scale is set based on the data, display X oring the scale determined
        # by orig
        # assumes no preprocessing. need to make preprocessors mark the new ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i,:] /= np.abs(orig[i,:]).max()
            else:
                rval /= np.abs(orig).max()
            rval = np.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval

    def get_test_set(self):
        return CIFAR10(which_set='test', center=self.center, rescale=self.rescale, gcn=self.gcn,
                one_hot=self.one_hot, toronto_prepro=self.toronto_prepro, axes=self.axes)


    @classmethod
    def _unpickle(cls, file):
        """
        TODO: wtf is this? why not just use serial.load like the CIFAR-100 class?
        whoever wrote it shows up as "unknown" in git blame
        """
#         from pylearn2.utils import string_utils
#         fname = os.path.join(
#                 string_utils.preprocess('${PYLEARN2_DATA_PATH}'),
#                 'cifar10',
#                 'cifar-10-batches-py',
#                 file)
#         if not os.path.exists(fname):
#             raise IOError(fname+" was not found. You probably need to download "
#                     " the CIFAR-10 dataset from http://www.cs.utoronto.ca/~kriz/cifar.html")
#         _logger.info('loading file %s' % fname)
#         fo = open(fname, 'rb')
#         dict = cPickle.load(fo)
#         fo.close()
#         return dict
        #import pdb
        #pdb.set_trace()
        with open(file) as f:
            import pdb
            pdb.set_trace()
            pk = cPickle.load(f)
        _logger.info('loading file.. %s' % file)
        return pk



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
            for i in xrange(10):
                count[str(i)] = 0
            for ele in data.y:
                index = np.argmax(ele, axis=0)
                #print index
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
                import pdb
                pdb.set_trace()
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
    
    