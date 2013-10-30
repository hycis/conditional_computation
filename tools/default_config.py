from jobman import DD


# TODO: left_slope is a hyperparam only used with rectifiedlinear
# DO NOT TOUCH THESE HYPERPARAMS from layer_config.
# These are the default hyperparams that will be used by each layer.
# If a layer (hidden or output) takes different hyperparam
# values than the ones in layer_config, then add these hyperparam
# values in model_config.layers
layer_config = {
    'layer_name'                    : None,
    'layer_class'                   : None,
    'dim'                           : None, # number of hidden units
    'istdev'                        : None,
    'sparse_init'                   : None, # TODO: to be used with relu and maxout
    'sparse_stdev'                  : 1.,
    'include_prob'                  : 1.0,
    'init_bias'                     : 0.,
    'max_col_norm'                  : None,
    'max_row_norm'                  : None,
    'dropout_probability'           : None,
    'dropout_scale'                 : None,
    'W_lr_scale'                    : None,
    'b_lr_scale'                    : None,
    'softmax_columns'               : False,
    'weight_decay'                  : None,
    'l1_weight_decay'               : None,
    'l2_weight_decay'               : None,
    'irange'                        : None,
    'use_bias'                      : True,
}

model_config = DD({
        # MLP
        'mlp' : DD({
            'model_class'                   : 'mlp',
            'train_class'                   : 'sgd',
            #'config_id'                     : 'GaussianNoise1000cifar200epoch',
            #'config_id'                     : 'Clean100cifar200epoch',
            #'config_id'                     : 'Clean100cifar200epochPreproc',
            #'config_id'                     : 'GaussianNoise1000cifar200epochPreproc',
            #'config_id'                     : 'GaussNoise2k-2kCifar200epochPreproc',
            #'config_id'                     : 'Noisy200-2kCifar200epochPreproc1',

            'config_id'                     : 'Clean200-200Cifar200epochPreproc1',



            # TODO: cached should always be True!
            'cached'                        : True,
            
            # dataset can be mnist or svhn or cifar10
            'dataset'                       : 'cifar10',
            
            'input_space_id'                : None,
            'nvis'                          : None,

            # Channel and dataset monitoring
            # mca : mean classification average of a minibatch
            #'channel_array'                 : ['mca'],
            'channel_array'                 : None,
            
            # valid or test or both
            'monitoring_dataset'           : ['valid'],

            'random_seed'                   : 251,
            'batch_size'                    : 200,
            'learning_rate'                 : ((1e-4, 1.0), float),
            'init_momentum'                 : ((0.5, 0.99), float),

            # for mnist
            #train_iteration_mode'          : 'random_uniform',
            # for svhn
            'train_iteration_mode'          : 'sequential',
            
            #<training modes>
            #sequential
            #shuffled_sequential
            #random_slice
            #random_uniform
            #batchwise_shuffled_sequential


            # Momentum and exponential decay
            'ext_array'                     : DD({
                'exp_decay' : DD({
                    'ext_class'             : 'exponentialdecayoverepoch',
                    'decay_factor'          : ((0.85, 0.999), float),


            # TODO: cached should always be True!
            'cached'                        : True,
            
            # dataset can be mnist or svhn or cifar10
            'dataset'                       : 'cifar10',
            
            'input_space_id'                : None,
            'nvis'                          : None,

            # Channel and dataset monitoring
            # mca : mean classification average of a minibatch
            #'channel_array'                 : ['mca'],
            'channel_array'                 : None,
            
            # valid or test or both
            'monitoring_dataset'           : ['valid'],

            'random_seed'                   : 251,
            'batch_size'                    : 200,
            'learning_rate'                 : ((1e-4, 1.0), float),
            'init_momentum'                 : ((0.5, 0.99), float),

            # for mnist
            #train_iteration_mode'          : 'random_uniform',
            # for svhn
            'train_iteration_mode'          : 'sequential',
            
            #<training modes>
            #sequential
            #shuffled_sequential
            #random_slice
            #random_uniform
            #batchwise_shuffled_sequential


            # Momentum and exponential decay
            'ext_array'                     : DD({
                'exp_decay' : DD({
                    'ext_class'             : 'exponentialdecayoverepoch',
                    'decay_factor'          : ((0.85, 0.999), float),
                    'min_lr_scale'          : ((1e-3, 1e-1), float),
                }),
                'moment_adj' : DD({
                    'ext_class'             : 'momentumadjustor',
                    'final_momentum'        : 0.9,
                    'start_epoch'           : 1,
                    'saturate_epoch'        : ((20, 50), int),
                }),
            }),

            # Termination criteria
            'term_array'                    : DD({
                # Max number of training epochs
                'epoch_count' : DD({
                    'term_class'            : 'epochcounter',
                    'max_epochs'            : 100,
                }),
                # Early stopping on validation set
                # If after max_epochs, we don't see significant improvement
                # on validation cost, we stop the training.
                'early_stopping' : DD({
                    'term_class'            : 'monitorbased',
                    'proportional_decrease' : 1e-4,
                    'max_epochs'            : 20,
                    'channel_name'          : 'valid_softmax2_nll',
                    'save_best_channel'     : 'valid_softmax2_nll',
                })
            }),

            'layers'                        : DD({
                # IMPORTANT: For each layer, only add hyperparams that are different than
                # the default hyperparams from layer_config

                # NOTE: always start the name of your hidden layers with hidden and your
                # output layers with output in order for the hidden layers
                # to be found first before the output layers when going
                # through the layers DD dictionary.

                # NOTE: the supported activation functions are:
                # tanh, sigmoid, rectifiedlinear, softmax

#                First hidden layer
#                 'hidden1' : DD({
#                     'layer_class'           : 'rectifiedlinear',
#                     #'dim'                   : ((100, 2000), int),
#                     'dim'                   : 200,
#                     'max_col_norm'          : ((0.1, 8.), float),
#                     #'weight_decay'          : ((0.1, 7.), float),
#                     'sparse_init'           : 15
#                 }),

#                 First hidden layer
                'hidden1' : DD({
                    'layer_class'           : 'tanh',
                    #'dim'                   : ((100, 2000), int),
                    'dim'                   : 200,
                    'max_col_norm'          : ((0.1, 5.), float)
                    #'weight_decay'          : ((1., 9.), float),
  
                }),
                
                 'hidden2' : DD({
                     'layer_class'           : 'tanh',
                     #'dim'                   : ((100, 2000), int),
                     'dim'                   : 200,
                     'max_col_norm'          : ((0.1, 5.), float)
                     #'weight_decay'          : ((1., 9.), float),
   
                 }),

#                 'hidden1' : DD({
#                     'layer_class'           : 'gaussianRELU',
#                     #'dim'                   : ((100, 2000), int),
#                     'dim'                   : 2000,
#                     'max_col_norm'          : ((0.1, 5.), float),
#                     'adjust_threshold_factor'   : ((0.0001, 1), float),
#                     'desired_active_rate'   : 0.1,
#                     'noise_std'             : ((0.1, 10), float),
#                      
#                     #'weight_decay'          : ((1., 9.), float),
#  
#                     'sparse_init'           : 15
#                 }),
#                                                   
#                'hidden2' : DD({
#                    'layer_class'           : 'gaussianRELU',
#                    #'dim'                   : ((100, 2000), int),
#                    'dim'                   : 2000,
#                    'max_col_norm'          : ((0.1, 5.), float),
#                    'adjust_threshold_factor'   : ((0.0001, 1), float),
#                    'desired_active_rate'   : 0.1,
#                    'noise_std'             : ((0.1, 10), float),
#                      
#                    #'weight_decay'          : ((1., 9.), float),
#  
#                    'sparse_init'           : 15
#                }),
                                              


                #First hidden layer
   
#                 'hidden1' : DD({
#                     'layer_class'           : 'noisyRELU',
#                     'sparse_init'           : 15,
#                     'dim'                   : 3000,
#                     'max_col_norm'          : ((0.1, 5.), float),
#                     'noise_factor'          : ((0.0001, 1.), float),
#                     'adjust_threshold_factor'   : ((0.0001, 1), float),
#                     'desired_active_rate'   : 0.1
#                     }),
# 		
                #Second hidden layer
#                 'hidden2' : DD({
#                     'layer_class'           : 'tanh',
#                     #'dim'                   : ((100, 2000), int),
#                     'dim'                   : 100,
#                     'max_col_norm'          : ((0.1, 5.), float)
#                     #'weight_decay'          : ((1., 9.), float),
#  
#                 }),


                # Last (output) layer
                # The fun model only takes 1 output.
                'output1' : DD({
                    'layer_class'           : 'softmax',
                    'dim'                   : 10,
                    'irange'                : 0.05
                    #'sparse_init'           : 15
                })
            }),
        }),
})