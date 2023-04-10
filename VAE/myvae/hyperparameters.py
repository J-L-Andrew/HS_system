import json
from collections import OrderedDict


def load_params(param_file=None, verbose=True):
    # Parameters from params.json and exp.json loaded here to override parameters set below
    if param_file is not None:
        hyper_p = json.loads(open(param_file).read(),
                             object_pairs_hook=OrderedDict)
        if verbose:
            print('Using hyper-parameters:')
            for key, value in hyper_p.items():
                print('{:25s} - {:12}'.format(key, str(value)))
            print('rest of parameters are set as default')
    parameters = {
        'radius': 0.5,
        
        # Continue training
        'model': None,

        # Model and Dataset
        'model_name': 'PointNetAutoEncoder',
        'dataset': 'lj_samples.npy',
        'non_center': 'store_true',
        
        # Model parameters
        'repeat_size': 8,
        'batch_size': 100,
        'cloud_size': 12,
        'latent_dim': 2,
        
        # Training parameters
        'epochs': 1000,
        'learning_rate': 0.001,
        
        # Loss parameters
        'l2_weight': 0.,
        'closs_weight': 0.001,
        'oloss_weight': 0.001,
        'rloss_weight': 0.1,
        'wloss_weight': 1.,
       
        # print output parameters
        "verbose_print": 0,
        
        if continue_training:
        MODEL = args_dict['model']
        old_args = json.load(open(f'model/{MODEL}/training/arguments.json'))
        args_dict.update(old_args)

    }
    # overwrite parameters
    parameters.update(hyper_p)
    return parameters
