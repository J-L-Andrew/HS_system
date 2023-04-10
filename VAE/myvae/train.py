import argparse
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from datetime import datetime
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
import yaml
import time

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from . import hyperparameters
from . import mol_utils as mu
from . import mol_callbacks as mol_cb
from keras.callbacks import CSVLogger
from models import Encoder, load_encoder
from models import Decoder, load_decoder
from models import property_predictor_model, load_property_predictor
from models import variational_layers
from functools import partial
from keras.layers import Lambda

import losses
import utils




def scheduler(epoch, lr):
    if epoch < 200:
        lr = lr + LEARNING_RATE / 200
    elif epoch < 1000:
        lr = lr * 0.999

    print('Adjust learning rate: ', lr)
    return lr

def load_models(params):

    def identity(x):
        return K.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = K.variable(params['kl_loss_weight'])

    if params['reload_model'] == True:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = Encoder(params)
        decoder = Decoder(params)

    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    # Decoder
    x_out = decoder(z_samp)

    x_out = Lambda(identity, name='x_pred')(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    return AE_only_model, encoder, decoder, kl_loss_var





def vectorize_data(params):
    # @out : Y_train /Y_test : each is list of datasets.
    #        i.e. if reg_tasks only : Y_train_reg = Y_train[0]
    #             if logit_tasks only : Y_train_logit = Y_train[0]
    #             if both reg and logit_tasks : Y_train_reg = Y_train[0], Y_train_reg = 1
    #             if no prop tasks : Y_train = []

    MAX_LEN = params['MAX_LEN']

    CHARS = yaml.safe_load(open(params['char_file']))
    params['NCHARS'] = len(CHARS)
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
    #INDICES_CHAR = dict((i, c) for i, c in enumerate(CHARS))

    ## Load data
    smiles = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN)

    if 'limit_data' in params.keys():
        sample_idx = np.random.choice(np.arange(len(smiles)), params['limit_data'], replace=False)
        smiles=list(np.array(smiles)[sample_idx])
        if params['do_prop_pred'] and ('data_file' in params):
            if "reg_prop_tasks" in params:
                Y_reg =  Y_reg[sample_idx]
            if "logit_prop_tasks" in params:
                Y_logit =  Y_logit[sample_idx]

    print('Training set size is', len(smiles))
    print('first smiles: \"', smiles[0], '\"')
    print('total chars:', NCHARS)

    print('Vectorization...')
    X = mu.smiles_to_hot(smiles, MAX_LEN, params[
                             'PADDING'], CHAR_INDICES, NCHARS)

    print('Total Data size', X.shape[0])
    if np.shape(X)[0] % params['batch_size'] != 0:
        X = X[:np.shape(X)[0] // params['batch_size']
              * params['batch_size']]
        if params['do_prop_pred']:
            if "reg_prop_tasks" in params:
                Y_reg = Y_reg[:np.shape(Y_reg)[0] // params['batch_size']
                      * params['batch_size']]
            if "logit_prop_tasks" in params:
                Y_logit = Y_logit[:np.shape(Y_logit)[0] // params['batch_size']
                      * params['batch_size']]

    np.random.seed(params['RAND_SEED'])
    rand_idx = np.arange(np.shape(X)[0])
    np.random.shuffle(rand_idx)

    TRAIN_FRAC = 1 - params['val_split']
    num_train = int(X.shape[0] * TRAIN_FRAC)

    if num_train % params['batch_size'] != 0:
        num_train = num_train // params['batch_size'] * \
            params['batch_size']

    train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train):]

    if 'test_idx_file' in params.keys():
        np.save(params['test_idx_file'], test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    print('shape of input vector : {}', np.shape(X_train))
    print('Training set size is {}, after filtering to max length of {}'.format(
        np.shape(X_train), MAX_LEN))

    return X_train, X_test


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    print('x_mean shape in kl_loss: ', x_mean.get_shape())
    kl_loss = - 0.5 * \
        K.mean(1 + x_log_var - K.square(x_mean) -
              K.exp(x_log_var), axis=-1)
    return kl_loss


if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser(description='Setting Training Parameters')
    
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    parser.add_argument('-d', '--directory',
                        help='exp directory', default=None)
    
    continue_training = parser.parse_args().model != None
    
    args_dict = vars(parser.parse_args())
    if args_dict['directory'] is not None:
        args_dict['exp_file'] = os.path.join(args_dict['directory'], args_dict['exp_file'])

    if continue_training:
        MODEL = args_dict['model']
        old_args = json.load(open(f'model/{MODEL}/training/arguments.json'))
        args_dict.update(old_args)
    
    # Make dirs
    if not continue_training:
        MODEL = 'C2_' + datetime.now().strftime('%m%d_%H%M') + '_' + str(os.getpid()) + '_' + parser.parse_args().dataset.split('.')[0] + f'_N{CLOUD_SIZE}' + f'_L{LATENT_DIM}'
        os.makedirs(f'model/{MODEL}/training')

    params = hyperparameters.load_params(args_dict['exp_file'])
    
    # Read data
    pointcloud_dataset = utils.PointCloudDataset(f'data/{DATASET}', CLOUD_SIZE, NON_CENTER)
    dataset = pointcloud_dataset.make_dataset()
    DATASET_SIZE, CLOUD_SIZE, DIMEN_SIZE = dataset.shape
    SAMPLE_SIZE = min(DATASET_SIZE, 10000)
    
    params = hyperparameters.load_params(args_dict['exp_file'])
    print("All params:", params)
    

    start_time = time.time()

    X_train, X_test = vectorize_data(params)
    model, encoder, decoder, kl_loss_var = load_models(params)

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_losses = {'x_pred': losses.CombinedReconstructionLoss(2*RADIUS, RLOSS_WEIGHT, WLOSS_WEIGHT),
                        'z_mean_log_var': kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)
    callbacks = [ vae_anneal_callback, csv_clb]
    
    
    # Callbacks
    if not continue_training:
        os.mkdir(f'model/{MODEL}/training/latestcheckpoint')
        os.mkdir(f'model/{MODEL}/training/bestcheckpoint')
    latestcheckpoint = callbacks.ModelCheckpoint(filepath=f'model/{MODEL}/' + 'training/latestcheckpoint/weights',
                                                save_best_only=False,
                                                verbose=1,
                                                save_weights_only=True)

    bestcheckpoint = callbacks.ModelCheckpoint(filepath=f'model/{MODEL}/' + 'training/bestcheckpoint/weights.{epoch:06d}-{loss:.6f}',
                                                save_best_only=True,
                                                monitor='loss',
                                                verbose=1,
                                                mode='min',
                                                save_weights_only=True)

    tensorboard = callbacks.TensorBoard(log_dir=f'model/{MODEL}/training/tensorboard',
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_images=True)

    csvlogger = callbacks.CSVLogger(filename=f'model/{MODEL}/training/logging.csv', append=True)


    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.), 
      loss=model_losses,
        loss_weights=[xent_loss_weight,
          kl_loss_var],
        metrics=[
                  losses.symmetric_chamfer_loss,
                  losses.symmetric_hausdorff_loss,
                  losses.weighted_chamfer_loss(np.inf if NON_CENTER else 2*RADIUS),
                  losses.weighted_hausdorff_loss(np.inf if NON_CENTER else 2*RADIUS)]
        )

    keras_verbose = params['verbose_print']
    
    # Train model
    print('[*] Start training ...')
    with open(f'model/{MODEL}/training/history.txt', 'a') as f:
        with redirect_stdout(f):
            history = model.fit(pointcloud_sequence,
                                epochs=EPOCHS,
                                verbose=2,
                                initial_epoch=initial_epoch,
                                callbacks=[
                                    latestcheckpoint,
                                    bestcheckpoint,
                                    # tensorboard,
                                    csvlogger,
                                    terminateonnan,
                                    lrscheduler
                                    ])

    AE_only_model.fit(X_train, model_train_targets,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    initial_epoch=params['prev_epochs'],
                    callbacks=callbacks,
                    verbose=keras_verbose,
                    validation_data=[ X_test, model_test_targets]
                    )

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')
    
    
