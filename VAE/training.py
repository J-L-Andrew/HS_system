import argparse
import math
import os
import shutil
from importlib import import_module

from sklearn import metrics

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras import callbacks, initializers, optimizers

import losses
import testing
import utils

# Parameters
parser = argparse.ArgumentParser(description='Setting Training Parameters')
# Model and Dataset
parser.add_argument('-mn', '--model_name', type=str, default='PointNetAutoEncoder',
                    help='model name in network.py')
parser.add_argument('-ds', '--dataset', type=str, default='lj_samples.npy',
                    help='dataset in data floder')
parser.add_argument('-nc', '--non_center', action='store_true',
                    help='whether to find non-central structure')

# Model parameters
parser.add_argument('-rs', '--repeat_size', type=int, default=8,
                    help='repeat size')
parser.add_argument('-bs', '--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('-cs', '--cloud_size', type=int, default=12,
                    help='point cloud size')
parser.add_argument('-ld', '--latent_dim', type=int, default=2,
                    help='latent dim')

# Training parameters
parser.add_argument('-ep', '--epochs', type=int, default=1000,
                    help='training epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='learning rate')

# Loss parameters
parser.add_argument('-lw', '--l2_weight', type=float, default=0.,
                    help='l2 regularization weight')
parser.add_argument('-cw', '--closs_weight', type=float, default=0.001,
                    help='cross rotational consistency loss weight')
parser.add_argument('-ow', '--oloss_weight', type=float, default=0.001,
                    help='overlap loss weight')
parser.add_argument('-rw', '--rloss_weight', type=float, default=0.1,
                    help='reconstruction loss weight')
parser.add_argument('-ww', '--wloss_weight', type=float, default=1.,
                    help='weighted reconstruction loss weight')

# Continue training
parser.add_argument('-m', '--model', type=str,
                    help='continue training for new epochs')

continue_training = parser.parse_args().model != None
args_dict = vars(parser.parse_args())

if continue_training:
    MODEL = args_dict['model']
    old_args = json.load(open(f'model/{MODEL}/training/arguments.json'))
    args_dict.update(old_args)

MODEL_NAME = args_dict['model_name']
DATASET = args_dict['dataset']
NON_CENTER = args_dict['non_center']

REPEAT_SIZE = args_dict['repeat_size']
BATCH_SIZE = args_dict['batch_size']
CLOUD_SIZE = args_dict['cloud_size']
LATENT_DIM = args_dict['latent_dim']

EPOCHS = args_dict['epochs']
LEARNING_RATE = args_dict['learning_rate']

L2_WEIGHT = args_dict['l2_weight']
CLOSS_WEIGHT = args_dict['closs_weight']
OLOSS_WEIGHT = args_dict['oloss_weight']
RLOSS_WEIGHT = args_dict['rloss_weight']
WLOSS_WEIGHT = args_dict['wloss_weight']

# RADIUS = 1 if DATASET[:2] != 'lj' else 0.5
RADIUS = 0.5

# Make dirs
if not continue_training:
    MODEL = 'C2_' + datetime.now().strftime('%m%d_%H%M') + '_' + str(os.getpid()) + '_' + parser.parse_args().dataset.split('.')[0] + f'_N{CLOUD_SIZE}' + f'_L{LATENT_DIM}'
    os.makedirs(f'model/{MODEL}/training')

# Read data
pointcloud_dataset = utils.PointCloudDataset(f'data/{DATASET}', CLOUD_SIZE, NON_CENTER)
dataset = pointcloud_dataset.make_dataset()
DATASET_SIZE, CLOUD_SIZE, DIMEN_SIZE = dataset.shape
SAMPLE_SIZE = min(DATASET_SIZE, 10000)

class PointCloudSequence(tf.keras.utils.Sequence):

    def __init__(self, dataset):
        self.dataset = dataset
        self.epochs = 0
        self.sample_weight = [1] * SAMPLE_SIZE
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_x = (np.array(self.x[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]), 
                   np.array([initializers.Orthogonal()((DIMEN_SIZE, DIMEN_SIZE)) for _ in range(REPEAT_SIZE)]))                   
        batch_y = np.array([self.x[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE] for _ in range(REPEAT_SIZE)])
        batch_s = np.array(self.sample_weight[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE])

        return batch_x, batch_y, batch_s

    def on_epoch_end(self):
        self.epochs += 1
        # np.random.seed(2021)
        slice = np.random.choice(DATASET_SIZE, size=SAMPLE_SIZE, replace=False)
        self.x = dataset[slice]

        # if self.epochs >= 200:
        #     model = eval('models.{}({}, {}, {}, {}, {})'.format(MODEL_NAME, REPEAT_SIZE, BATCH_SIZE, LATENT_DIM, CLOUD_SIZE, DIMEN_SIZE))
        #     model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss=models.PointCloudLoss())
        #     checkpoint = tf.train.Checkpoint(model)
        #     save_path = tf.train.latest_checkpoint(f'model/{MODEL}/training/checkpoint')
        #     checkpoint.restore(save_path).expect_partial()
        #     encoded = model.encoder.predict(self.x, batch_size=BATCH_SIZE)

        #     clusterer = hdbscan.HDBSCAN(min_cluster_size=SAMPLE_SIZE // 10)
        #     clusterer.fit(encoded)
        #     labels = clusterer.labels_ + 1
        #     count = np.bincount(labels)

        #     # if len(count) == 1:
        #     #     self.sample_weight = [1] * SAMPLE_SIZE
        #     #     return

        #     self.sample_weight = compute_sample_weight('balanced', y=labels)
        #     # for i in range(len(self.sample_weight)):
        #     #     if labels[i] == 0:
        #     #         self.sample_weight[i] = 0
        #     print('Setting sample weight', Counter(self.sample_weight))
            
pointcloud_sequence = PointCloudSequence(dataset)

print(f'[*] Read {DATASET_SIZE} samples with batch size {BATCH_SIZE}.')

# Arguments
with open(f'model/{MODEL}/training/arguments.json', 'w') as f:
    args_dict.pop('model')
    args_dict['dimen_size'] = DIMEN_SIZE
    args_dict['radius'] = RADIUS
    json.dump(args_dict, f, indent=4)

# Build model
if continue_training:
    df = pd.read_csv(f'model/{MODEL}/training/logging.csv')
    initial_epoch = len(df)

    network = import_module(f'model.{MODEL}.training.network')
    model = network.PointNetAutoEncoder(REPEAT_SIZE, BATCH_SIZE, CLOUD_SIZE, DIMEN_SIZE, LATENT_DIM, L2_WEIGHT, CLOSS_WEIGHT, OLOSS_WEIGHT, RADIUS)
    checkpoint = tf.train.Checkpoint(model)
    latest_ckpt = tf.train.latest_checkpoint(f'model/{MODEL}/training/checkpoint')
    checkpoint.restore(latest_ckpt).expect_partial()
    print(f'[*] Continue training {MODEL} from {initial_epoch} epochs')

else:
    shutil.copy('network.py', f'model/{MODEL}/training/network.py')
    initial_epoch = 0

    network = import_module(f'model.{MODEL}.training.network')
    model = network.PointNetAutoEncoder(REPEAT_SIZE, BATCH_SIZE, CLOUD_SIZE, DIMEN_SIZE, LATENT_DIM, L2_WEIGHT, CLOSS_WEIGHT, OLOSS_WEIGHT, RADIUS)
    print(f'[*] Build model {MODEL} successfully!')

    # Summary
    with open(f'model/{MODEL}/training/summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.transformer.summary()
            model.encoder.summary()
            model.decoder.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=0.), 
              loss=losses.CombinedReconstructionLoss(2*RADIUS, RLOSS_WEIGHT, WLOSS_WEIGHT),
              metrics=[
                  losses.symmetric_chamfer_loss,
                  losses.symmetric_hausdorff_loss,
                  losses.weighted_chamfer_loss(np.inf if NON_CENTER else 2*RADIUS),
                  losses.weighted_hausdorff_loss(np.inf if NON_CENTER else 2*RADIUS)],
              run_eagerly=False)

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

def scheduler(epoch, lr):
    if epoch < 200:
        lr = lr + LEARNING_RATE / 200
    elif epoch < 1000:
        lr = lr * 0.999

    print('Adjust learning rate: ', lr)
    return lr

lrscheduler = callbacks.LearningRateScheduler(scheduler)

terminateonnan = callbacks.TerminateOnNaN()

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

# Test model
del model
testing.test(MODEL)