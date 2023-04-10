import argparse
import math
import os
import shutil

from sklearn import metrics

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import json
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime

import hdbscan
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras import callbacks, initializers, optimizers

import losses
import models
import testing
import utils

# Parameters
parser = argparse.ArgumentParser(description='Setting Training Parameters')
parser.add_argument('-mn', '--model_name', type=str, default='PointNetAutoEncoder',
                    help='model name in application.py')
parser.add_argument('-ld', '--latent_dim', type=int, default=2,
                    help='latent dim')
parser.add_argument('-ds', '--dataset', type=str, default='sphere_packing_0.701597.xyz',
                    help='dataset in data floder')
parser.add_argument('-bs', '--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('-cs', '--cloud_size', type=int, default=12,
                    help='point cloud size')
parser.add_argument('-rs', '--repeat_size', type=int, default=8,
                    help='repeat size')
parser.add_argument('-ep', '--epochs', type=int, default=2000,
                    help='training epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('-cw', '--crc_weight', type=float, default=0.001,
                    help='cross rotational consistency loss weight')
parser.add_argument('-lw', '--l2_weight', type=float, default=0.000001,
                    help='cross rotational consistency loss weight')
args = parser.parse_args()

MODEL_NAME = args.model_name
LATENT_DIM = args.latent_dim
BATCH_SIZE = args.batch_size
CLOUD_SIZE = args.cloud_size
REPEAT_SIZE = args.repeat_size
DATASET = args.dataset
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
CRC_WEIGHT = args.crc_weight
L2_WEIGHT = args.l2_weight
RADIUS = 1 if DATASET[:8] != 'Baseline' else 0.5

# Make dirs
MODEL = 'B2_' + datetime.now().strftime('%m%d_%H%M') + '_' + str(os.getpid()) + '_' + DATASET.split('.')[0]
os.makedirs('model/' + MODEL + '/training')
# shutil.copy('network.py', 'model/' + MODEL + '/training/network.py')

# Read data
_, neighbors, pack_list = utils.read_packing('data/' + DATASET, CLOUD_SIZE)
DATASET_SIZE, CLOUD_SIZE, DIMEN_SIZE = neighbors.shape
DATASET_SIZE = len(neighbors)
SAMPLE_SIZE = min(DATASET_SIZE, 10000)

# Read data
class PointCloudSequence(tf.keras.utils.Sequence):

    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_x = (np.array(self.x[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]), np.array([initializers.Orthogonal()((DIMEN_SIZE, DIMEN_SIZE)) for _ in range(REPEAT_SIZE)]))
        batch_y = np.array(self.x[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE])

        return batch_x, batch_y

    def on_epoch_end(self):
        # self.x = []
        # for pc in self.point_cloud:
        #     for row in range(len(pc)):
        #         if np.random.rand() > 0.8:
        #             pc[row] = [0., 0., 0.]
        #     self.x.append(pc)
        self.x = np.array([[[-1.,-1.,-1.],
                  [-0.9,-0.9,-0.9],
                  [-0.8,-0.8,-0.8],
                  [-0.7,-0.7,-0.7],
                  [-0.6,-0.6,-0.6],
                  [-0.5,-0.5,-0.5],
                  [-0.4,-0.4,-0.4],
                  [-0.3,-0.3,-0.3],
                  [-0.2,-0.2,-0.2],
                  [-0.1,-0.1,-0.1],
                  [0.1,0.1,0.1],
                  [0.2,0.2,0.2],
                  [0.3,0.3,0.3],
                  [0.4,0.4,0.4],
                  [0.5,0.5,0.5],
                  [0.6,0.6,0.6],
                  [0.7,0.7,0.7],
                  [0.8,0.8,0.8],
                  [0.9,0.9,0.9],
                  [1.,1.,1.]]] * SAMPLE_SIZE)

            
dataset = PointCloudSequence(neighbors)

print('[*] Read {} samples with batch size {}.'.format(DATASET_SIZE, BATCH_SIZE))

# Build model

model = models.Transformer(REPEAT_SIZE, BATCH_SIZE, LATENT_DIM, CLOUD_SIZE, DIMEN_SIZE)
# model.compile(optimizer=optimizers.Adam(learning_rate=0.), 
#               loss=losses.SymmetricReconstructionLoss(),
#               metrics=[losses.symmetric_chamfer_loss, losses.symmetric_hausdorff_loss],
#               run_eagerly=False)
model.compile(optimizer=optimizers.Adam(learning_rate=0.), 
            #   loss=losses.CombinedReconstructionLoss(2*RADIUS),
            #   metrics=[losses.weighted_chamfer_loss(2*RADIUS), losses.weighted_hausdorff_loss(2*RADIUS)],
              run_eagerly=False)

print('[*] Build model successfully!')

# Arguments
with open('model/' + MODEL + '/training/arguments.json', 'w') as f:
    args_dict = vars(args)
    args_dict['dimen_size'] = DIMEN_SIZE
    args_dict['radius'] = RADIUS
    json.dump(args_dict, f, indent=4)

# Summary
with open('model/' + MODEL + '/training/summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.transformer.summary()

# Callbacks
os.makedirs('model/' + MODEL + '/training/checkpoint')
checkpoint = callbacks.ModelCheckpoint(filepath='model/' + MODEL + '/training/checkpoint/weights.{epoch:03d}-{loss:.6f}',
                                       save_best_only=False,
                                       monitor='loss',
                                       verbose=1,
                                       mode='min',
                                       save_weights_only=False)

tensorboard = callbacks.TensorBoard(log_dir='model/' + MODEL + '/training/tensorboard',
                                    histogram_freq=1,
                                    write_graph=True,
                                    write_images=True)

# earlystopping = callbacks.EarlyStopping(monitor='val_loss',
#                                         patience=100)

csvlogger = callbacks.CSVLogger(filename='model/' + MODEL + '/training/logging.csv')

def scheduler(epoch, lr):
    if epoch < 200:
        lr = lr + LEARNING_RATE / 200
    else:
        lr = lr * 0.999

    print('Adjust learning rate: ', lr)
    return lr

lrscheduler = callbacks.LearningRateScheduler(scheduler)

terminateonnan = callbacks.TerminateOnNaN()

crcweightscheduler = utils.CrcWeightScheduler()

# Fit model
with open('model/' + MODEL + '/training/history.txt', 'w') as f:
    with redirect_stdout(f):
        history = model.fit(dataset,
                            epochs=EPOCHS,
                            verbose=2,
                            # validation_data=(neighbors, neighbors),
                            # validation_batch_size=BATCH_SIZE,
                            callbacks=[checkpoint, tensorboard, csvlogger, terminateonnan, lrscheduler])

# Test model
testing.test(MODEL)
