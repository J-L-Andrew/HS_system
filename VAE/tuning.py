import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import kerastuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, initializers, optimizers

import models
import utils

DATASET = 'sphere_packing'
REPEAT_SIZE = 8
BATCH_SIZE = 32
LATENT_DIM = 2
CLOUD_SIZE = 12
DIMEN_SIZE = 2

def model_builder(hp):
	hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
	# hp_chamfer_loss_weight = hp.Choice('chamfer_loss_weight', values=[1e0, 1e-1, 1e-2, 1e-3])
	hp_hausdorff_loss_weight = hp.Choice('hausdorff_loss_weight', values=[1e0, 1e-1, 1e-2, 1e-3])

	model = models.AutoEncoderV4(REPEAT_SIZE, BATCH_SIZE, LATENT_DIM, CLOUD_SIZE, DIMEN_SIZE)
	model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate), 
				  loss=[models.chamfer_loss, models.hausdorff_loss],
				  loss_weights=[1, hp_hausdorff_loss_weight],
				  metrics=[models.chamfer_loss, models.hausdorff_loss])

	return model

tuner = kt.Hyperband(model_builder,
					 objective=kt.Objective('chamfer_loss', direction='min'),
					 max_epochs=100,
					 hyperband_iterations=10,
					 factor=3,
					 directory='tuning',
					 project_name=DATASET)

# tuner = kt.BayesianOptimization(model_builder,
# 								objective=kt.Objective('chamfer_loss', direction='min'),
# 								max_trials=10,
# 								directory='tuning',
# 								project_name=DATASET)

centers, neighbors, pack_list = utils.read_packing('data/' + DATASET, CLOUD_SIZE)
DIMEN_SIZE = centers.shape[1]
DATASET_SIZE = len(neighbors)

dataset = tf.data.Dataset.from_tensor_slices(np.float32(neighbors))
dataset = dataset.shuffle(DATASET_SIZE).batch(BATCH_SIZE)
dataset = dataset.map(lambda x: ((x, [initializers.Orthogonal()((DIMEN_SIZE, DIMEN_SIZE)) for _ in range(REPEAT_SIZE)]), x))

stop_early = callbacks.EarlyStopping(monitor='chamfer_loss', patience=10)
tuner.search(dataset, epochs=100, verbose=2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
learning_rate: {best_hps.get('learning_rate')}
hausdorff_loss_weight: {best_hps.get('hausdorff_loss_weight')}
""")