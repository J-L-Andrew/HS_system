import numpy as np
import tensorflow as tf
from scipy.stats import special_ortho_group as sog
from tensorflow.keras import Model, Sequential, layers, initializers, regularizers, losses
from tensorflow_graphics.geometry import transformation
from tensorflow_graphics.geometry.transformation.quaternion import multiply
from tensorflow_graphics.nn import loss

class AutoEncoder(Model):
	def __init__(self, latent_dim, cloud_size):
		super(AutoEncoder, self).__init__()
		self.latent_dim = latent_dim
		self.cloud_size = cloud_size

		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,3)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=32, kernel_size=(1,3), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256, activation='relu'),
			layers.BatchNormalization(),

			layers.Dense(units=128, activation='relu'),
			layers.BatchNormalization(),

			layers.Dense(units=4, activation='linear')
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,3)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=32, kernel_size=(1,3), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=32, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256, activation='relu'),
			layers.BatchNormalization(),

			layers.Dense(units=128, activation='relu'),
			layers.BatchNormalization(),

			layers.Dense(units=self.latent_dim, activation='linear')
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=32, activation='relu'),
			layers.BatchNormalization(),

			layers.Dense(units=64, activation='relu'),
			layers.BatchNormalization(),

			layers.Dense(units=128, activation='relu'),
			layers.BatchNormalization(),

			layers.Dense(units=self.cloud_size * 3, activation='linear'),

			layers.Reshape((self.cloud_size, 3))
		], name = 'Decoder')

	def call(self, x):
		np.random.seed()
		self.batch_size = tf.shape(x)[0]
		ortho_mats = np.float32(sog.rvs(dim=3)) # np.array([sog.rvs(dim=3) for _ in range(self.batch_size)], dtype=np.float32)
		x = x @ ortho_mats

		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		quaternion = self.transformer(x)
		normalized = transformation.quaternion.normalize(quaternion)
		broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
		transformed = transformation.quaternion.rotate(decoded, broadcasted)

		y = transformed @ tf.linalg.inv(ortho_mats)
		return y

class AutoEncoderL2(Model):
	def __init__(self, latent_dim, cloud_size):
		super(AutoEncoderL2, self).__init__()
		self.latent_dim = latent_dim
		self.cloud_size = cloud_size
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,3)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=32, kernel_size=(1,3), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256, activation='relu', kernel_regularizer='l2'),
			layers.BatchNormalization(),

			layers.Dense(units=128, activation='relu', kernel_regularizer='l2'),
			layers.BatchNormalization(),

			layers.Dense(units=4, activation='linear')
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,3)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=32, kernel_size=(1,3), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=32, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid", activation='relu'),
			layers.BatchNormalization(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256, activation='relu', kernel_regularizer='l2'),
			layers.BatchNormalization(),

			layers.Dense(units=128, activation='relu', kernel_regularizer='l2'),
			layers.BatchNormalization(),

			layers.Dense(units=self.latent_dim, activation='linear')
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=32, activation='relu', kernel_regularizer='l2'),
			layers.BatchNormalization(),

			layers.Dense(units=64, activation='relu', kernel_regularizer='l2'),
			layers.BatchNormalization(),

			layers.Dense(units=128, activation='relu', kernel_regularizer='l2'),
			layers.BatchNormalization(),

			layers.Dense(units=self.cloud_size * 3, activation='linear'),

			layers.Reshape((self.cloud_size, 3))
		], name = 'Decoder')

	def call(self, x):
		np.random.seed()
		self.batch_size = tf.shape(x)[0]
		ortho_mats = np.float32(sog.rvs(dim=3)) # np.array([sog.rvs(dim=3) for _ in range(self.batch_size)], dtype=np.float32)
		x = x @ ortho_mats

		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		quaternion = self.transformer(x)
		normalized = transformation.quaternion.normalize(quaternion)
		broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
		transformed = transformation.quaternion.rotate(decoded, broadcasted)

		y = transformed @ tf.linalg.inv(ortho_mats)
		return y
		
class AutoEncoderV2(Model):
	def __init__(self, batch_size, latent_dim, cloud_size, dimen_size):
		super(AutoEncoderV2, self).__init__()

		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.cloud_size = cloud_size
		self.dimen_size = dimen_size
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=32, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=4 if self.dimen_size == 3 else 1)
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=32, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=32, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=self.latent_dim)
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=32),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=64),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=self.cloud_size * self.dimen_size),
			layers.Reshape((self.cloud_size, self.dimen_size))
		], name = 'Decoder')

	def call(self, x, training=None):

		encoded = self.encoder(x, training=training)
		decoded = self.decoder(encoded, training=training)
		transfm = self.transformer(x, training=training)
		
		if self.dimen_size == 3:
			normalized = transformation.quaternion.normalize(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
			y = transformation.quaternion.rotate(decoded, broadcasted)
		elif self.dimen_size == 2:
			normalized = transformation.rotation_matrix_2d.from_euler(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -3), [self.batch_size, self.cloud_size, 2, 2])
			y = transformation.rotation_matrix_2d.rotate(decoded, broadcasted)

		return y
	
	def train_step(self, data):
		# Unpack the data. Its structure depends on your model and
		# on what you pass to `fit()`.
		x, y = data, data

		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)  # Forward pass
			# Compute the loss value
			# (the loss function is configured in `compile()`)
			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		# Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(y, y_pred)
		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}

class AutoEncoderV2L2(Model):
	def __init__(self, latent_dim, cloud_size):
		super(AutoEncoderV2L2, self).__init__()
		self.latent_dim = latent_dim
		self.cloud_size = cloud_size
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,3)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=32, kernel_size=(1,3), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256, kernel_regularizer='l2'),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128, kernel_regularizer='l2'),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=4)
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,3)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=32, kernel_size=(1,3), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=32, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256, kernel_regularizer='l2'),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128, kernel_regularizer='l2'),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=self.latent_dim)
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=32, kernel_regularizer='l2'),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=64, kernel_regularizer='l2'),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128, kernel_regularizer='l2'),
			layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=self.cloud_size * 3),
			layers.Reshape((self.cloud_size, 3))
		], name = 'Decoder')

	def call(self, x):
		np.random.seed()
		self.batch_size = tf.shape(x)[0]
		ortho_mats = np.float32(sog.rvs(dim=3)) # np.array([sog.rvs(dim=3) for _ in range(self.batch_size)], dtype=np.float32)
		x = x @ ortho_mats

		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		quaternion = self.transformer(x)
		normalized = transformation.quaternion.normalize(quaternion)
		broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
		transformed = transformation.quaternion.rotate(decoded, broadcasted)

		y = transformed @ tf.linalg.inv(ortho_mats)
		return y

class AutoEncoderV3(Model):
	def __init__(self, batch_size, latent_dim, cloud_size, dimen_size):
		super(AutoEncoderV3, self).__init__()

		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.cloud_size = cloud_size
		self.dimen_size = dimen_size
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=32, kernel_size=(1,self.dimen_size), padding="valid"),
			# layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			# layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=4 if self.dimen_size == 3 else 1)
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=32, kernel_size=(1,self.dimen_size), padding="valid"),
			# layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=32, kernel_size=(1,1), padding="valid"),
			# layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			# layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			# layers.BatchNormalization(),
			layers.ReLU(6),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),
			
			layers.Dense(units=256),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=self.latent_dim)
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=32),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=64),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=128),
			# layers.BatchNormalization(),
			layers.ReLU(6),

			layers.Dense(units=self.cloud_size * self.dimen_size),
			layers.Reshape((self.cloud_size, self.dimen_size))
		], name = 'Decoder')

	def call(self, x, training=None):

		encoded = self.encoder(x, training=training)
		decoded = self.decoder(encoded, training=training)
		transfm = self.transformer(x, training=training)
		
		if self.dimen_size == 3:
			normalized = transformation.quaternion.normalize(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
			y = transformation.quaternion.rotate(decoded, broadcasted)
		elif self.dimen_size == 2:
			normalized = transformation.rotation_matrix_2d.from_euler(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -3), [self.batch_size, self.cloud_size, 2, 2])
			y = transformation.rotation_matrix_2d.rotate(decoded, broadcasted)

		return y
	
	def train_step(self, data):
		# Unpack the data. Its structure depends on your model and
		# on what you pass to `fit()`.
		x, y = data, data

		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)  # Forward pass
			# Compute the loss value
			# (the loss function is configured in `compile()`)
			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		# Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(y, y_pred)
		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}

class AutoEncoderV4(Model):
	def __init__(self, repeat_size, batch_size, cloud_size, dimen_size, latent_dim, crc_weight, radius):
		super(AutoEncoderV4, self).__init__()

		self.repeat_size = repeat_size
		self.batch_size = batch_size
		self.cloud_size = cloud_size
		self.dimen_size = dimen_size
		self.latent_dim = latent_dim
		self.crc_weight = crc_weight
		self.radius = radius
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=256, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=128, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=4 if self.dimen_size == 3 else 1)
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=256, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=128, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=self.latent_dim)
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=16, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=32, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=36, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=self.cloud_size * self.dimen_size),
			layers.Reshape((self.cloud_size, self.dimen_size))
		], name = 'Decoder')

	def call(self, x, training=None):

		if training:
			point_cloud, rotate_matrix = x
			transfm_list, encoded_list, rotated_list = [], [], []

			main_transfm = self.transformer(point_cloud, training=training)
			# main_transfm_norm = transformation.quaternion.normalize(main_transfm)
			for i in range(self.repeat_size):
				rotmat = tf.broadcast_to(rotate_matrix[i], [self.batch_size, self.cloud_size, self.dimen_size, self.dimen_size]) 
				point_cloud_rotated = self.rotate(point_cloud, rotmat)
				# transfm = self.transformer(point_cloud_rotated, training=training)
				# transfm_norm = transformation.quaternion.normalize(transfm)
				encoded = self.encoder(point_cloud_rotated, training=training)
				decoded = self.decoder(encoded, training=training)
				rotated = self.transform(decoded, main_transfm)

				# transfm_list.append(transfm_norm) # 10, 32, 4
				encoded_list.append(encoded) # 10, 32, 2
				rotated_list.append(rotated) # 10, 32, 12, 3

			encoded_list = tf.stack(encoded_list)

			# cross_rotational_consistency_loss
			# encoded_0 = tf.broadcast_to(tf.expand_dims(encoded_list, 0), [self.repeat_size, self.repeat_size, self.batch_size, self.latent_dim]) 
			# encoded_1 = tf.broadcast_to(tf.expand_dims(encoded_list, 1), [self.repeat_size, self.repeat_size, self.batch_size, self.latent_dim])
			# cross_rotational_consistency_loss = tf.reduce_mean(tf.square(tf.norm(encoded_0 - encoded_1, ord=2, axis=-1)))
			crc_loss_list = []
			for i in range(self.repeat_size):
				for j in range(i+1, self.repeat_size):
					crc_loss_list.append(tf.square(tf.norm(encoded_list[i] - encoded_list[j], ord=2, axis=-1)))
			cross_rotational_consistency_loss = tf.reduce_mean(crc_loss_list)

			self.add_loss(self.crc_weight * cross_rotational_consistency_loss)
			self.add_metric(cross_rotational_consistency_loss, name='cross_rotational_consistency_loss')

			# rotation_loss
			# @tf.function
			# def geodesic_distance(R1, R2):
			# 	matmuled = tf.matmul(R1, R2, transpose_b=True)
			# 	traced = tf.linalg.trace(matmuled)
			# 	cosval = (traced - 1) / 2
			# 	return tf.math.log(2 - cosval)

			# pred_rotate_matrix = transformation.rotation_matrix_3d.from_quaternion(transfm_list) # 10, 32, 3, 3
			# true_rotate_matrix = tf.broadcast_to(tf.expand_dims(rotate_matrix, 1), [self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# matmuled = tf.matmul(true_rotate_matrix, pred_rotate_matrix, transpose_a=True)
			# matmuled_0 = tf.broadcast_to(tf.expand_dims(matmuled, 0), [self.repeat_size, self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# matmuled_1 = tf.broadcast_to(tf.expand_dims(matmuled, 1), [self.repeat_size, self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# rotation_loss = tf.reduce_mean(geodesic_distance(matmuled_0, matmuled_1))

			# rot_loss_list = []
			# for i in range(self.repeat_size):
			# 	for j in range(i+1, self.repeat_size):
			# 		rot_loss_list.append(geodesic_distance(matmuled[i], matmuled[j]))
			# rotation_loss = tf.reduce_mean(rot_loss_list)

			# main_transfm_bt = tf.broadcast_to(main_transfm_norm, [self.repeat_size, self.batch_size, 4])
			# main_rotmat = transformation.rotation_matrix_3d.from_quaternion(main_transfm_bt)
			# relative_angle = geodesic_distance(main_rotmat, matmuled)
			# rotation_loss = tf.reduce_mean(relative_angle)
			# self.add_loss(rotation_loss)
			# self.add_metric(rotation_loss, name='rotation_loss')

			# overlap loss
			rotated_bt_2 = tf.broadcast_to(tf.expand_dims(rotated_list, -2), [self.repeat_size, self.batch_size, self.cloud_size, self.cloud_size, self.dimen_size])
			rotated_bt_3 = tf.broadcast_to(tf.expand_dims(rotated_list, -3), [self.repeat_size, self.batch_size, self.cloud_size, self.cloud_size, self.dimen_size])
			distance = tf.reduce_sum(tf.square(rotated_bt_2 - rotated_bt_3), axis=-1) # rs, bs, cs, cs
			removezeros = tf.linalg.set_diag(distance, 2*self.radius * tf.ones((self.repeat_size, self.batch_size, self.cloud_size)))
			mindis = tf.reduce_min(removezeros, axis=-1) # rs, bs, cs
			truncated = 2*self.radius - tf.minimum(mindis, 2*self.radius) # mindis < 2*radius
			# truncated = tf.nn.relu(2*self.radius - mindis)
			
			overlap_loss = tf.reduce_mean(truncated)
			self.add_loss(0.001 * overlap_loss)
			self.add_metric(overlap_loss, name='overlap_loss')

			return tf.stack(rotated_list)

		else:
			point_cloud = x

			transfm = self.transformer(point_cloud, training=training)
			encoded = self.encoder(point_cloud, training=training)
			decoded = self.decoder(encoded, training=training)
			rotated = self.transform(decoded, transfm)

			return rotated

	def transform(self, decoded, transfm):
		if self.dimen_size == 3:
			normalized = transformation.quaternion.normalize(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
			y = transformation.quaternion.rotate(decoded, broadcasted)
		elif self.dimen_size == 2:
			rotate_matrix = transformation.rotation_matrix_2d.from_euler(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(rotate_matrix, -3), [self.batch_size, self.cloud_size, 2, 2])
			y = transformation.rotation_matrix_2d.rotate(decoded, broadcasted)

		return y

	def rotate(self, point_cloud, rotmat):
		if self.dimen_size == 3:
			y = transformation.rotation_matrix_3d.rotate(point_cloud, rotmat)
		elif self.dimen_size == 2:
			y = transformation.rotation_matrix_2d.rotate(point_cloud, rotmat)
		return y

class AutoEncoderV41(Model):
	def __init__(self, repeat_size, batch_size, cloud_size, dimen_size, latent_dim, crc_weight, radius):
		super(AutoEncoderV41, self).__init__()

		self.repeat_size = repeat_size
		self.batch_size = batch_size
		self.cloud_size = cloud_size
		self.dimen_size = dimen_size
		self.latent_dim = latent_dim
		self.crc_weight = crc_weight
		self.radius = radius
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=256, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=1024, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=128, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=4 if self.dimen_size == 3 else 1)
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=256, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=1024, kernel_size=(1,1), padding="valid"),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=128, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=self.latent_dim)
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=16, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=32, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=36, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=self.cloud_size * self.dimen_size),
			layers.Reshape((self.cloud_size, self.dimen_size))
		], name = 'Decoder')

	def call(self, x, training=None):

		if training:
			point_cloud, rotate_matrix = x
			transfm_list, encoded_list, rotated_list = [], [], []

			main_transfm = self.transformer(point_cloud, training=training)
			# main_transfm_norm = transformation.quaternion.normalize(main_transfm)
			for i in range(self.repeat_size):
				rotmat = tf.broadcast_to(rotate_matrix[i], [self.batch_size, self.cloud_size, self.dimen_size, self.dimen_size]) 
				point_cloud_rotated = self.rotate(point_cloud, rotmat)
				# transfm = self.transformer(point_cloud_rotated, training=training)
				# transfm_norm = transformation.quaternion.normalize(transfm)
				encoded = self.encoder(point_cloud_rotated, training=training)
				decoded = self.decoder(encoded, training=training)
				rotated = self.transform(decoded, main_transfm)

				# transfm_list.append(transfm_norm) # 10, 32, 4
				encoded_list.append(encoded) # 10, 32, 2
				rotated_list.append(rotated) # 10, 32, 12, 3

			encoded_list = tf.stack(encoded_list)

			# cross_rotational_consistency_loss
			# encoded_0 = tf.broadcast_to(tf.expand_dims(encoded_list, 0), [self.repeat_size, self.repeat_size, self.batch_size, self.latent_dim]) 
			# encoded_1 = tf.broadcast_to(tf.expand_dims(encoded_list, 1), [self.repeat_size, self.repeat_size, self.batch_size, self.latent_dim])
			# cross_rotational_consistency_loss = tf.reduce_mean(tf.square(tf.norm(encoded_0 - encoded_1, ord=2, axis=-1)))
			crc_loss_list = []
			for i in range(self.repeat_size):
				for j in range(i+1, self.repeat_size):
					crc_loss_list.append(tf.square(tf.norm(encoded_list[i] - encoded_list[j], ord=2, axis=-1)))
			cross_rotational_consistency_loss = tf.reduce_mean(crc_loss_list)

			self.add_loss(self.crc_weight * cross_rotational_consistency_loss)
			self.add_metric(cross_rotational_consistency_loss, name='cross_rotational_consistency_loss')

			# rotation_loss
			# @tf.function
			# def geodesic_distance(R1, R2):
			# 	matmuled = tf.matmul(R1, R2, transpose_b=True)
			# 	traced = tf.linalg.trace(matmuled)
			# 	cosval = (traced - 1) / 2
			# 	return tf.math.log(2 - cosval)

			# pred_rotate_matrix = transformation.rotation_matrix_3d.from_quaternion(transfm_list) # 10, 32, 3, 3
			# true_rotate_matrix = tf.broadcast_to(tf.expand_dims(rotate_matrix, 1), [self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# matmuled = tf.matmul(true_rotate_matrix, pred_rotate_matrix, transpose_a=True)
			# matmuled_0 = tf.broadcast_to(tf.expand_dims(matmuled, 0), [self.repeat_size, self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# matmuled_1 = tf.broadcast_to(tf.expand_dims(matmuled, 1), [self.repeat_size, self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# rotation_loss = tf.reduce_mean(geodesic_distance(matmuled_0, matmuled_1))

			# rot_loss_list = []
			# for i in range(self.repeat_size):
			# 	for j in range(i+1, self.repeat_size):
			# 		rot_loss_list.append(geodesic_distance(matmuled[i], matmuled[j]))
			# rotation_loss = tf.reduce_mean(rot_loss_list)

			# main_transfm_bt = tf.broadcast_to(main_transfm_norm, [self.repeat_size, self.batch_size, 4])
			# main_rotmat = transformation.rotation_matrix_3d.from_quaternion(main_transfm_bt)
			# relative_angle = geodesic_distance(main_rotmat, matmuled)
			# rotation_loss = tf.reduce_mean(relative_angle)
			# self.add_loss(rotation_loss)
			# self.add_metric(rotation_loss, name='rotation_loss')

			# overlap loss
			rotated_bt_2 = tf.broadcast_to(tf.expand_dims(rotated_list, -2), [self.repeat_size, self.batch_size, self.cloud_size, self.cloud_size, self.dimen_size])
			rotated_bt_3 = tf.broadcast_to(tf.expand_dims(rotated_list, -3), [self.repeat_size, self.batch_size, self.cloud_size, self.cloud_size, self.dimen_size])
			distance = tf.reduce_sum(tf.square(rotated_bt_2 - rotated_bt_3), axis=-1) # rs, bs, cs, cs
			removezeros = tf.linalg.set_diag(distance, 2*self.radius * tf.ones((self.repeat_size, self.batch_size, self.cloud_size)))
			mindis = tf.reduce_min(removezeros, axis=-1) # rs, bs, cs
			truncated = 2*self.radius - tf.minimum(mindis, 2*self.radius) # mindis < 2*radius
			# truncated = tf.nn.relu(2*self.radius - mindis)
			
			overlap_loss = tf.reduce_mean(truncated)
			self.add_loss(0.001 * overlap_loss)
			self.add_metric(overlap_loss, name='overlap_loss')

			return tf.stack(rotated_list)

		else:
			point_cloud = x

			transfm = self.transformer(point_cloud, training=training)
			encoded = self.encoder(point_cloud, training=training)
			decoded = self.decoder(encoded, training=training)
			rotated = self.transform(decoded, transfm)

			return rotated

	def transform(self, decoded, transfm):
		if self.dimen_size == 3:
			normalized = transformation.quaternion.normalize(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
			y = transformation.quaternion.rotate(decoded, broadcasted)
		elif self.dimen_size == 2:
			rotate_matrix = transformation.rotation_matrix_2d.from_euler(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(rotate_matrix, -3), [self.batch_size, self.cloud_size, 2, 2])
			y = transformation.rotation_matrix_2d.rotate(decoded, broadcasted)

		return y

	def rotate(self, point_cloud, rotmat):
		if self.dimen_size == 3:
			y = transformation.rotation_matrix_3d.rotate(point_cloud, rotmat)
		elif self.dimen_size == 2:
			y = transformation.rotation_matrix_2d.rotate(point_cloud, rotmat)
		return y

class PointNetAutoEncoder(Model):
	def __init__(self, repeat_size, batch_size, cloud_size, dimen_size, latent_dim, crc_weight, l2_weight, radius):
		super(PointNetAutoEncoder, self).__init__()

		self.repeat_size = repeat_size
		self.batch_size = batch_size
		self.cloud_size = cloud_size
		self.dimen_size = dimen_size
		self.latent_dim = latent_dim
		self.crc_weight = crc_weight
		self.l2_weight = l2_weight
		self.radius = radius
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=1024, kernel_size=(1,1), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=512, kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=4 if self.dimen_size == 3 else 1)
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=1024, kernel_size=(1,1), padding="valid", kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=512, kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=self.latent_dim)
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=16, kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=32, kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=64, kernel_regularizer=regularizers.L2(self.l2_weight)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=self.cloud_size * self.dimen_size),
			layers.Reshape((self.cloud_size, self.dimen_size))
		], name = 'Decoder')

	def call(self, x, training=None):

		if training:
			point_cloud, rotate_matrix = x
			transfm_list, encoded_list, rotated_list = [], [], []

			main_transfm = self.transformer(point_cloud, training=training)
			# main_transfm_norm = transformation.quaternion.normalize(main_transfm)
			for i in range(self.repeat_size):
				rotmat = tf.broadcast_to(rotate_matrix[i], [self.batch_size, self.cloud_size, self.dimen_size, self.dimen_size]) 
				point_cloud_rotated = self.rotate(point_cloud, rotmat)
				# transfm = self.transformer(point_cloud_rotated, training=training)
				# transfm_norm = transformation.quaternion.normalize(transfm)
				encoded = self.encoder(point_cloud_rotated, training=training)
				decoded = self.decoder(encoded, training=training)
				rotated = self.transform(decoded, main_transfm)

				# transfm_list.append(transfm_norm) # 10, 32, 4
				encoded_list.append(encoded) # 10, 32, 2
				rotated_list.append(rotated) # 10, 32, 12, 3

			encoded_list = tf.stack(encoded_list)

			# cross_rotational_consistency_loss
			# encoded_0 = tf.broadcast_to(tf.expand_dims(encoded_list, 0), [self.repeat_size, self.repeat_size, self.batch_size, self.latent_dim]) 
			# encoded_1 = tf.broadcast_to(tf.expand_dims(encoded_list, 1), [self.repeat_size, self.repeat_size, self.batch_size, self.latent_dim])
			# cross_rotational_consistency_loss = tf.reduce_mean(tf.square(tf.norm(encoded_0 - encoded_1, ord=2, axis=-1)))
			crc_loss_list = []
			for i in range(self.repeat_size):
				for j in range(i+1, self.repeat_size):
					crc_loss_list.append(tf.square(tf.norm(encoded_list[i] - encoded_list[j], ord=2, axis=-1)))
			cross_rotational_consistency_loss = tf.reduce_mean(crc_loss_list)

			self.add_loss(self.crc_weight * cross_rotational_consistency_loss)
			self.add_metric(cross_rotational_consistency_loss, name='cross_rotational_consistency_loss')

			# rotation_loss
			# @tf.function
			# def geodesic_distance(R1, R2):
			# 	matmuled = tf.matmul(R1, R2, transpose_b=True)
			# 	traced = tf.linalg.trace(matmuled)
			# 	cosval = (traced - 1) / 2
			# 	return tf.math.log(2 - cosval)

			# pred_rotate_matrix = transformation.rotation_matrix_3d.from_quaternion(transfm_list) # 10, 32, 3, 3
			# true_rotate_matrix = tf.broadcast_to(tf.expand_dims(rotate_matrix, 1), [self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# matmuled = tf.matmul(true_rotate_matrix, pred_rotate_matrix, transpose_a=True)
			# matmuled_0 = tf.broadcast_to(tf.expand_dims(matmuled, 0), [self.repeat_size, self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# matmuled_1 = tf.broadcast_to(tf.expand_dims(matmuled, 1), [self.repeat_size, self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# rotation_loss = tf.reduce_mean(geodesic_distance(matmuled_0, matmuled_1))

			# rot_loss_list = []
			# for i in range(self.repeat_size):
			# 	for j in range(i+1, self.repeat_size):
			# 		rot_loss_list.append(geodesic_distance(matmuled[i], matmuled[j]))
			# rotation_loss = tf.reduce_mean(rot_loss_list)

			# main_transfm_bt = tf.broadcast_to(main_transfm_norm, [self.repeat_size, self.batch_size, 4])
			# main_rotmat = transformation.rotation_matrix_3d.from_quaternion(main_transfm_bt)
			# relative_angle = geodesic_distance(main_rotmat, matmuled)
			# rotation_loss = tf.reduce_mean(relative_angle)
			# self.add_loss(rotation_loss)
			# self.add_metric(rotation_loss, name='rotation_loss')

			# overlap loss
			rotated_bt_2 = tf.broadcast_to(tf.expand_dims(rotated_list, -2), [self.repeat_size, self.batch_size, self.cloud_size, self.cloud_size, self.dimen_size])
			rotated_bt_3 = tf.broadcast_to(tf.expand_dims(rotated_list, -3), [self.repeat_size, self.batch_size, self.cloud_size, self.cloud_size, self.dimen_size])
			distance = tf.reduce_sum(tf.square(rotated_bt_2 - rotated_bt_3), axis=-1) # rs, bs, cs, cs
			removezeros = tf.linalg.set_diag(distance, 2*self.radius * tf.ones((self.repeat_size, self.batch_size, self.cloud_size)))
			mindis = tf.reduce_min(removezeros, axis=-1) # rs, bs, cs
			truncated = 2*self.radius - tf.minimum(mindis, 2*self.radius) # mindis < 2*radius
			# truncated = tf.nn.relu(2*self.radius - mindis)
			
			overlap_loss = tf.reduce_mean(truncated)
			self.add_loss(0.001 * overlap_loss)
			self.add_metric(overlap_loss, name='overlap_loss')

			return tf.stack(rotated_list)

		else:
			point_cloud = x

			transfm = self.transformer(point_cloud, training=training)
			encoded = self.encoder(point_cloud, training=training)
			decoded = self.decoder(encoded, training=training)
			rotated = self.transform(decoded, transfm)

			return rotated

	def transform(self, decoded, transfm):
		if self.dimen_size == 3:
			normalized = transformation.quaternion.normalize(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
			y = transformation.quaternion.rotate(decoded, broadcasted)
		elif self.dimen_size == 2:
			rotate_matrix = transformation.rotation_matrix_2d.from_euler(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(rotate_matrix, -3), [self.batch_size, self.cloud_size, 2, 2])
			y = transformation.rotation_matrix_2d.rotate(decoded, broadcasted)

		return y

	def rotate(self, point_cloud, rotmat):
		if self.dimen_size == 3:
			y = transformation.rotation_matrix_3d.rotate(point_cloud, rotmat)
		elif self.dimen_size == 2:
			y = transformation.rotation_matrix_2d.rotate(point_cloud, rotmat)
		return y

class AutoEncoderV5(Model):
	def __init__(self, repeat_size, batch_size, latent_dim, cloud_size, dimen_size):
		super(AutoEncoderV5, self).__init__()

		self.repeat_size = repeat_size
		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.cloud_size = cloud_size
		self.dimen_size = dimen_size
		self.crc_weight = 0
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=256, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=128, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=4 if self.dimen_size == 3 else 1)
		], name = 'Transformer')

		# Encoder
		self.encoder = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=256, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=512, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=128, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=self.latent_dim)
		], name = 'Encoder')

		# Decoder
		self.decoder = Sequential([
			layers.Input(shape=(self.latent_dim,)),

			layers.Dense(units=16, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=32, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=36, kernel_regularizer=regularizers.L2(0)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=self.cloud_size * self.dimen_size),
			layers.Reshape((self.cloud_size, self.dimen_size))
		], name = 'Decoder')

	def call(self, x, training=None):

		if training:
			point_cloud, rotate_matrix = x
			transfm_list, encoded_list, rotated_list = [], [], []

			main_transfm = self.transformer(point_cloud, training=training)
			# main_transfm_norm = transformation.quaternion.normalize(main_transfm)
			for i in range(self.repeat_size):
				rotmat = tf.broadcast_to(rotate_matrix[i], [self.batch_size, self.cloud_size, self.dimen_size, self.dimen_size]) 
				point_cloud_rotated = self.rotate(point_cloud, rotmat)
				# transfm = self.transformer(point_cloud_rotated, training=training)
				# transfm_norm = transformation.quaternion.normalize(transfm)
				encoded = self.encoder(point_cloud_rotated, training=training)
				decoded = self.decoder(encoded, training=training)
				rotated = self.transform(decoded, main_transfm)

				# transfm_list.append(transfm_norm) # 10, 32, 4
				encoded_list.append(encoded) # 10, 32, 2
				rotated_list.append(rotated) # 10, 32, 12, 3

			encoded_list = tf.stack(encoded_list)

			# cross_rotational_consistency_loss
			# encoded_0 = tf.broadcast_to(tf.expand_dims(encoded_list, 0), [self.repeat_size, self.repeat_size, self.batch_size, self.latent_dim]) 
			# encoded_1 = tf.broadcast_to(tf.expand_dims(encoded_list, 1), [self.repeat_size, self.repeat_size, self.batch_size, self.latent_dim])
			# cross_rotational_consistency_loss = tf.reduce_mean(tf.square(tf.norm(encoded_0 - encoded_1, ord=2, axis=-1)))
			crc_loss_list = []
			for i in range(self.repeat_size):
				for j in range(i+1, self.repeat_size):
					crc_loss_list.append(tf.square(tf.norm(encoded_list[i] - encoded_list[j], ord=2, axis=-1)))
			cross_rotational_consistency_loss = tf.reduce_mean(crc_loss_list)

			self.add_loss(0.001 * cross_rotational_consistency_loss)
			self.add_metric(cross_rotational_consistency_loss, name='cross_rotational_consistency_loss')

			# rotation_loss
			# @tf.function
			# def geodesic_distance(R1, R2):
			# 	matmuled = tf.matmul(R1, R2, transpose_b=True)
			# 	traced = tf.linalg.trace(matmuled)
			# 	cosval = (traced - 1) / 2
			# 	return 1 - cosval

			# pred_rotate_matrix = transformation.rotation_matrix_3d.from_quaternion(transfm_list) # 10, 32, 3, 3
			# true_rotate_matrix = tf.broadcast_to(tf.expand_dims(rotate_matrix, 1), [self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# matmuled = tf.matmul(true_rotate_matrix, pred_rotate_matrix, transpose_a=True)
			# matmuled_0 = tf.broadcast_to(tf.expand_dims(matmuled, 0), [self.repeat_size, self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# matmuled_1 = tf.broadcast_to(tf.expand_dims(matmuled, 1), [self.repeat_size, self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			# rotation_loss = tf.reduce_mean(geodesic_distance(matmuled_0, matmuled_1))

			# rot_loss_list = []
			# for i in range(self.repeat_size):
			# 	for j in range(i+1, self.repeat_size):
			# 		rot_loss_list.append(geodesic_distance(matmuled[i], matmuled[j]))
			# rotation_loss = tf.reduce_mean(rot_loss_list)

			# main_transfm_bt = tf.broadcast_to(main_transfm_norm, [self.repeat_size, self.batch_size, 4])
			# main_rotmat = transformation.rotation_matrix_3d.from_quaternion(main_transfm_bt)
			# relative_angle = geodesic_distance(main_rotmat, matmuled)
			# rotation_loss = tf.reduce_mean(relative_angle)
			# self.add_loss(rotation_loss)
			# self.add_metric(rotation_loss, name='rotation_loss')

			return tf.stack(rotated_list)

		else:
			point_cloud = x

			transfm = self.transformer(point_cloud, training=training)
			encoded = self.encoder(point_cloud, training=training)
			decoded = self.decoder(encoded, training=training)
			rotated = self.transform(decoded, transfm)

			return rotated

	def transform(self, decoded, transfm):
		if self.dimen_size == 3:
			normalized = transformation.quaternion.normalize(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(normalized, -2), [self.batch_size, self.cloud_size, 4])
			y = transformation.quaternion.rotate(decoded, broadcasted)
		elif self.dimen_size == 2:
			rotate_matrix = transformation.rotation_matrix_2d.from_euler(transfm)
			broadcasted = tf.broadcast_to(tf.expand_dims(rotate_matrix, -3), [self.batch_size, self.cloud_size, 2, 2])
			y = transformation.rotation_matrix_2d.rotate(decoded, broadcasted)

		return y

	def rotate(self, point_cloud, rotmat):
		if self.dimen_size == 3:
			y = transformation.rotation_matrix_3d.rotate(point_cloud, rotmat)
		elif self.dimen_size == 2:
			y = transformation.rotation_matrix_2d.rotate(point_cloud, rotmat)
		return y

class Transformer(Model):
	def __init__(self, repeat_size, batch_size, latent_dim, cloud_size, dimen_size):
		super(Transformer, self).__init__()

		self.repeat_size = repeat_size
		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.cloud_size = cloud_size
		self.dimen_size = dimen_size
		# Transformer
		self.transformer = Sequential([
			layers.Input(shape=(self.cloud_size,self.dimen_size)),
			layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
			
			layers.Conv2D(filters=64, kernel_size=(1,self.dimen_size), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=64, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=128, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),
			layers.Conv2D(filters=1024, kernel_size=(1,1), padding="valid"),
			layers.BatchNormalization(),
			layers.ReLU(),

			layers.MaxPool2D(pool_size=(self.cloud_size,1), padding='valid'),
			layers.Flatten(),

			layers.Dense(units=512, kernel_regularizer=regularizers.L2(1e-6)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=256, kernel_regularizer=regularizers.L2(1e-6)),
			layers.LayerNormalization(),
			layers.ReLU(),

			layers.Dense(units=4 if self.dimen_size == 3 else 1)
		], name = 'Transformer')

	def call(self, x, training=None):

		if training:

			point_cloud, rotate_matrix = x
			transfm_list = []

			main_transfm = self.transformer(point_cloud, training=training)
			main_transfm = transformation.quaternion.normalize(main_transfm)
			for i in range(self.repeat_size):
				rotmat = tf.broadcast_to(rotate_matrix[i], [self.batch_size, self.cloud_size, self.dimen_size, self.dimen_size]) 
				point_cloud_rotated = transformation.rotation_matrix_3d.rotate(point_cloud, rotmat)
				transfm = self.transformer(point_cloud_rotated, training=training)
				transfm = transformation.quaternion.normalize(transfm)

				transfm_list.append(transfm) # 10, 32, 4

			# @tf.function
			def geodesic_distance(R1, R2):
				matmuled = tf.matmul(R1, R2, transpose_b=True)
				traced = tf.linalg.trace(matmuled)
				cosval = (traced - 1) / 2
				return 1 - cosval

			pred_rotate_matrix = transformation.rotation_matrix_3d.from_quaternion(transfm_list) # 10, 32, 3, 3
			true_rotate_matrix = tf.broadcast_to(tf.expand_dims(rotate_matrix, 1), [self.repeat_size, self.batch_size, self.dimen_size, self.dimen_size]) 
			matmuled = tf.matmul(true_rotate_matrix, pred_rotate_matrix, transpose_a=True)
			# # transformation.rotation_matrix_3d.assert_rotation_matrix_normalized(matmuled)
			# quaternion = transformation.quaternion.from_rotation_matrix(matmuled)
			# inversed = transformation.quaternion.inverse(quaternion)
			# multiplied = transformation.quaternion.multiply(quaternion, inversed)
			# rot_loss_list = tf.norm(multiplied, axis=-1)

			# rotation_loss = tf.reduce_mean(rot_loss_list)

			main_transfm_bt = tf.broadcast_to(main_transfm, [self.repeat_size, self.batch_size, 4])
			main_rotmat = transformation.rotation_matrix_3d.from_quaternion(main_transfm_bt)
			relative_angle = geodesic_distance(main_rotmat, matmuled)
			rotation_loss = tf.reduce_mean(relative_angle)

			self.add_loss(rotation_loss)
			self.add_metric(rotation_loss, name='rotation_loss')

		return None