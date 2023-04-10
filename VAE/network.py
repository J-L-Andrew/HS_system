import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, initializers, regularizers
from tensorflow_graphics.geometry import transformation
from tensorflow_graphics.geometry.transformation.quaternion import multiply

class PointNetAutoEncoder(Model):
	def __init__(self, repeat_size, batch_size, cloud_size, dimen_size, latent_dim, l2_weight, closs_weight, oloss_weight, radius):
		super(PointNetAutoEncoder, self).__init__()

		self.repeat_size = repeat_size
		self.batch_size = batch_size
		self.cloud_size = cloud_size
		self.dimen_size = dimen_size
		self.latent_dim = latent_dim
		self.l2_weight = l2_weight
		self.closs_weight = closs_weight
		self.oloss_weight = oloss_weight
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

			layers.Dense(units=36, kernel_regularizer=regularizers.L2(self.l2_weight)),
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

			self.add_loss(self.closs_weight * cross_rotational_consistency_loss)
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
			self.add_loss(self.oloss_weight * overlap_loss)
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