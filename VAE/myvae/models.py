import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers, regularizers, losses
from keras.layers.convolutional import Convolution1D

from keras.models import load_model
from keras import backend as K
from keras.models import Model


class PointNetAutoEncoder(Model):
	def __init__(self, params):
		super(PointNetAutoEncoder, self).__init__()
		self.build()
		
	def build(self):
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
		self.encoder = Encoder()
		
		# Decoder
		self.decoder = Decoder()

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

# =============================
# Encoder functions
# =============================

def Encoder(params):
    # K_params is dictionary of keras variables
    x_in = layers.Input(shape=(params['cloud_size'],3))

    # Convolution layers
    x = layers.Lambda(lambda w: tf.expand_dims(w, axis=-1))(x_in)

    x = layers.Conv2D(filters=32, kernel_size=(1,3), padding="valid", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1,3), padding="valid", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1,3), padding="valid", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=512, kernel_size=(1,3), padding="valid", activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool2D(pool_size=(params['cloud_size'],1), padding='valid')(x)
    x = layers.Flatten()(x)
		
    # Middle layers
    x = layers.Dense(units=256, activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(units=128, activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    
    z_mean = layers.Dense(params['latent_dim'], activation='linear', name='z_mean_sample')(x)

    # return both mean and last encoding layer for std dev sampling
    return Model(x_in, [z_mean, x], name="encoder")

def load_encoder(params):
    # Need to handle K_params somehow...
    # Also are we going to be able to save this layer?
    # encoder = encoder_model(params, K.constant(0))
    # encoder.load_weights(params['encoder_weights_file'])
    # return encoder
    # !# not sure if this is the right format
    return load_model(params['encoder_weights_file'])


# ===========================================
# Decoder functions
# ===========================================

def Decoder(params):
    z_in = layers.Input(shape=(params['latent_dim'],))

    z = layers.Dense(units=32, activation='relu')(z_in)
    z = layers.BatchNormalization()(z)
    
    z = layers.Dense(units=64, activation='relu')(z)
    z = layers.BatchNormalization()(z)

    z = layers.Dense(units=128, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    
    z = layers.Dense(units=params['cloud_size'] * 3, activation='linear')(z)
    x_out = layers.Reshape((params['cloud_size'], 3))(z)
    
    return Model(z_in, x_out, name="decoder")


def load_decoder(params):
    return load_model(params['decoder_weights_file'])


##====================
## Middle part (var)
##====================

def variational_layers(z_mean, enc, kl_loss_var, params):
    # @inp mean : mean generated from encoder
    # @inp enc : output generated by encoding
    # @inp params : parameter dictionary passed throughout entire model.

    def sampling(args):
        z_mean, z_log_var = args

        epsilon = K.random_normal_variable(shape=(params['batch_size'], params['hidden_dim']),
                                           mean=0., scale=1.)
        # insert kl loss here

        z_rand = z_mean + K.exp(z_log_var / 2) * kl_loss_var * epsilon
        return K.in_train_phase(z_rand, z_mean)


    # variational encoding
    z_log_var_layer = layers.Dense(params['hidden_dim'], name='z_log_var_sample')
    z_log_var = z_log_var_layer(enc)

    z_mean_log_var_output = layers.Concatenate(name='z_mean_log_var')([z_mean, z_log_var])

    z_samp = layers.Lambda(sampling)([z_mean, z_log_var])
    z_samp = layers.BatchNormalization(axis=-1)(z_samp)

    return z_samp, z_mean_log_var_output


# ====================
# Property Prediction
# ====================

def property_predictor_model(params):
    if ('reg_prop_tasks' not in params) and ('logit_prop_tasks' not in params):
        raise ValueError('You must specify either regression tasks and/or logistic tasks for property prediction')

    ls_in = layers.Input(shape=(params['hidden_dim'],), name='prop_pred_input')

    prop_mid = layers.Dense(params['prop_hidden_dim'], activation=params['prop_pred_activation'])(ls_in)
    if params['prop_pred_dropout'] > 0:
        prop_mid = layers.Dropout(params['prop_pred_dropout'])(prop_mid)

    if params['prop_pred_depth'] > 1:
        for p_i in range(1, params['prop_pred_depth']):
            prop_mid = layers.Dense(int(params['prop_hidden_dim'] *
                                 params['prop_growth_factor'] ** p_i),
                             activation=params['prop_pred_activation'],
                             name="property_predictor_dense{}".format(p_i)
                             )(prop_mid)
            if params['prop_pred_dropout'] > 0:
                prop_mid = layers.Dropout(params['prop_pred_dropout'])(prop_mid)
            if 'prop_batchnorm' in params and params['prop_batchnorm']:
                prop_mid = layers.BatchNormalization()(prop_mid)

    # for regression tasks
    if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0):
        reg_prop_pred = layers.Dense(len(params['reg_prop_tasks']), activation='linear',
                              name='reg_property_output')(prop_mid)

    # for logistic tasks
    if ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0):
        logit_prop_pred = layers.Dense(len(params['logit_prop_tasks']), activation='sigmoid',
                                name='logit_property_output')(prop_mid)

    # both regression and logistic
    if (('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0) and
            ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0)):

        return Model(ls_in, [reg_prop_pred, logit_prop_pred], name="property_predictor")

        # regression only scenario
    elif ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0):
        return Model(ls_in, reg_prop_pred, name="property_predictor")

        # logit only scenario
    else:
        return Model(ls_in, logit_prop_pred, name="property_predictor")


def load_property_predictor(params):
    return load_model(params['prop_pred_weights_file'])