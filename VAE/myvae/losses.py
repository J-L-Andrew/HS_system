import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, initializers, regularizers, losses
from tensorflow_graphics.geometry import transformation
from tensorflow_graphics.geometry.transformation.quaternion import multiply
from tensorflow_graphics.nn import loss

# def symmetric_chamfer_loss(y_true, y_pred):
	
# 	y_true = tf.reshape(y_true, y_pred.shape)
# 	closs = loss.chamfer_distance.evaluate(y_true, y_pred)
# 	if len(tf.shape(y_pred)) == 4:
# 		closs = tf.reduce_mean(closs, axis=0)
# 	return closs

# def symmetric_hausdorff_loss(y_true, y_pred):

# 	y_true = tf.reshape(y_true, y_pred.shape)
# 	hloss = loss.hausdorff_distance.evaluate(y_true, y_pred) + loss.hausdorff_distance.evaluate(y_pred, y_true)
# 	if len(tf.shape(y_pred)) == 4:
# 		hloss = tf.reduce_mean(hloss, axis=0)
# 	return hloss

def weighted_chamfer_loss(radius_cut=np.inf):

	def weighted_chamfer_loss(y_true, y_pred):
		
		y_true = tf.convert_to_tensor(y_true)
		if len(tf.shape(y_pred)) == 3:
			y_true = tf.expand_dims(y_true, 0)
			y_pred = tf.expand_dims(y_pred, 0)

		rs = tf.shape(y_pred)[0]
		bs = tf.shape(y_pred)[1]
		cs = tf.shape(y_pred)[2]
		ds = tf.shape(y_pred)[3]

		y_true_bt = tf.broadcast_to(tf.expand_dims(y_true, -2), [rs, bs, cs, cs, ds])
		y_pred_bt = tf.broadcast_to(tf.expand_dims(y_pred, -3), [rs, bs, cs, cs, ds])
		distance = tf.reduce_sum(tf.square(y_true_bt - y_pred_bt), axis=-1) # rs, bs, cs, cs
		mindis = tf.reduce_min(distance, axis=-1) # rs, bs, cs
		shelldis = tf.norm(y_true, axis=-1) # rs, bs, cs
		weight = tf.minimum(tf.divide(radius_cut, shelldis), 1) # rs, bs, cs
		wcloss = tf.reduce_mean(tf.multiply(mindis, weight), axis=-1) # rs, bs
		wcloss = tf.reduce_mean(wcloss, axis=0)

		return wcloss
	return weighted_chamfer_loss

def weighted_hausdorff_loss(radius_cut=np.inf):
	def weighted_hausdorff_loss(y_true, y_pred):

		y_true = tf.convert_to_tensor(y_true)
		if len(tf.shape(y_pred)) == 3:
			y_true = tf.expand_dims(y_true, 0)
			y_pred = tf.expand_dims(y_pred, 0)

		rs = tf.shape(y_pred)[0]
		bs = tf.shape(y_pred)[1]
		cs = tf.shape(y_pred)[2]
		ds = tf.shape(y_pred)[3]

		y_true_bt = tf.broadcast_to(tf.expand_dims(y_true, -2), [rs, bs, cs, cs, ds])
		y_pred_bt = tf.broadcast_to(tf.expand_dims(y_pred, -3), [rs, bs, cs, cs, ds])
		distance = tf.reduce_sum(tf.square(y_true_bt - y_pred_bt), axis=-1) # rs, bs, cs, cs
		mindis = tf.reduce_min(distance, axis=-1) # rs, bs, cs
		shelldis = tf.norm(y_true, axis=-1) # rs, bs, cs
		weight = tf.minimum(tf.divide(radius_cut, shelldis), 1) # rs, bs, cs
		whloss = tf.reduce_max(tf.multiply(mindis, weight), axis=-1) # rs, bs
		whloss = tf.reduce_mean(whloss, axis=0)

		return whloss
	return weighted_hausdorff_loss

def symmetric_chamfer_loss(y_true, y_pred):

	y_true = tf.convert_to_tensor(y_true)
	if len(tf.shape(y_pred)) == 3:
		y_true = tf.expand_dims(y_true, 0)
		y_pred = tf.expand_dims(y_pred, 0)

	rs = tf.shape(y_pred)[0]
	bs = tf.shape(y_pred)[1]
	cs = tf.shape(y_pred)[2]
	ds = tf.shape(y_pred)[3]

	y_true_bt = tf.broadcast_to(tf.expand_dims(y_true, -2), [rs, bs, cs, cs, ds])
	y_pred_bt = tf.broadcast_to(tf.expand_dims(y_pred, -3), [rs, bs, cs, cs, ds])
	distance = tf.reduce_sum(tf.square(y_true_bt - y_pred_bt), axis=-1) # rs, bs, cs, cs
	mindis = tf.reduce_min(distance, axis=-1) # rs, bs, cs
	hloss = tf.reduce_mean(mindis, axis=-1) # rs, bs

	y_true_bt = tf.broadcast_to(tf.expand_dims(y_true, -3), [rs, bs, cs, cs, ds])
	y_pred_bt = tf.broadcast_to(tf.expand_dims(y_pred, -2), [rs, bs, cs, cs, ds])
	distance = tf.reduce_sum(tf.square(y_true_bt - y_pred_bt), axis=-1) # rs, bs, cs, cs
	mindis = tf.reduce_min(distance, axis=-1) # rs, bs, cs
	rhloss = tf.reduce_mean(mindis, axis=-1) # rs, bs

	hloss_total = tf.reduce_mean(hloss + rhloss, axis=0)

	return hloss_total

def symmetric_hausdorff_loss(y_true, y_pred):

	y_true = tf.convert_to_tensor(y_true)
	if len(tf.shape(y_pred)) == 3:
		y_true = tf.expand_dims(y_true, 0)
		y_pred = tf.expand_dims(y_pred, 0)

	rs = tf.shape(y_pred)[0]
	bs = tf.shape(y_pred)[1]
	cs = tf.shape(y_pred)[2]
	ds = tf.shape(y_pred)[3]

	y_true_bt = tf.broadcast_to(tf.expand_dims(y_true, -2), [rs, bs, cs, cs, ds])
	y_pred_bt = tf.broadcast_to(tf.expand_dims(y_pred, -3), [rs, bs, cs, cs, ds])
	distance = tf.reduce_sum(tf.square(y_true_bt - y_pred_bt), axis=-1) # rs, bs, cs, cs
	mindis = tf.reduce_min(distance, axis=-1) # rs, bs, cs
	# mindis = tf.sqrt(mindis) # checking with tfg function
	closs = tf.reduce_max(mindis, axis=-1) # rs, bs

	y_true_bt = tf.broadcast_to(tf.expand_dims(y_true, -3), [rs, bs, cs, cs, ds])
	y_pred_bt = tf.broadcast_to(tf.expand_dims(y_pred, -2), [rs, bs, cs, cs, ds])
	distance = tf.reduce_sum(tf.square(y_true_bt - y_pred_bt), axis=-1) # rs, bs, cs, cs
	mindis = tf.reduce_min(distance, axis=-1) # rs, bs, cs
	# mindis = tf.sqrt(mindis) # checking with tfg function
	rcloss = tf.reduce_max(mindis, axis=-1) # rs, bs

	closs_total = tf.reduce_mean(closs + rcloss, axis=0)

	return closs_total

class SymmetricReconstructionLoss(losses.Loss):

	def call(self, y_true, y_pred):
		
		closs = symmetric_chamfer_loss(y_true, y_pred)
		hloss = symmetric_hausdorff_loss(y_true, y_pred)

		reconstruction_loss = closs + hloss

		return reconstruction_loss

class WeightedReconstructionLoss(losses.Loss):

	def __init__(self, radius_cut=np.inf):
		super().__init__()
		self.radius_cut = radius_cut

	def call(self, y_true, y_pred):

		wcloss = weighted_chamfer_loss(self.radius_cut)(y_true, y_pred)
		whloss = weighted_hausdorff_loss(self.radius_cut)(y_true, y_pred)
		weighted_reconstruction_loss = wcloss + 0.1 * whloss
		
		return weighted_reconstruction_loss

class CombinedReconstructionLoss(losses.Loss):

	def __init__(self, radius_cut=np.inf, rloss_weight=1, wloss_weight=1):
		super().__init__()
		self.radius_cut = radius_cut
		self.rloss_weight = rloss_weight
		self.wloss_weight = wloss_weight

	def call(self, y_true, y_pred):

		closs = symmetric_chamfer_loss(y_true, y_pred)
		hloss = symmetric_hausdorff_loss(y_true, y_pred)
		reconstruction_loss = closs + 0.1 * hloss

		wcloss = weighted_chamfer_loss(self.radius_cut)(y_true, y_pred)
		whloss = weighted_hausdorff_loss(self.radius_cut)(y_true, y_pred)
		weighted_reconstruction_loss = wcloss + 0.1 * whloss
		
		return self.rloss_weight * reconstruction_loss + self.wloss_weight * weighted_reconstruction_loss