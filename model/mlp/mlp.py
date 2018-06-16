import numpy as np
import os
import math
from scipy import io
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
import tensorflow as tf
import argparse
import random

class MLPConfig(object):
	"""docstring for RNNConfig"""
	def __init__(self, args, feature):
		super(MLPConfig, self).__init__()
		# self.feature = args.feature
		self.hidden_dim = args.hidden_dim
		self.layers = args.layers
		self.learning_rate = args.learning_rate
		self.beta1 = 0.5
		self.feature = feature
		self.keep_prob = args.keep_prob
		self.checkpoint = args.checkpoint
		self.save_every = args.save_every
		self.print_every = args.print_every
		self.cuda = args.cuda

class MLP(object):
	"""docstring for MLP"""
	def __init__(self, config):
		super(MLP, self).__init__()
		self.config = config
		self.cuda_config = tf.ConfigProto(allow_soft_placement=True)
		if self.config.cuda:
			self.cuda_config.gpu_options.allow_growth = True
		self._init_graph()

	def _init_graph(self):
		print('Begin defining graphs ...')
		# Define the graph of cnn
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.x = tf.placeholder(tf.float32, (None, self.config.feature), name='x')
			self.y = tf.placeholder(tf.int32, (None, ), name='y')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

			# dense layer
			with tf.name_scope('dense'):
				fc = self.x
				for l in range(self.config.layers):
					fc = tf.nn.relu(tf.layers.dense(fc, self.config.hidden_dim))

			# droup out
			with tf.name_scope('drop_out'):
				drop = tf.contrib.layers.dropout(fc, keep_prob=self.keep_prob)
				logits = tf.layers.dense(drop, 2)

			# loss
			with tf.name_scope('loss'):
				self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, 2), logits=logits))

			# pred
			with tf.name_scope('pred'):
				self.preds, self.probs = tf.to_int32(tf.argmax(logits, 1)), tf.nn.softmax(logits)

			# training ops
			with tf.name_scope('optimizer'):
				solver = tf.train.AdamOptimizer(self.config.learning_rate, self.config.beta1)
				self.steps = solver.minimize(self.loss)

			# accuracy
			with tf.name_scope('accuracy'):
				TP = tf.cast(tf.equal(self.preds, self.y), tf.float32) * tf.cast(self.y, tf.float32)
				self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.preds), tf.float32))
				self.recall = tf.reduce_sum(tf.cast(TP, tf.float32)) / tf.maximum(tf.reduce_sum(tf.cast(self.y, tf.float32)), 1e-7)
				self.precision = tf.reduce_sum(tf.cast(TP, tf.float32)) / tf.maximum(tf.reduce_sum(tf.cast(self.preds, tf.float32)), 1e-7)

			# saver
			self.saver = tf.train.Saver()

	def load_datasets(self, pos, neg):
		self.pos = pos
		self.neg = neg

	def train_batch(self, batch):
		n_pos, n_neg = self.pos.shape[0], self.neg.shape[0]
		pos_idx, neg_idx = np.random.permutation(n_pos), np.random.permutation(n_neg)
		pos_start, neg_start = 0, 0
		while pos_start < n_pos:
			pos_end = min(pos_start + batch, n_pos)
			neg_end = neg_start + pos_end - pos_start
			if neg_end > n_neg:
				neg_idx = np.random.permutation(n_neg)
				neg_start = 0
				neg_end = pos_end - pos_start
			pos = self.pos[pos_idx[pos_start:pos_end]]
			neg = self.neg[neg_idx[neg_start:neg_end]]
			yield pos, np.asarray([1]*pos.shape[0]), neg, np.asarray([0]*neg.shape[0])
			pos_start = pos_end
			neg_start = neg_end

	def test_batch(self, data, batch):
		data = np.asarray(data)
		n = data.shape[0]
		start = 0
		while start < n:
			end = min(start + batch, n)
			minibatch = data[start:end]
			yield minibatch
			start = end

	def train(self, epoch, batch, override=False):
		model_file = self.config.checkpoint
		model_path = os.path.split(self.config.checkpoint)[0]
		if not os.path.isdir(model_path):
			os.makedirs(model_path)
		with tf.Session(config=self.cuda_config, graph=self.graph) as sess:
			if override:
				sess.run(tf.global_variables_initializer())
			else:
				try:
					self.saver.restore(sess, tf.train.latest_checkpoint(model_path))
				except:
					print("Start a new model")
					sess.run(tf.global_variables_initializer())
			step = 0
			step_per_epoch = self.pos.shape[0] // batch
			for ep in range(epoch):
				for px_, py_, nx_, ny_ in self.train_batch(batch):
					idxs = np.asarray(list(range(len(py_)+len(ny_))))
					np.random.shuffle(idxs)
					x_, y_ = np.concatenate((px_, nx_), axis=0)[idxs, ], np.concatenate((py_, ny_), axis=0)[idxs, ]
					sess.run(self.steps, feed_dict={self.x: x_, self.y: y_, self.keep_prob: self.config.keep_prob})
					if step % self.config.print_every == 0:
						loss, acc, rec, prec = sess.run([self.loss, self.acc, self.recall, self.precision], feed_dict={self.x: x_, self.y: y_, self.keep_prob: 1.0})
						info = '{}/{}[{}]/[{}], L: {:.4f}, A: {:.2f}%, R: {:.2f}%, P: {:.2f}%'.format(step, step_per_epoch, ep+1, epoch, loss, 100*acc, 100*rec, 100*prec)
						print(info, end='\r')
					if step % self.config.save_every == 0:
						print()
						print('Saving model as {}'.format(model_file))
						self.saver.save(sess, model_file)
					step += 1
				print()
				print('Epoch: {} finished'.format(ep+1))		
				self.saver.save(sess, model_file)

	def predict(self, data, batch):
		y = []
		with tf.Session(config=self.cuda_config, graph=self.graph) as sess:
			try:
				self.saver.restore(sess, tf.train.latest_checkpoint(os.path.split(self.config.checkpoint)[0]))
			except:
				raise NotImplementedError('No model inside, please train first.')
			for b in self.test_batch(data, batch=batch):
				y.extend(sess.run(self.probs, feed_dict={self.x: b, self.keep_prob: 1.0})[:, 1].tolist())
		return y

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to the input data directory, containing "pos", "neg" folders', required=True, type=str)
	parser.add_argument('--output', help='Path to the output prediction to be saved', default='./cnn_pred.txt', type=str)
	parser.add_argument('--batch', help='Size of batch to be used in training', default=16, type=int)
	parser.add_argument('--epoch', help='Total number of epoches', default=10, type=int)
	parser.add_argument('--hidden_dim', help='Dimension of hidden layers', default=1024, type=int)
	parser.add_argument('--layers', help='Number of hidden layers', default=2, type=int)
	parser.add_argument('--feature', help='Dimension of features', default=-1, type=int)
	parser.add_argument('--pos_sample', help='The ratio to sample the pos data, can be bigger than 1 (use the floor to duplicate the data)', default=1, type=float)
	parser.add_argument('--neg_sample', help='The ratio to sample the neg data, can be bigger than 1 (use the floor to duplicate the data)', default=1, type=float)
	parser.add_argument('--keep_prob', help='Keeping probability in dropout layer', default=0.5, type=float)
	parser.add_argument('--learning_rate', help='Learning rate', default=1e-3, type=float)
	parser.add_argument('--checkpoint', help='Checkpoint of cnn model', default='./cnn.ckpt', type=str)
	parser.add_argument('--save_every', help='Per which the model will be saved', default=1000, type=int)
	parser.add_argument('--print_every', help='Per which the training information will be printed', default=1000, type=int)
	parser.add_argument('--cuda', help='If use GPU', action='store_true')
	parser.add_argument('--test', help='If test', action='store_true')
	parser.add_argument('--sparse', help='If use sparse matrix', action='store_true')
	args = parser.parse_args()

	# Prepare the training set
	print('Loading {} data ...'.format('pos'))
	if not args.sparse:
		file = os.path.join(args.input, 'pos/feature.npy')
		with open(file, 'rb') as f:
			pos_x = np.load(f)
	else:
		file = os.path.join(args.input, 'pos/feature.mtx')
		pos_x = io.mmread(file).tocsc()
	print('Pos data: '+str(pos_x.shape[0]))

	print('Loading {} data ...'.format('neg'))
	if not args.sparse:
		file = os.path.join(args.input, 'neg/feature.npy')
		with open(file, 'rb') as f:
			neg_x = np.load(f)
	else:
		file = os.path.join(args.input, 'neg/feature.mtx')
		neg_x = io.mmread(file).tocsc()
	print('Neg data: '+str(neg_x.shape[0]))

	if args.feature > 0:
		pos_x = pos_x[:, :args.feature]
		neg_x = neg_x[:, :args.feature]
		feature = args.feature
	else:
		feature = pos_x.shape[-1]

	if not args.test:
		if args.pos_sample >= 1:
			pos_y = [1] * pos_x.shape[0] * math.floor(args.pos_sample)
			pos_x = [pos_x] * math.floor(args.pos_sample)
		else:
			down = math.floor(pos_x.shape[0] * args.pos_sample)
			rand_idx = random.sample(list(range(pos_x.shape[0])), down)
			pos_y = [1] * down
			pos_x = [pos_x[rand_idx, :]]
		if args.neg_sample >= 1:
			neg_y = [0] * neg_x.shape[0] * math.floor(args.neg_sample)
			neg_x = [neg_x] * math.floor(args.neg_sample)
		else:
			down = math.floor(neg_x.shape[0] * args.neg_sample)
			neg_y = [0] * down
			rand_idx = random.sample(list(range(neg_x.shape[0])), down)
			neg_x = [neg_x[rand_idx, :]]
	else:
		pos_y = [1] * pos_x.shape[0]
		neg_y = [0] * neg_x.shape[0]
		pos_x = [pos_x]
		neg_x = [neg_x]

	split_point = len(pos_y)
	x = pos_x + neg_x
	y = pos_y + neg_y

	if args.sparse:
		x = vstack(x)
		x = x.toarray()
	else:
		x = np.concatenate(x, axis=0)
	config = MLPConfig(args, feature)

	# initialize the model
	if not args.test:
		mlp = MLP(config)
		mlp.load_datasets(x[:split_point, :], x[split_point:, :])
		mlp.train(args.epoch, args.batch)
	else:
		mlp = MLP(config)
		y_ = mlp.predict(x, args.batch)
		with open(args.output, 'w') as f:
			for p in zip(y_):
				f.write(str(p)+'\n')