import argparse as ap
import numpy as np
import os
import tensorflow as tf
import argparse
import pickle as pk

class CNNConfig(object):
	"""docstring for CNNConfig"""
	def __init__(self, args):
		super(CNNConfig, self).__init__()
		self.kernel = tuple(map(int, args.kernel.split(',')))
		self.hidden_dim = args.hidden_dim
		self.learning_rate = args.learning_rate
		self.beta1 = 0.5
		self.keep_prob = args.keep_prob
		self.sequence_length = args.sequence_length
		self.word_num = args.word_num
		self.embed_dim = args.embed_dim
		self.checkpoint = args.checkpoint
		self.save_every = args.save_every
		self.print_every = args.print_every
		self.cuda = args.cuda

class TextCNN(object):
	"""docstring for TextCNN"""
	def __init__(self, config):
		super(TextCNN, self).__init__()
		self.config = config
		self.cuda_config = tf.ConfigProto(allow_soft_placement=True)
		if self.config.cuda:
			self.cuda_config.gpu_options.allow_growth = True
		self._init_graph()

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

	def _init_graph(self):
		print('Begin defining graphs ...')
		# Define the graph of cnn
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.x = tf.placeholder(tf.int32, (None, self.config.sequence_length), name='x')
			self.y = tf.placeholder(tf.int32, (None, ), name='y')
			self.is_training = tf.placeholder(tf.bool, name='is_training')

			# embedding layer
			with tf.device('/cpu:0'), tf.name_scope('embedding'):
				W = tf.get_variable('W', initializer=tf.random_uniform((self.config.word_num, self.config.embed_dim), -1.0, 1.0))
				embedding = tf.expand_dims(tf.nn.embedding_lookup(W, self.x), 2)

			# convolution layer
			N, L, _, C= embedding.get_shape()
			conv_outs = []
			with tf.name_scope('conv_layer'):
				for k in self.config.kernel:
					x = tf.layers.conv2d(embedding, filters=self.config.hidden_dim, kernel_size=(k, 1), strides=(1, 1), padding='valid', activation=tf.nn.relu)
					x = tf.layers.max_pooling2d(x, (L - k + 1, 1), strides=(L - k + 1, 1), padding='valid')
					conv_outs.append(x)

			# concat
			with tf.name_scope('concat'):
				pool_flat = tf.concat(conv_outs, 3)
				pool_flat = tf.reshape(pool_flat, (-1, len(self.config.kernel)*self.config.hidden_dim))

			# dense layer
			with tf.name_scope('dense'):
				fc = tf.nn.relu(tf.layers.dense(pool_flat, 1024))
				fc = tf.contrib.layers.dropout(fc, keep_prob=self.config.keep_prob, is_training=self.is_training)
				logits = tf.layers.dense(fc, 2)

			# loss
			with tf.name_scope('loss'):
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, 2), logits=logits))

			# pred
			with tf.name_scope('pred'):
				self.preds, self.probs = tf.to_int32(tf.argmax(logits, 1)), tf.nn.softmax(logits)

			# training ops
			with tf.name_scope('optimizer'):
				solver = tf.train.AdamOptimizer(self.config.learning_rate, self.config.beta1)
				self.steps = solver.minimize(self.loss)

			# accuracy
			with tf.name_scope('accuracy'):
				TP = tf.cast(tf.equal(self.preds, self.y), tf.float32) * self.y
				self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.preds), tf.float32))
				self.recall = tf.reduce_sum(tf.cast(TP, tf.float32)) / tf.maximum(tf.reduce_sum(self.y), 1e-7)
				self.precision = tf.reduce_sum(tf.cast(TP, tf.float32)) / tf.maximum(tf.reduce_sum(self.preds), 1e-7)

			# saver
			self.saver = tf.train.Saver()

	def train(self, epoch, batch): 
		model_file = self.config.checkpoint
		model_path = os.path.split(self.config.checkpoint)[0]
		if not os.path.isdir(model_path):
			os.makedirs(model_path)
		with tf.Session(config=self.cuda_config, graph=self.graph) as sess:
			try:
				self.saver.restore(sess, tf.train.latest_checkpoint(model_path))
			except:
				print("No model found, using initializer.")
				sess.run(tf.global_variables_initializer())
			# make prediction of test data
			step = 0
			step_per_epoch = self.pos.shape[0] // batch
			for ep in range(epoch):
				for px_, py_, nx_, ny_ in self.train_batch(batch):
					idxs = np.asarray(list(range(len(py_)+len(ny_))))
					np.random.shuffle(idxs)
					x_, y_ = np.concatenate((px_, nx_), axis=0)[idxs, ], np.concatenate((py_, ny_), axis=0)[idxs, ]
					sess.run(self.steps, feed_dict={self.x: x_, self.y: y_, self.is_training: True})
					if step % self.config.print_every == 0:
						loss, acc, rec, prec = sess.run([self.loss, self.acc, self.recall, self.precision], feed_dict={self.x: x_, self.y: y_, self.is_training: False})
						info = '{}/{}[{}]/[{}], L: {:.4f}, A: {:.2f}%, R: {:.2f}%, P: {:.2f}%'.format(step, step_per_epoch, ep, epoch, loss, 100*acc, 100*rec, 100*prec)
						print(info, end='\r')
					if step % self.config.save_every == 0:
						print()
						print('Saving model as {}'.format(model_file))
						self.saver.save(sess, model_file)
					step += 1
				print()
				print('Epoch: {} finished'.format(ep))		
				self.saver.save(sess, model_file)

	def pred(self, data, batch):
		y = []
		with tf.Session(config=self.cuda_config, graph=self.graph) as sess:
			try:
				self.saver.restore(sess, tf.train.latest_checkpoint(os.path.split(self.config.checkpoint)[0]))
			except:
				raise NotImplementedError('No model inside, please train first.')
			for b in test_batch(data, batch=batch):
				y.append(np.asscalar(sess.run(self.probs, feed_dict={self.x: b, self.is_training: False})[:, 1]))
		return y

def convert_to_id(raw, vocab, word_num, sequence_length):
	id_ = []
	total = min(len(vocab), word_num-1)
	for r in raw:
		tmp = [vocab[w] for w in list(r) if w in vocab and vocab[w]<word_num-1]
		if len(tmp) >= sequence_length:
			id_.append(tmp[:sequence_length])
		else:
			tmp += [total]*(sequence_length-len(tmp))
			id_.append(tmp)
	return np.asarray(id_)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to the input data directory, containing "pos", "neg" folders', required=True, type=str)
	parser.add_argument('--output', help='Path to the output prediction to be saved', default='./cnn_pred.txt', type=str)
	parser.add_argument('--vocabulary', help='Path to the vocabulary to be used', default='./word2id.pkl', type=str)
	parser.add_argument('--kernel', help='Kernel to be used', default='5,7,9', type=str)
	parser.add_argument('--embed_dim', help='Dimension of embedding', default=256, type=int)
	parser.add_argument('--word_num', help='Total number of words to be used', default=8192, type=int)
	parser.add_argument('--hidden_dim', help='Dimension of hidden layers', default=1024, type=int)
	parser.add_argument('--sequence_length', help='Maximum length of sequence', default=1024, type=int)
	parser.add_argument('--keep_prob', help='Keeping probability in dropout layer', default=0.5, type=float)
	# parser.add_argument('--word2vec', help='Balance the number of pos and neg data', default=None, type=str)
	parser.add_argument('--learning_rate', help='Learning rate', default=1e-3, type=float)
	parser.add_argument('--checkpoint', help='Checkpoint of cnn model', default='./cnn.ckpt', type=str)
	parser.add_argument('--save_every', help='Per which the model will be saved', default=1000, type=int)
	parser.add_argument('--print_every', help='Per which the training information will be printed', default=1000, type=int)
	parser.add_argument('--cuda', help='If use GPU', action='store_true')
	parser.add_argument('--test', help='If test', action='store_true')

	args = parser.parse_args()

	print('Loading vocabulary ...')
	with open(args.vocabulary, 'rb') as f:
		word2id = pk.load(f)

	print('Loading raw data ...')
	x = []
	with open(os.path.join(args.input, 'pos/data.txt'), 'r', encoding='utf-8') as f:
		raw_data = [line.strip() for line in f.readlines() if bool(line.strip())]
		x.extend(raw_data)
		split_point = len(raw_data)
		# y.extend(len(pos)*[1])
	with open(os.path.join(args.input, 'neg/data.txt'), 'r', encoding='utf-8') as f:
		raw_data = [line.strip() for line in f.readlines() if bool(line.strip())]
		x.extend(raw_data)
		# y.extend(len(neg)*[0])

	print('Converting data ...')
	x = convert_to_id(x, word2id, args.word_num, args.sequence_length)
	print('Coverted data shape: {}'.format(x.shape))
	config = CNNConfig(args)
	text_cnn = TextCNN(config)
	if args.test:
		text_cnn.pred(x)
	else:
		text_cnn.load_datasets(x[:split_point, :], x[split_point:, :])
		text_cnn.train(100, 32)





