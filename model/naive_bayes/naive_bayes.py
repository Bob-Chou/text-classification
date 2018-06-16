from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from scipy import io
from scipy.sparse import vstack
import os
import numpy as np
import argparse as ap
import math

class TextNaiveBayes(object):
	"""docstring for TextNaiveBayes"""
	def __init__(self, check_point='checkpoint/naive_bayes.clf'):
		super(TextNaiveBayes, self).__init__()
		self.check_point = check_point
		self.classifier = None
	
	def train(self, x, y):
		print('Begin training NB ...')
		model_path, model_file = os.path.split(self.check_point)
		self.classifier = MultinomialNB(fit_prior=False).fit(x, y)
		print('Saving NB check_point ...')
		if not os.path.isdir(model_path):
			os.makedirs(model_path)
		joblib.dump(self.classifier, os.path.join(self.check_point))

	def predict(self, x):
		if self.classifier is None:
			try:
				self.classifier = joblib.load(self.check_point)
			except:
				raise NotImplementedError('Please train a check_point first.')
		return self.classifier.predict_proba(x)[:, 1].tolist()

if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument('--input', help='Path to the input data directory, containing "pos", "neg" folders', required=True, type=str)
	parser.add_argument('--output', help='Path to the output to be saved', default='./naive_bayes_pred.txt', type=str)
	parser.add_argument('--checkpoint', help='Pretrained model to be loaded. Only be used in test stage', default='./', type=str)
	parser.add_argument('--feature', help='The maximum features to be used', default=-1, type=int)
	parser.add_argument('--pos_sample', help='The ratio to sample the pos data, can be bigger than 1 (use the floor to duplicate the data)', default=1, type=float)
	parser.add_argument('--neg_sample', help='The ratio to sample the neg data, can be bigger than 1 (use the floor to duplicate the data)', default=1, type=float)
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
	if args.feature > 0:
		pos_x = pos_x[:, :args.feature]
	print('Pos data: '+str(pos_x.shape[0]))

	print('Loading {} data ...'.format('neg'))
	if not args.sparse:
		file = os.path.join(args.input, 'neg/feature.npy')
		with open(file, 'rb') as f:
			neg_x = np.load(f)
	else:
		file = os.path.join(args.input, 'neg/feature.mtx')
		neg_x = io.mmread(file).tocsc()
	if args.feature > 0:
		neg_x = neg_x[:, :args.feature]
	print('Neg data: '+str(neg_x.shape[0]))

	# sample the dataset
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

	x = pos_x + neg_x
	y = pos_y + neg_y

	if args.sparse:
		x = vstack(x)
	else:
		x = np.concatenate(x, axis=0)

	if not args.test:
		model = TextNaiveBayes(check_point=args.checkpoint)
		model.train(x, y)
	else:
		model = TextNaiveBayes(check_point=args.checkpoint)
		y_ = model.predict(x)
		print('pred: '+str(len(y_))+'truth: '+str(len(y)))

		with open(args.output, 'w') as f:
			f.write('pred,truth\n')
			for p, t in zip(y_, y):
				f.write(str(p)+','+str(t)+'\n')