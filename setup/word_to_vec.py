import gensim
import os
import pickle as pk
import argparse
import numpy as np
import math
class Sentences(object):
	"""docstring for Sentences"""
	def __init__(self, file_list):
		super(Sentences, self).__init__()
		self.file_list = file_list
		self.epoch = 0
	def __iter__(self):
		print('======================== Epoch {} ========================'.format(self.epoch))
		for file in self.file_list:
			with open(file, 'r', encoding='utf-8') as f:
				total_line = len(f.readlines())
			print('Training {} with total sentences {}'.format(file, total_line))
			with open(file, 'r', encoding='utf-8') as f:
				curr_line = 0
				print_per = math.floor(0.01*total_line)
				while True:
					sen = f.readline()
					if bool(sen):
						if curr_line % print_per == 0:
							print('Process: {:.0f}%'.format(100*curr_line/total_line), end='\r')
						yield sen.strip().split(' ')
					else:
						print()
						print('Training {} finished'.format(file))
						break
					curr_line += 1
		print('==================== Epoch {} finished ===================='.format(self.epoch))
		self.epoch += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to the input data directory, containing "pos", "neg"', required=True, type=str)
	parser.add_argument('--output', help='Path to the output word2vec to be saved', default='./', type=str)
	parser.add_argument('--vocabulary', help='Path to the input word2id dictionary to be loaded', default='./word2id.pkl', type=str)
	parser.add_argument('--dim', help='Dimension of word2vec', default=256, type=int)
	parser.add_argument('--epoch', help='Maximum epoches of training', default=5, type=int)
	parser.add_argument('--window', help='Window size of word2vec', default=10, type=int)
	parser.add_argument('--min_count', help='Min_count of word2vec', default=10, type=int)
	parser.add_argument('--workers', help='Parallel cpu threads workers to train word2vec model', default=1, type=int)
	args = parser.parse_args()
	
	if os.path.isdir(args.output):
		raise EnvironmentError('The output is a name of folder.')

	with open(args.vocabulary, 'rb') as f:
		vocab = pk.load(f)

	print('Training word2vec model ...')
	file_list = [os.path.join(args.input, 'pos/cut.txt'), os.path.join(args.input, 'neg/cut.txt')]
	model = gensim.models.Word2Vec(Sentences(file_list), size=args.dim, window=args.window, min_count=args.min_count, workers=args.workers, iter=args.epoch)
	model.wv.save_word2vec_format(args.output)
