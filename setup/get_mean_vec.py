import gensim
import argparse
import os
import numpy as np
import math

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to the input data directory, containing "train", "test', required=True, type=str)
	parser.add_argument('--output', help='Path to the output mean vec to be saved', default='./', type=str)
	parser.add_argument('--model', help='Path to the word2vec model to be loaded', default='./word2vec.embedding', type=str)
	args = parser.parse_args()
	
	print('Loading word2vec model {} ...'.format(args.model))
	model = gensim.models.KeyedVectors.load_word2vec_format(args.model)
	if not os.path.isdir(os.path.split(args.output)[0]):
		raise EnvironmentError('Cannot find the folder of output.')

	if not os.path.isdir(os.path.join(args.output, 'train/pos')):
		os.makedirs(os.path.join(args.output, 'train/pos'))
	if not os.path.isdir(os.path.join(args.output, 'train/neg')):
		os.makedirs(os.path.join(args.output, 'train/neg'))
	if not os.path.isdir(os.path.join(args.output, 'test/pos')):
		os.makedirs(os.path.join(args.output, 'test/pos'))
	if not os.path.isdir(os.path.join(args.output, 'test/neg')):
		os.makedirs(os.path.join(args.output, 'test/neg'))

	for path in ['train/pos/', 'train/neg/', 'test/pos/', 'test/neg/']:
		print('Computing {} ...'.format(path))
		with open(os.path.join(args.input, path+'cut.txt'), 'r', encoding='utf-8') as f:
			mean_vec = []
			corpus = f.readlines()
			total_line = len(corpus)
			for cur_line, sen in enumerate(corpus):
				words = sen.strip().split(' ')
				if bool(words):
					cnt = 0
					curr_mean_vec = 0
					for word in words:
						if word in model.wv:
							curr_mean_vec += model.wv[word]
							cnt += 1
					mean_vec.append(curr_mean_vec / max(cnt, 1e-7))
				if cur_line % math.floor(total_line*0.01) == 0:
					print('process ... {:.0f}'.format(100*cur_line/total_line), end='\r')
		print()
		with open(os.join(args.output, path+'feature.npy'), 'wb') as f:
			np.save(f, mean_vec)