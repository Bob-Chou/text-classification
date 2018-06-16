from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy import io
import argparse as ap
import numpy as np
import os
import pickle as pk

def extract_feature(corpus, vocabulary, max_features=None):
	vectorizer = CountVectorizer(vocabulary=vocabulary, max_features=max_features)
	transformer = TfidfTransformer()
	features = transformer.fit_transform(vectorizer.fit_transform(corpus))
	return features

if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument('--input', help='Path to the input data directory, containing "pos", "neg" and "test" folders', required=True, type=str)
	parser.add_argument('--output', help='Path to the output to be saved', default='./', type=str)
	parser.add_argument('--vocabulary', help='Path to the word2id vocabulary to be loaded', required=True, type=str)

	args = parser.parse_args()

	print('Loading vocabulary ...')
	with open(args.vocabulary, 'rb') as f:
		word2id = pk.load(f)

	if not os.path.isdir(args.output):
		os.makedirs(args.output)

	for path in ['train/pos', 'train/neg', 'test/pos', 'test/neg']:
		# Load data
		file = os.path.join(args.input, path+'/cut.txt')
		if os.path.isfile(file):
			print('Loading {} data ...'.format(path))
			with open(file, 'r', encoding='utf-8') as f:
				data = [line.strip() for line in f.readlines() if bool(line.strip())]
			print('Extracting features of {} data ...'.format(path))
			feature = extract_feature(data, word2id)
			print('Saving {} features ...'.format(path))

			out_path = os.path.join(args.output, path)
			if not os.path.isdir(out_path):
				os.makedirs(out_path)
			out = os.path.join(out_path, 'feature.mtx')
			io.mmwrite(out, feature)
