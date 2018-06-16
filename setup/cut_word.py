import jieba
import math
import argparse as ap
import os
import logging
import sys

def cut_words(text, stops=[]):
	text = jieba.cut(text)
	segment = [t for t in text if t not in stops]
	return segment

if __name__ == '__main__':
	console = logging.StreamHandler(sys.stderr)
	log = logging.getLogger(__name__)
	log.setLevel(logging.DEBUG)
	log.addHandler(console)

	parser = ap.ArgumentParser()
	parser.add_argument('--input', help='Path to the input data directory, containing "pos", "neg" and "test" folders', required=True, type=str)
	parser.add_argument('--output', help='Path to the output dataset directory to be saved.', default='./', type=str)
	parser.add_argument('--stopword', help='Path to the input stopwords', default=None)
	args = parser.parse_args()

	if not os.path.isdir(args.output):
		os.makedirs(args.output)

	if bool(args.stopword) and type(args.stopword==str) and os.path.isfile(args.stopword):
		with open(args.stopword, 'r', encoding='utf-8') as f:
			stopword = [line.strip() for line in f.readlines() if bool(line.strip())]
		log.debug('Length of stop words list: {}'.format(len(stopword)))
	else:
		stopword = []

	for path in ['test/pos', 'test/neg', 'train/pos', 'train/neg']:
		file = os.path.join(args.input, path+'/data.txt')
		if os.path.isfile(file):
			log.debug('Begin loading {} ... '.format(file))
			with open(file, 'r', encoding='utf-8') as f:
				data = [line.strip() for line in f.readlines() if bool(line.strip())]
			total = len(data)
			log.debug('length of {} data: {}'.format(path, total))
			print_every = math.floor(0.1 * total)
			cut = []
			for i, d in enumerate(data):
				cut.append(' '.join(cut_words(d, stops=stopword)))
				if i % print_every == 0:
					log.debug('Cutting {} data ... {:.0f}% done.'.format(path, 100 * i / total))
			save_path = os.path.join(args.output, path)
			if not os.path.isdir(save_path):
				os.makedirs(save_path)
			save_file = os.path.join(save_path, 'cut.txt')
			log.debug('Saving {} ... '.format(save_file))
			with open(save_file, 'w', encoding='utf-8') as f:
				for c in cut:
					f.write(c+'\n')


