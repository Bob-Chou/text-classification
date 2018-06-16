import collections
import os
import pickle as pk
import argparse

def derive_vocab(corpus, num=None):
	print('Deriving vocabulary from corpus ...')
	counter = collections.Counter(corpus)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	if num is not None and num<len(count_pairs):
		count_pairs = count_pairs[:num]
	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))

	return word_to_id

def derive_dict(file_list, word_num=None, word_split=' '):
	corpus = []
	print('Reading corpus ...')
	for file in file_list:
		if os.path.isfile(file):
			with open(file, 'r', encoding='utf-8') as f:
				for line in f.readlines():
					if bool(line.strip()):
						if bool(word_split):
							corpus.extend(line.strip().split(word_split))
						else:
							corpus.extend(list(line.strip()))
	word_to_id = derive_vocab(corpus, word_num)
	id_to_word = {v: k for k, v in word_to_id.items()}
	return word_to_id, id_to_word

def save_dict(self, checkpoint='./'):
	word_path = os.path.join(checkpoint, 'word2id.pkl')
	id_path = os.path.join(checkpoint, 'id2word.pkl')
	with open(word_path, 'wb') as f:
		pk.dump(word_to_id, f, protocol=1)
	with open(id_path, 'wb') as f:
		pk.dump(id_to_word, f, protocol=1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to the input data directory, containing "pos", "neg"', required=True, type=str)
	parser.add_argument('--output', help='Path to the output word2id/id2word dictionary to be saved', default='./', type=str)
	parser.add_argument('--word_num', help='Maximum number of words (the top-n frequent words will be picked)', default=None, type=int)
	parser.add_argument('--char', help='If split "char"', action='store_true')
	args = parser.parse_args()

	if not os.path.isdir(args.output):
		raise FileNotFoundError('Folders not exist')
	if not args.char:
		file_list=[os.path.join(args.input, 'pos/cut.txt'), os.path.join(args.input, 'neg/cut.txt')]
		word_to_id, id_to_word = derive_dict(file_list)
	else:
		file_list=[os.path.join(args.input, 'pos/data.txt'), os.path.join(args.input, 'neg/data.txt')]
		word_to_id, id_to_word = derive_dict(file_list, word_split='')

	print('Saving the dictionary ...')
	word_path = os.path.join(args.output, 'word2id.pkl')
	id_path = os.path.join(args.output, 'id2word.pkl')
	with open(word_path, 'wb') as f:
		pk.dump(word_to_id, f, protocol=1)
	with open(id_path, 'wb') as f:
		pk.dump(id_to_word, f, protocol=1)

