import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc
import argparse
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to the input test data directory, containing "pos", "neg" folders', required=True, type=str)
	parser.add_argument('--prediction', help='Path to the input prediction directory', required=True, type=str)
	parser.add_argument('--name', help='Model name to plot on figure', default='Classifier ROC', type=str)
	args = parser.parse_args()

	y = []
	for path in ('pos/data.txt', 'neg/data.txt'):
		file = os.path.join(args.input, path)
		with open(file, 'r', encoding='utf-8') as f:
			if path=='pos/data.txt':
				y.extend([1] * len(f.readlines()))
			else:
				y.extend([0] * len(f.readlines()))
	preds = args.prediction.strip().split(',')
	names = args.name.strip().split(',')

	fpr, tpr, threshold, roc_auc = [], [], [], []

	for pred in preds:
		with open(pred, 'r') as f:
			y_ = [float(line.strip()) for line in f.readlines() if bool(line.strip())]
			# Compute ROC curve and ROC area for each class  
			roc = roc_curve(y, y_)
			roc_auc.append(auc(roc[0], roc[1]))
			fpr.append(roc[0])
			tpr.append(roc[1])
			threshold.append(roc[2])
	lw = 2
	plt.figure(figsize=(5,5))
	for f, t, a, name in zip(fpr, tpr, roc_auc, names):
		plt.plot(f, t, lw=lw, label=name + ' (AUC = {:.4f})'.format(a))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Random Baseline (AUC = 0.5000)', linestyle='--')  
	plt.xlim([0.0, 1.05])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve of Model(s)')
	plt.legend(loc="lower right")
	plt.grid()
	plt.show()