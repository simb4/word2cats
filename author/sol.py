import numpy as np
import pandas as pd
import sys
import re

# from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize
# import nltk
# nltk.download('punkt')
# ps = PorterStemmer()

ID = 0
TEXT = 1
CATEGORY = 2
FEATURES_ARRAY = ["id","text","author"]
FEATURES_ARRAY_NO_CAT = ["id","text"]

categoires=['EAP','HPL','MWS']

def addWord(word, category, wordFrequency, wordTotal):
	if not wordFrequency.get(word):
		wordFrequency[word] = dict()
	wordFrequency[word][category] = wordFrequency[word].get(category, 0) + 1
	wordTotal[word] = wordTotal.get(word, 0) + 1

def training_process(data, wordFrequency, wordTotal):
	print 'start training...'
	# wordFrequency : {word : {category : word frequency in category}}
	for row in data:
		category = row[CATEGORY]
		if category not in categoires:
			categoires.append(category)
			categoires.sort()
		description = row[TEXT]
		words = re.split('[.?!;\',"\s]', description)
		for word in words:
			if len(word) < 3:
				continue
			addWord(word, category, wordFrequency, wordTotal)
	print 'finished training...'

def test_process(data, wordFrequency, wordTotal, isTesting):
	print '===================================='
	print 'start testing...'
	import re
	stat = 0
	bads = 0
	tp = {}
	fp = {}
	fn = {}
	for cat in categoires:
		tp[cat] = 0
		fp[cat] = 0
		fn[cat] = 0

	def gen_row(id, x, y, z):
		row = list()
		row.append(id)
		row.append(x)
		row.append(y)
		row.append(z)
		return row

	for row in data:
		if isTesting:
			real_category = row[CATEGORY]
		description = row[TEXT]
		words = re.split('[,\s]', description)

		score = dict()
		new_row = list()
		for word in words:
			if not wordFrequency.get(word) or len(word) < 3:
				continue
			for category in wordFrequency[word]:
				wordProportion = wordFrequency[word][category] * 1. / wordTotal[word]
				score[category] = score.get(category, 0.0) + wordProportion
		# total sum of scroing
		scoreSum = 0
		for category in score:
			scoreSum += score[category]
		# get answer, scroing correction
		ans = None
		if scoreSum > 0:
			for category in categoires:
				score[category] = score.get(category, 0.0) * 1. / scoreSum
				if not ans or score[ans] < score[category]:
					ans = category

			for cat in categoires:
				if ans != cat:
					score[cat] /= 2
					score[ans] += score[cat]

			new_row = gen_row(row[ID],
				score['EAP'],
				score['HPL'],
				score['MWS'])
		else:
			bads += 1
			new_row = gen_row(row[ID], 0.33333, 0.33333, 0.33333)
			ans = 'EAP'


		# out testing results
		if isTesting:
			if ans == real_category:
				stat += 1
				tp[ans] += 1
			else:
				fp[ans] += 1
				fn[real_category] += 1
		else:
			result.append(new_row)


	if isTesting:
		print 'stats:', stat, 'out of', len(data)
		stat *= 1. / len(data)
		print 'accuracy', stat
		precision = recall = 0
		for cat in categoires:
			if tp[cat] + fp[cat] > 0:
				precision += tp[cat] * 1. / (tp[cat] + fp[cat])
			if tp[cat] + fn[cat] > 0:
				recall += tp[cat] * 1. / (tp[cat] + fn[cat])
		precision *= 1. / len(categoires)
		recall *= 1. / len(categoires)
	else:
		print 'bad rows', bads, bads * 1.0/len(data)
	print 'finished test...'
	print '===================================='
	return stat, precision, recall

def testing():
	full_data = pd.read_csv("train.csv", names=["id","text","author"], encoding='utf-8').as_matrix()
	full_data = np.delete(full_data, 0, 0)
	N = len(full_data)
	X = []
	Yacc = []
	Ypr = []
	Yrc = []
	for p in range(5, 100, 5):
		M = int(N * p / 100.0)
		print p , '% for training'
		wordFrequency = dict()
		wordTotal = dict()
		training_process(full_data[0:M], wordFrequency, wordTotal)
		test_data = full_data[M:]
		acc, pr, rc = test_process(test_data, wordFrequency, wordTotal, True)

		X.append(p)
		Yacc.append(acc)
		Ypr.append(pr)
		Yrc.append(rc)

	print X
	print Yacc
	print Ypr
	print Yrc

	output = pd.DataFrame( data={"X": X, "Yacc": Yacc, "Ypr": Ypr, "Yrc": Yrc} )
	output.to_csv( "result.csv", index=False, quoting=3 )


def solving():
	full_data = pd.read_csv("train.csv", names=FEATURES_ARRAY, encoding='utf-8').as_matrix()
	full_data = np.delete(full_data, 0, 0)

	wordFrequency = dict()
	wordTotal = dict()
	training_process(full_data, wordFrequency, wordTotal)

	test_data = pd.read_csv("test.csv", names=FEATURES_ARRAY_NO_CAT, encoding='utf-8').as_matrix()
	test_data = np.delete(test_data, 0, 0)

	test_process(test_data, wordFrequency, wordTotal, False)

	out = pd.DataFrame(result, columns=["id", "EAP", "HPL", "MWS"])
	out.to_csv("out.csv",
		index=False,
		columns=["id", "EAP", "HPL", "MWS"],
		quoting=3)


result = list()
testing()
# solving()

