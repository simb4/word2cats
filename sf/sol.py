import numpy as np
import pandas as pd
import sys
import re

categoires=[]

def training_process(data, wordFrequency):
	print 'start training...'
	# wordFrequency : {word : {category : word frequency in category}}
	print len(data)
	for row in data:
		category = row[0]
		description = row[1]
		if category not in categoires:
			categoires.append(category)
			categoires.sort()
		words = re.split('[,\s]', description)
		for word in words:
			if len(word) < 3:
				continue
			if not wordFrequency.get(word):
				wordFrequency[word] = dict()
			wordFrequency[word][category] = wordFrequency[word].get(category, 0) + 1
	for word in wordFrequency:
		total = 0
		for category in wordFrequency[word]:
			total += wordFrequency[word][category]
		for category in wordFrequency[word]:
			wordFrequency[word][category] = wordFrequency[word][category] * 1. / total
	print 'finished training...'

def test_process(data, wordFrequency):
	print '='*10
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

	for row in data:
		real_category = row[0]
		if real_category not in categoires:
			categoires.append(real_category)
			tp[real_category] = 0
			fp[real_category] = 0
			fn[real_category] = 0
			categoires.sort()
		description = row[1]
		words = re.split('[,\s]', description)

		candidates = dict()
		for word in words:
			if len(word) >= 2:
				if not wordFrequency.get(word):
					continue
				for category in wordFrequency[word]:
					candidates[category] = candidates.get(category, 0.0) + wordFrequency[word][category]

		ans = None
		for category in candidates:
			if not ans or candidates[ans] < candidates[category]:
				ans = category	

		if ans == None:
			bads += 1
			ans = categoires[0]
		# print ans, real_category
		if ans == real_category:
			stat += 1
			tp[ans] += 1
		else:
			fp[ans] += 1
			fn[real_category] += 1

	print 'stats:', stat, 'out of', len(data)
	print 'accuracy', stat * 1. / len(data)

	avgPr = avgRc = 0
	for cat in categoires:
		precision = recall = 1
		if tp[cat] + fp[cat] > 0:
			precision = tp[cat] * 1.0 / (tp[cat] + fp[cat])
		if tp[cat] + fn[cat] > 0:
			recall = tp[cat] * 1.0 / (tp[cat] + fn[cat])
		avgRc += recall
		avgPr += precision

		# print cat + ':', ' '*(30-len(cat)), 'precision = %.4f' % (precision), ' '*5, 'recall = %.4f' % (recall)

	print 'finished test...'
	print '='*10
	stat *= 1. / len(data)
	avgPr *= 1. / len(categoires)
	avgRc *= 1. / len(categoires)
	return stat, avgPr, avgRc

# 0 - Category 		
# 1 - Descript		replace
# 2 - DayOfWeek		
# 3 - PdDistrict	
# 4 - Resolution 	replace
# 5 - Address 	 	remove
# 6 - X 			
# 7 - Y

full_data = pd.read_csv("train.csv", names=["Category", "Descript", "DayOfWeek", "PdDistrict", "Resolution", "Address", "X", "Y"]).as_matrix()
full_data = np.delete(full_data, 0, 0)
N = len(full_data)

Xacc = []
Xpr = []
Xrc = []
Yacc = []
Ypr = []
Yrc = []

np.random.shuffle(full_data)
for p in range(5, 100, 5):
	print p , '% for training'

	M = int(N * p / 100.0)
	wordFrequency = dict()
	training_process(full_data[0:M], wordFrequency)
	acc, pr, rc = test_process(full_data[M:], wordFrequency)

	Xacc.append(M)
	Xpr.append(M)
	Xrc.append(M)
	Yacc.append(acc)
	Ypr.append(pr)
	Yrc.append(rc)

print Xacc
print Xpr
print Xrc
print Yacc
print Ypr
print Yrc


