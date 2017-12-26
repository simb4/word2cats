import numpy as np
import pandas as pd
import sys
import re
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import nltk
from nltk.corpus import stopwords # Import the stop word list
stops = set(stopwords.words("english"))

DESC = 'text'
CAT = 'author'
full_data = pd.read_csv("train.csv", names=["id","text","author"])
# full_data = np.delete(full_data, 0, 0)
N = len(full_data)

def review_to_words( text ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  


# Initialize an empty list to hold the clean reviews
def training(train):
	clean_train_reviews = []
	for row in train[DESC]:
		clean_train_reviews.append( review_to_words( row ) )

	print "Creating the bag of words..."
	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.  
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             tokenizer = None,    \
	                             preprocessor = None, \
	                             stop_words = None,   \
	                             max_features = 5000) 

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	train_data_features = train_data_features.toarray()

	print "Training the random forest..."
	forest = RandomForestClassifier(n_estimators = 10)
	# Fit the forest to the training set, using the bag of words as 
	# features and the sentiment labels as the response variable
	# This may take a few minutes to run
	forest = forest.fit( train_data_features, train[CAT] )
	return forest, vectorizer

def testing(test, forest, vectorizer):
	# Verify that there are 25,000 rows and 2 columns
	print test.shape
	# Create an empty list and append the clean reviews one by one
	num_reviews = len(test[DESC])
	clean_test_reviews = [] 
	print "Cleaning and parsing the test set movie reviews...\n"
	cnt = 0
	for row in test[DESC]:
		cnt += 1
		if cnt % 10000 == 0:
			print "Review %d of %d" % (cnt, num_reviews)
		clean_test_reviews.append( review_to_words( row ) )

	# Get a bag of words for the test set, and convert to a numpy array
	test_data_features = vectorizer.transform(clean_test_reviews)
	test_data_features = test_data_features.toarray()

	# Use the random forest to make sentiment label predictions
	print 'predicting result...'
	results = forest.predict(test_data_features)
	return get_stat(test, results)

def get_stat(test, results):
	# results = pd.read_csv("result.csv", names=[CAT,])[CAT].as_matrix()
	categoires = set(test[CAT].as_matrix()).union(set(results))
	accuracy = 0
	tp = {}
	fp = {}
	fn = {}
	for cat in categoires:
		tp[cat] = 0
		fp[cat] = 0
		fn[cat] = 0

	answers = test[CAT].as_matrix()
	L = len(answers)
	for i in range(0, L):
		ans = results[i]
		real_category = answers[i]
		if (i+1) % 10000 == 0:
			print "stats %d of %d" % (i+1, L)
		if ans == real_category:
			accuracy += 1
			tp[ans] += 1
		else:
			fp[ans] += 1
			fn[real_category] += 1


	avgRc = 0
	avgPr = 0
	for cat in categoires:
		precision = recall = 0
		if tp[cat] + fp[cat] > 0:
			precision = tp[cat] * 1.0 / (tp[cat] + fp[cat])
		if tp[cat] + fn[cat] > 0:
			recall = tp[cat] * 1.0 / (tp[cat] + fn[cat])
		avgRc += recall
		avgPr += precision
	accuracy *= 1. / len(results)
	avgPr *= 1. / len(categoires)
	avgRc *= 1. / len(categoires)
	return accuracy, avgPr, avgRc


X = []
Yacc = []
Ypr = []
Yrc = []
# np.random.shuffle(full_data)
for p in range(5, 100, 5):
	print p , '% for training'

	M = int(N * p / 100.0)
	forest, vectorizer = training(full_data[0:M])
	acc, pr, rc = testing(full_data[M:], forest, vectorizer)
	print p, acc, pr, rc
	X.append(p)
	Yacc.append(acc)
	Ypr.append(pr)
	Yrc.append(rc)


print X
print Yacc
print Ypr
print Yrc

output = pd.DataFrame( data={"X": X, "Yacc": Yacc, "Ypr": Ypr, "Yrc": Yrc} )
output.to_csv( "result_forest.csv", index=False, quoting=3 )






