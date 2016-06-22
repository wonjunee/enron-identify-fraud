#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus','eso_deferred_income'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = {}

for k,v in data_dict.iteritems():
	if k != "TOTAL":
		my_dataset[k] = v

print len(my_dataset)

es = []
di = []
for k,v in my_dataset.iteritems():
	if v["exercised_stock_options"] == "NaN":
		es.append(0.0)
	else:	
		es.append(float(v["exercised_stock_options"]))

	if v["deferred_income"] == "NaN":
		di.append(0.0)
	else:
		di.append(float(v["deferred_income"]))

es = np.array(es)
es = np.reshape(es, [145,1])

di = np.array(di)
di = np.reshape(di, [145,1])

### Rescale each feature
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

### deferred_income
di = min_max_scaler.fit_transform(di) 

### exercised_stock_options
es = min_max_scaler.fit_transform(es) 

### Creating a new feature - eso_expenses
tmp_array = np.empty([145,1])
for i in range(145):
    ### If both values are zero then set it 0
    if di[i] * es[i] == 0:
        tmp_array[i,0] = 0
    else:
        tmp_array[i,0] = np.power(es[i] / di[i], .2) * np.power(di[i]*di[i] + es[i]*es[i], .5)

i = 0
for k,v in my_dataset.iteritems():
	my_dataset[k]["eso_deferred_income"] = tmp_array[i]
	i += 1

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Set the classifier
from sklearn.neighbors import KNeighborsRegressor
clf = KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)