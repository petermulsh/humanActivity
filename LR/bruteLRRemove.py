import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support

"""This function will take a training dataset and attempt taking out each feature out seeing which one is
better to be removed if any. The initial plan was if there was one better removed it will test that array recursively. Once the
best array is found it will return it and its predictive score. However due to time constraint I was limited to finding only one
feature to remove. The code can be easily altered to recursively find the best combination of all features."""

def bestPrecision(train, test, features, score):
		kfeatures = features[0:len(features) -2]
		for i in kfeatures:
				cur_features = list(features)
				cur_features.remove(i)
				if(len(cur_features)-2 <= 0):
						break
				new_train = train[cur_features]
				array = new_train.values
				info = array[:,0:len(cur_features)-2]
				activity = array[:,len(cur_features)-1]

				new_test = test[cur_features]
				arr2 = new_test.values
				inf2 = arr2[:,0:len(cur_features)-2]
				act2 = arr2[:,len(cur_features)-1]

				lr = LogisticRegression()
				lr.fit(info, activity)
				predictions = lr.predict(inf2)
				temp_score = precision_recall_fscore_support(act2, predictions, average = 'weighted')[0]
				if temp_score > score:
						score = temp_score
						new_features = cur_features
						print(len(new_features))
						print(list(set(list(train[0:len(train)-1])).symmetric_difference(new_features)))
						change = True
		return new_features

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

score = 0
features = list(train)
features = bestPrecision(train,test,features,score)
train = train[features]
array = train.values
info = array[:,0:len(features)-2]
activity = array[:,len(features)-1]

test = test[features]
arr2 = test.values
inf2 = arr2[:,0:len(features)-2]
act2 = arr2[:,len(features)-1]


lr = LogisticRegression()
lr.fit(info, activity)

predictions = lr.predict(inf2)

activity_names = ['STANDING', 'SITTING', 'LAYING', 'WALKING',
									'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']
print(classification_report(act2, predictions))
print(precision_recall_fscore_support(act2,predictions,average = 'weighted')[0])
print(list(set(list(train[0:len(train)-1])).symmetric_difference(set(features))))
