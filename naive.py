import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
msg = pd.read_csv('document.csv', names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42)
count_vect = CountVectorizer()
Xtrain_dtm = count_vect.fit_transform(Xtrain)
Xtest_dtm = count_vect.transform(Xtest)
clf = MultinomialNB().fit(Xtrain_dtm, ytrain)
predicted = clf.predict(Xtest_dtm)
print("The dimension of the dataset:", msg.shape)
print(X)
print(y)
print("\nThe total no of training data:", ytrain.shape)
print("\nThe total no of testing data:", ytest.shape)
print("\nThe words or tokens in the text document:\n")
print(count_vect.get_feature_names_out())
print("\nAccuracy of classifier:", metrics.accuracy_score(ytest, predicted))
print("\nConfusion matrix:")
print(metrics.confusion_matrix(ytest, predicted))
print("\nThe value of precision:", metrics.precision_score(ytest, predicted))
print("\nThe value of recall:", metrics.recall_score(ytest, predicted))

#output
The dimension of the dataset: (18, 3)
0                      I love this sandwich
1                  This is an amazing place
2        I feel very good about these beers
3                      This is my best work
4                      What an awesome view
5             I do not like this restaurant
6                  I am tired of this stuff
7                    I can't deal with this
8                      He is my sworn enemy
9                       My boss is horrible
10                 This is an awesome place
11    I do not like the taste of this juice
12                          I love to dance
13        I am sick and tired of this place
14                     What a great holiday
15           That is a bad locality to stay
16           We will have good fun tomorrow
17         I went to my enemy's house today
Name: message, dtype: object
0     1
1     1
2     1
3     1
4     1
5     0
6     0
7     0
8     0
9     0
10    1
11    0
12    1
13    0
14    1
15    0
16    1
17    0
Name: labelnum, dtype: int64

The total no of training data: (12,)

The total no of testing data: (6,)

The words or tokens in the text document:

['about' 'am' 'an' 'awesome' 'bad' 'beers' 'boss' 'can' 'dance' 'deal'
 'do' 'enemy' 'feel' 'fun' 'good' 'great' 'have' 'holiday' 'horrible'
 'house' 'is' 'juice' 'like' 'locality' 'love' 'my' 'not' 'of' 'place'
 'stay' 'stuff' 'taste' 'that' 'the' 'these' 'this' 'tired' 'to' 'today'
 'tomorrow' 'very' 'view' 'we' 'went' 'what' 'will' 'with']

Accuracy of classifier: 0.8333333333333334

Confusion matrix:
[[3 0]
 [1 2]]

The value of precision: 1.0

The value of recall: 0.6666666666666666
