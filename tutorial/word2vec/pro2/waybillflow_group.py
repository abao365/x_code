# coding: utf-8

# In[1]:

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import misc
from sklearn import tree
import pydotplus
from IPython.display import Image

#from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, accuracy_score


pd.set_option('display.notebook_repr_html', True)

# make matplotlib graphics to show up inline
get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-white')


# In[2]:

def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names

    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return(graph)


# In[3]:

data = pd.read_csv('data.csv')
data.head()


# In[4]:

data.head()


# In[5]:

train, test = train_test_split(data, test_size = 0.20)
print("Training samples: {}; Test samples: {}".format(len(train), len(test)))
#np.random.seed(10)


# In[6]:

c = tree.DecisionTreeClassifier(min_samples_leaf=3, random_state=10)


# In[7]:

features = ["valence", "energy", "danceability", "speechiness", "acousticness", "instrumentalness", "loudness","duration_ms","liveness","tempo","time_signature","mode","key"]

X = data[features]
y = data["target"]
X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]

dt = c.fit(X_train, y_train)


# In[8]:

y_pred = c.predict(X_test)


# In[9]:

score = accuracy_score(y_test, y_pred) * 100
rounded_score = round(score, 1)
print("Decision Tree Classifier Accuracy: {}%".format(rounded_score))


# In[10]:

dt2 = c.fit(X, y)


# In[11]:

y_pred2 = c.predict(X)


# In[15]:

score = accuracy_score(y, y_pred2) * 100
rounded_score = round(score, 1)
print("Decision Tree Classifier Accuracy: {}%".format(rounded_score))


# In[38]:

#c2 = tree.DecisionTreeClassifier(min_samples_leaf=3, random_state=10)
#kf_10 = KFold(n_splits=10)
#predicted = cross_val_predict(c2, X, y, cv=kf_10, method='predict')
#print ("Error Rate: ", 1-accuracy_score(y, predicted), "\n")
#print (classification_report(y, predicted))


# In[14]:

c2 = tree.DecisionTreeClassifier(min_samples_leaf=3, random_state=10)
kf_10 = KFold(n_splits=10)
predicted = cross_val_predict(c2, X, y, cv=kf_10, method='predict')
print ("Error Rate: ", 1-accuracy_score(y, predicted), "\n")
print (classification_report(y, predicted))


# In[ ]:




# In[10]:

graph = print_tree(c, features, class_names=['No', 'Yes'])
Image(graph.create_png())


# In[11]:

tuned_parameters = [{'min_samples_leaf': [1, 2, 3, 5, 10]},{'max_leaf_nodes': [2, 3, 5, 10]},]
clf = GridSearchCV(tree.DecisionTreeClassifier, tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_train, y_train)
#clf.grid_scores_
#clf.cv_results_
print(clf.cv_results_['mean_test_score'])
print(clf.cv_results_['params'])


# In[12]:

print (c.feature_importances_)


# In[13]:

#drop 5 features


# In[14]:

pos_valence = data[data['target'] == 1]['valence']
neg_valence = data[data['target'] == 0]['valence']
pos_energy = data[data['target'] == 1]['energy']
neg_energy = data[data['target'] == 0]['energy']
pos_dance = data[data['target'] == 1]['danceability']
neg_dance = data[data['target'] == 0]['danceability']
pos_speechiness = data[data['target'] == 1]['speechiness']
neg_speechiness = data[data['target'] == 0]['speechiness']
pos_instrumentalness = data[data['target'] == 1]['instrumentalness']
neg_instrumentalness = data[data['target'] == 0]['instrumentalness']
pos_duration = data[data['target'] == 1]['duration_ms']
neg_duration = data[data['target'] == 0]['duration_ms']
pos_loudness = data[data['target'] == 1]['loudness']
neg_loudness = data[data['target'] == 0]['loudness']
pos_acousticness = data[data['target'] == 1]['acousticness']
neg_acousticness = data[data['target'] == 0]['acousticness']
pos_key = data[data['target'] == 1]['key']
neg_key = data[data['target'] == 0]['key']


# In[15]:

fig2 = plt.figure(figsize=(15, 15))

# Danceability
ax3 = fig2.add_subplot(331)
ax3.set_xlabel('Danceability')
ax3.set_ylabel('Count')
ax3.set_title("Song Danceability Like Distribution")
pos_dance.hist(alpha=0.5, bins=30)
ax4 = fig2.add_subplot(331)
neg_dance.hist(alpha=0.5, bins=30)


# Duration
ax5 = fig2.add_subplot(332)
pos_duration.hist(alpha=0.5, bins=30)
ax5.set_xlabel('Duration (ms)')
ax5.set_ylabel('Count')
ax5.set_title("Song Duration Like Distribution")
ax6 = fig2.add_subplot(332)
neg_duration.hist(alpha=0.5, bins=30)


# Loudness
ax7 = fig2.add_subplot(333)
pos_loudness.hist(alpha=0.5, bins=30)
ax7.set_xlabel('Loudness')
ax7.set_ylabel('Count')
ax7.set_title("Song Loudness Like Distribution")

ax8 = fig2.add_subplot(333)
neg_loudness.hist(alpha=0.5, bins=30)

# Speechiness
ax9 = fig2.add_subplot(334)
pos_speechiness.hist(alpha=0.5, bins=30)
ax9.set_xlabel('Speechiness')
ax9.set_ylabel('Count')
ax9.set_title("Song Speechiness Like Distribution")

ax10 = fig2.add_subplot(334)
neg_speechiness.hist(alpha=0.5, bins=30)

# Valence
ax11 = fig2.add_subplot(335)
pos_valence.hist(alpha=0.5, bins=30)
ax11.set_xlabel('Valence')
ax11.set_ylabel('Count')
ax11.set_title("Song Valence Like Distribution")

ax12 = fig2.add_subplot(335)
neg_valence.hist(alpha=0.5, bins=30)

# Energy
ax13 = fig2.add_subplot(336)
pos_energy.hist(alpha=0.5, bins=30)
ax13.set_xlabel('Energy')
ax13.set_ylabel('Count')
ax13.set_title("Song Energy Like Distribution")

ax14 = fig2.add_subplot(336)
neg_energy.hist(alpha=0.5, bins=30)

# Key
ax15 = fig2.add_subplot(337)
pos_key.hist(alpha=0.5, bins=30)
ax15.set_xlabel('Key')
ax15.set_ylabel('Count')
ax15.set_title("Song Key Like Distribution")

ax15 = fig2.add_subplot(337)
neg_key.hist(alpha=0.5, bins=30)

# Acousticness
ax16 = fig2.add_subplot(338)
pos_acousticness.hist(alpha=0.5, bins=30)
ax16.set_xlabel('Acousticness')
ax16.set_ylabel('Count')
ax16.set_title("Song Acousticness Like Distribution")

ax16 = fig2.add_subplot(338)
neg_acousticness.hist(alpha=0.5, bins=30)

# Instrumentalness
ax17 = fig2.add_subplot(339)
pos_instrumentalness.hist(alpha=0.5, bins=30)
ax17.set_xlabel('Instrumentalness')
ax17.set_ylabel('Count')
ax17.set_title("Song Instrumentalness Like Distribution")

ax17 = fig2.add_subplot(339)
neg_instrumentalness.hist(alpha=0.5, bins=30)


# In[16]:

#keep dancability, loudness, Speechiness, Valence,Energy, Instrumentalness.


# In[25]:

features_rec = ["valence", "energy", "danceability", "speechiness", "instrumentalness", "loudness"]
X_train2 = train[features_rec]
y_train2 = train["target"]
X_test2 = test[features_rec]
y_test2 = test["target"]
X2 = data[features_rec]
y2 = data["target"]


# In[30]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 10)
clf.fit(X_train2, y_train2)


# In[31]:

forest_y_pred = clf.predict(X2)
score = accuracy_score(y2, forest_y_pred) * 100
rounded_score = round(score, 1)
print("Random Forest (n_est: 100) Accuracy: {}%".format(rounded_score))


# In[32]:

clf2 = RandomForestClassifier(n_estimators = 100)
kf_10 = KFold(n_splits=10)
predicted = cross_val_predict(clf2, X2, y2, cv=kf_10, method='predict')
print ("Error Rate: ", 1-accuracy_score(y2, predicted), "\n")
print (classification_report(y2, predicted))


# # Build Ada-boost using scikit-learn

# In[39]:

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[40]:

abc = AdaBoostClassifier()
abc.fit(X_train2, y_train2)
tst_pred = abc.predict(X_test2)
print (np.count_nonzero(tst_pred == y_test2) / float(y_test2.size))


# In[41]:

abc2 = AdaBoostClassifier()
kf_10 = KFold(n_splits=10)
predicted = cross_val_predict(clf2, X, y, cv=kf_10, method='predict')
print ("Error Rate: ", 1-accuracy_score(y, predicted), "\n")
print (classification_report(y, predicted))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

hidden_neuron_nums = list(range(1,10))
#[2,3,4,5,6...9, 10, 20, 30, ... 90, 100, 125, 150, 175]
total_performance_records = []
for hn in hidden_neuron_nums:
    c_ = tree.DecisionTreeClassifier(min_samples_leaf=hn, random_state=10)
    perf_records_ = []
    for i in range(10):
         c_.fit(X_train, y_train)
         tst_p_ = c_.predict(X_test)
         performance = np.sum(tst_p_ == y_test) / float(tst_p_.size)
         perf_records_.append(performance)
    total_performance_records.append(np.mean(perf_records_))
    print ("Evaluate hidden layer {} done, accuracy {:.2f}".format(
        hn, total_performance_records[-1]))


# In[19]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifierssifier(n_estimators = 100)
clf.fit(X_train, y_train)


# In[20]:

forest_y_pred = clf.predict(X_test)
score = accuracy_score(y_test, forest_y_pred) * 100
rounded_score = round(score, 1)
print("Random Forest (n_est: 100) Accuracy: {}%".format(rounded_score))


# In[ ]:

print (c.feature_importances_)


# In[ ]:

speaker_df = data.groupby('artist').count().reset_index()[['artist', 'target']]
speaker_df.columns = ['artist', 'appearances']
speaker_df = speaker_df.sort_values('appearances', ascending=False)
speaker_df.head(10)


# In[ ]:

plt.figure(figsize=(15,5))
sns.barplot(x='artist', y='appearances', data=speaker_df.head(10))
plt.show()


# In[ ]:




# In[ ]:



