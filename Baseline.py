#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:33:39 2020

@author: annpham
"""


from sklearn.datasets 

import re
import os
from sklearn.model_selection import train_test_split
os.chdir("/Users/annpham/Desktop/Advmachine")
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


#labels = readLines('Data_og_RNN-kode/labels.txt')

labels = open('Data_og_RNN-kode/labels.txt', 'r')
adult = open('Data_og_RNN-kode/adult_texts.txt', 'r')

line_lable = labels.readline()
line_adult = adult.readline()

X=[]
y=[]


while line_lable != "" and line_adult != "":
    tekst=remove_emojis(line_adult[:-1]).lower()
    tekst= re.sub(r'[^\w\s]',' ',tekst)
    tekst=" ".join(tekst.split())
    #tekst= re.sub(' +',' ',tekst)
    tekst= re.sub('_','',tekst)
    if tekst!='':
        if tekst[0]==" ":
            tekst=tekst[1:] 
        X.append(tekst)   
        y.append(line_lable[:-1])
            
    line_lable = labels.readline()
    line_adult = adult.readline()
labels.close()

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0,shuffle=True)

labelsdict=dict()
words_train=dict()
words_test=dict()
target_names=[]
counter=0
targetstrain=[]
targetstest=[]
for i in range(len(X_train)):
    if y_train[i] not in labelsdict:
        target_names.append(y_train[i])
        labelsdict[y_train[i]]=counter
        counter+=1
            
    targetstrain.append(labelsdict[y_train[i]])
    
for i in range(len(X_test)):
    if y_train[i] not in labelsdict:
        target_names.append(y_test[i])
        labelsdict[y_test[i]]=counter
        counter+=1
            
    targetstest.append(labelsdict[y_test[i]])

words_train['data']=X_train
words_train['target']=targetstrain
words_train['target_names']=target_names
words_test['data']=X_test
words_test['target']=targetstest
words_test['target_names']=target_names

#%%
words_train['target_names']

len(words_train['data'])

#len(twenty_train.filenames)

print("\n".join(words_train['data'][0].split("\n")[:3]))

print(words_train['target_names'][words_train['target'][0]])

words_train['target'][:10]
#%%
for t in words_train['target'][:10]:
    print(words_train['target_names'][t])
    
#%%
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(words_train['data'])
X_train_counts.shape
count_vect.vocabulary_.get(u'algorithm')

#%%
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#%%

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, words_train['target'])


docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, words_train['target_names'][category]))
    
    
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(words_train['data'], words_train['target'])
#%%

import numpy as np
#twenty_test = fetch_20newsgroups(subset='test',
#    categories=categories, shuffle=True, random_state=42)
docs_test = words_test['data']
predicted = text_clf.predict(docs_test)
np.mean(predicted == words_test['target'])

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(words_train['data'], words_train['target'])

predicted = text_clf.predict(docs_test)
np.mean(predicted == words_test['target'])

from sklearn import metrics
print(metrics.classification_report(words_test['target'], predicted,
    target_names=words_test['target_names']))

metrics.confusion_matrix(words_test['target'], predicted)




