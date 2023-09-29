#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Names : Niharika Gadhave.


# In[55]:


from sklearn import svm 
import tensorflow as tf 
import os
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix


# In[ ]:


C = 1.0
model = svm.SVC(kernel='linear', C=C).fit(X, y) 


# In[ ]:


epochs = range(30)
learning_rate = 0.1 
for epoch in epochs:
    with tf.GradientTape() as tape:
    loss = compute_loss(outputs, model(inputs)) 
    dW, db = tape.gradient(loss, [model.W, model.b])


# In[56]:


def make_dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    list_to_remove=[k for k in dictionary]
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    print(dictionary)
    return dictionary


# In[ ]:


train_dir="C:\\Users\\Niharika\\Downloads\\ML Assignment\\spambase"
dictionary=make_dictionary(train_dir)


# In[59]:


def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
        with open(fil) as fi:
            for i,line in enumerate(fi):
                if i == 2:
                words = line.spilt()
                for word in words:
                    wordID = 0
                for i,d in enumerate(dictionary):
                    if d[0] == word:
                    wordID = 1
                    features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    print(features_matrix)
    return features_matrix


# In[ ]:


def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    prediction = model.predict(sequence)[0]
    return int2label[np.argmax(prediction)]


# In[ ]:


test_dir = "C:\\Users\\Niharika\\Downloads\\ML Assignment\\spambase"
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1


# In[ ]:


model1 = LinearSVC()
model1.fit(train_matrix,train_labels)


# In[ ]:


result1 = model1.predict(test_matrix)
print(confusion_matrix(test_labels,result1))


# In[ ]:


def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    prediction = model.predict(sequence)[0]
    return int2label[np.argmax(prediction)]


# In[62]:


text = "Get A+ in Machine Learning in Cyber Security, pay $1000"
get_predictions(text)


# In[64]:


text = "Hey John, How're you?"
get_predictions(text)

