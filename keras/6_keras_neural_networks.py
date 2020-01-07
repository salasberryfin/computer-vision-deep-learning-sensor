#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
import numpy as np
import tensorflow as tf

# Load pickled data
with open('small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)


# In[5]:


# split data
X_train, y_train = data['features'], data['labels']


# In[8]:


# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten


# In[13]:


# TODO: Build the Fully Connected Neural Network in Keras Here
num_classes = 5

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[14]:


# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
# TODO: change the number of training epochs to 3
history = model.fit(X_normalized, y_one_hot, epochs=3, validation_split=0.2)


# In[15]:


### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(model, history)
except Exception as err:
    print(str(err))


# In[ ]:




