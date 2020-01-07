#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import tensorflow as tf

# Load pickled data
with open('small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)    


# In[4]:


# split data
X_train, y_train= data['features'], data['labels']


# In[3]:


# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D


# In[12]:


# TODO: Build Convolutional Neural Network in Keras Here
num_classes=5
input_shape = 32, 32, 3

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))


# In[13]:


# Preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)


# In[14]:


# compile and train model
# Training for 3 epochs should result in > 50% accuracy
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
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




