#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Solution is available in the other "solution.py"
import tensorflow as tf
from test import *
tf.set_random_seed(123456)


hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]


# In[2]:


# set random seed
tf.set_random_seed(123456)


# In[3]:


# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]


# In[4]:


# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])


# In[13]:


# TODO: Create Model with Dropout
keep_prob = tf.placeholder(tf.float32)
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])


# In[16]:


# TODO: save and print session results as variable named "output"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(logits, feed_dict={keep_prob: 0.5})
    print(output)


# ### Running the Grader
# 
# To run the grader below, you'll want to run the above training from scratch (if you have otherwise already ran it multiple times). You can reset your kernel and then run all cells for the grader code to appropriately check that you weights and biases achieved the desired end result.

# In[17]:


### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(output)
except Exception as err:
    print(str(err))


# In[ ]:




