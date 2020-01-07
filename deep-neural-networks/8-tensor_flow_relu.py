#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Solution is available in the other "solution.py" tab
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]


# In[4]:


# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]


# In[5]:


# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])


# In[10]:


# TODO: Create Model
hidden_layer = tf.add(tf.matmul(features, hidden_layer_weights), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)

logits = tf.add(tf.matmul(hidden_layer, out_weights), biases[1])


# In[11]:


# TODO: save and print session results on a variable named "output"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(logits)
    print(output)


# ### Running the Grader
# 
# To run the grader below, you'll want to run the above training from scratch (if you have otherwise already ran it multiple times). You can reset your kernel and then run all cells for the grader code to appropriately check that you weights and biases achieved the desired end result.

# In[ ]:


### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(output)
except Exception as err:
    print(str(err))


# In[ ]:




