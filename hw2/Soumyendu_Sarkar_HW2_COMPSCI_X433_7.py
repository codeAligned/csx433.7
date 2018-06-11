
# coding: utf-8

# In[ ]:


# Soumyendu Sarkar
# Student ID: X123160
# COMPSCI X433.7 Machine Learning With Tensorflow
# Home Work 2
# June 09, 2018

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# for consistent results across runs
np.random.seed(0)

# Seperate input layer
with tf.name_scope("Input_placeholder"):
    # Create a placeholder vector of flexible length with data type float32
    a = tf.placeholder(tf.float32, shape=None, name="Input_a")
    
# Seperate middle layer
with tf.name_scope("Middle_section"):
    # Compute the product b for the input vector
    b = tf.reduce_prod(a, name="prod_b")
    # Compute the mean c for the input vector
    c = tf.reduce_mean(a, name="mean_c")
    # Compute the sum d for the input vector
    d = tf.reduce_sum(a, name="sum_d")
    # Compute e adding b and c
    e = tf.add(b, c, name="add_e")
    
# Seperate output layer
with tf.name_scope("Final_node"):
    # Compute the product f multiplying e and d
    f = tf.multiply(d, e, name="mul_f")
    
#input_data : random normal with mean = 1.0 and std deviation = 2.0 
input_data =np.random.normal(1.0,2.0,100)

# Create a dictionary to pass into 'feed_dict'
input_dict  = {a: input_data}

# execute the graph
with tf.Session() as sess :
    # write graph for Tensorboard
    writer = tf.summary.FileWriter('./summaries', graph=sess.graph)
    writer.flush()
    writer.close()
    # run session
    val = sess.run(f, input_dict)
    print("Value of f : %s" % val)
    
# Plot the distribution of the randomly generated input array by the array index position
plt.title("Distribution of Random Normal Input Values 'a'")
plt.scatter(list(range(0, 100)), input_data, s=20, alpha=0.7)
plt.xlabel("Index")
plt.ylabel("Input Value 'a'")
plt.show()

# Plot histogram for the distribution of the randomly generated input array
plt.hist(input_data, bins=20)
plt.title("Histogram of Random Normal Input Values 'a'")
plt.ylabel("Frequency")
plt.xlabel("Input Value 'a'")
plt.show()

