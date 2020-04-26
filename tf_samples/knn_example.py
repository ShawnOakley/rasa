import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

training_digits, training_labels = mnist.train.next_batch(5000)
test_digits, test_labels = mnist.next_batch(200)

# First index is number of images, second is total pixels in image
# i.e., width * height
training_digits_pl = tf.placeholder("float", [None, 784])

test_digit_pl = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 distance
# Calculate distance for every element in vector
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digit_pl)))

# Find distance using distance of each pixel, summed
distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction: Get min distance index (Nearest Neighbor)
pred = tf.arg_min(distance, 0)

# Run the model

accuracy = 0.

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(test_digits)):
        nn_index = sess.run(pred, \
                            feed_dict={training_digits_pl: training_digits, test_digit_pl: test_digits[i, :]})

        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "PredictionL:", np.argmax(training_labels[nn_index]), \
              "True label:", np.argmax(test_labels[i]))