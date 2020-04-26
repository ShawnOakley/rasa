import tensorflow as tf

W = tf.Variable([2.5, 4.0], tf.float32, name='var_W')

# Initialize all variables defined
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Need to pass in init to use variable
    sess.run(init)