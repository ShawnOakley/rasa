import tensorflow as tf
x = tf.placeholder(tf.int32, shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3], name='y')

sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='sum_x')

sess = tf.Session()

print ("sum(x): ", sess.run(sum_x, feed_dict={x:[100,200,300]}))