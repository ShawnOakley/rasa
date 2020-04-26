import tensorflow as tf

a = tf.constant(6, name='constnat_a')
b = tf.constant(3, name='constant_b')
c = tf.constant(10, name='constant_c')
d = tf.constant(5, name='constant_d')

mul = tf.multiply(a, b, name='mul')
div = tf.divide(c,d, name='div')
addn = mul + div

# Launch the graph in a session.
sess = tf.compat.v1.Session()

# Evaluate the tensor `mul`.
print(sess.run(addn))
