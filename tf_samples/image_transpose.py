import tensorflow as tf
import matplotlib.image as mp_img
import matpoloylib.pyplot as plot
import os

filename = "./GregHume_Dandelion.JPG"

image = np_img.imread(filename)

print("Image shape: ", image.shape)
print("Image array: ", image)

plot.imshow(image)
plot.show()

x = tf.Variable(image, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # perm uses indices 0,1,2 to signify order as they exist in original
    # rearrange in perm variable to rearrange via traspose
    transpose = tf.transpose(x, perm=[1,0,2])
    # could also use tf.image.transpose_image(x)
    result  = sess.run(transpose)
