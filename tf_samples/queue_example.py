import tensorflow as tf
from PIL import Image

original_image_list = ["./image/1.jpg", "./images/2.jpg"]

# Make a queue of file names including all the images specified
filename_queue = tf.train.string_input_producer(original_image_list)

# Read an entire image file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    # Coordinate the loading of image files
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image_list = [];
    for i in range(len(original_image_list)):
        # Read a whole file from the queue, the first returned value
        # in tuple is filename, which is ignored
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as a JPEG file, this will turn it into
        # a Tensor which we can the use in training
        image = tf.image.decode_jpeg(image_file)

        # Get a tensor of resized image
        # Specify target dimensions
        image = tf.image.resize_images(image, [224, 224])
        image.set_shape([224,224,3])


        # Run resize
        image_array = sess.run(image)

        # Can also turn image into a Tensor using tf.stack
        image_tensor = tf.stack(image_array)

        # add a new dimension to add image id as 4th dimension
        image_list.append(tf.expand_dims(image_tensor, 0))

        #similarly, can turn image_list into tensor
        images_tensor = tf.stack(images_list)
    # Finish off the filename queue coordinator
    coord.request_stop()
    coord.join(threads)