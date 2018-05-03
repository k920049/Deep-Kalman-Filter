import tensorflow as tf

x = tf.random_normal(shape=[10, 32, 32, 3])
y = tf.random_normal(shape=[5, 32, 32, 3])

with tf.variable_scope(name_or_scope="convolution", reuse=tf.AUTO_REUSE):
    conv1 = tf.layers.conv2d(inputs=x, filters=3, kernel_size=[3, 3], padding='same', name='conv')
    conv2 = tf.layers.conv2d(inputs=y, filters=3, kernel_size=[3, 3], padding='same', name='conv')

with tf.name_scope("miscellaneous"):
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print([x.name for x in tf.trainable_variables(scope="convolution")])