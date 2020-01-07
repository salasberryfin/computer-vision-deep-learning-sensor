import tensorflow as tf

x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x, y), tf.cast(tf.constant(1), tf.float64))


def run():
    with tf.Session() as sess:
        output = sess.run(z)

        return output

out = run()
print(out)
