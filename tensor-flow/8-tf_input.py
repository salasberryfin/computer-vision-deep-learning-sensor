import tensorflow as tf

x = tf.placeholder(tf.int32)

def run():
    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 123})

    return output


try:
    out = run()
    print(out)
except Exception as err:
    print(str(err))
