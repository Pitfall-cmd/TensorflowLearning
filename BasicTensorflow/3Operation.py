import  tensorflow as tf

a=tf.constant(1)
b=tf.constant(2);
with tf.Session() as sess:

    print(sess.run(a))
    print(sess.run(b))

x=tf.placeholder(tf.int32,shape=(),name='xx')
y=tf.placeholder(tf.int32,shape=(),name='y')

add=tf.add(x,y)
with tf.Session() as sess:
    print(sess.run(add,feed_dict={x:1,y:2}))
