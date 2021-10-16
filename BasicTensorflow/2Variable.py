import  tensorflow as tf

w=tf.Variable(initial_value=tf.random_normal(shape=(1,4),mean=100,stddev=0.35),name="w")
b=tf.Variable(initial_value=tf.zeros([4]),name="b")
print(b)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run([w,b])
sess.run(b.assign_add([1,1,1,1]))#这个要shape一样 挺傻逼的
print(sess.run(b))


#saver
#指名要保存的变量
saver=tf.train.Saver({'w':w,'b':b})
saver.save(sess,'./summary/test.ckpt',global_step=0)#注意这个0 一般用来表示轮次
sess.run(b.assign_add([1,1,1,1]))
print(sess.run(b))

saver.restore(sess,'./summary/test.ckpt-0') #注意这里有0
print(sess.run(b))
#用来回复数据流图结构 没用过
#tf.train.import_meta_graph