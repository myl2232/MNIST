import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)#读取mnist数据
x = tf.placeholder("float",[None,784])#创建占位符x，【60000，{28*28=784}】
w = tf.Variable(tf.zeros([784,10]))#占位符w，权值
b = tf.Variable(tf.zeros([10]))#占位符b，偏移
y = tf.nn.softmax(tf.matmul(x,w)+b)#softmax回归得到的预测概率分布
y_ = tf.placeholder("float",[None,10])#实际概率分布
cross_entropy = -tf.reduce_sum(y_*tf.log(y))#交叉煽
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#梯度下降法以0.01的学习速率最小化交叉熵
init = tf.global_variables_initializer()#开启初始化
sess = tf.Session()
sess.run(init)
for i in range(1000):#训练1000次
    batch_xs,batch_ys = mnist.train.next_batch(100)#每次从随机位置取100个样本
    sess.run(train_step,feed_dict = {x:batch_xs,y_:batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
acrruncy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(acrruncy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))#计算损失