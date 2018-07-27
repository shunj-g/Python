#encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
#x不是一个输入的特定的值，而是一个占位符placeholder,我们在Tensorflow运行计算是输入这个值
#我们希望能够输入任意数量的Mnist图像，每一张图展开784维的
#向量。使用[None,784]，None表示此张亮的第一个维度可以是任意的长度

#在模型中需要设置权重值和偏置量，当然我们可以吧他们当做另外的输入（使用展位符）
#但是TensorFlow有一个更好的方法来表示他们：Variable。一个Variable代表一个可以修改的
#存在在TensorFlow的用于描述交互操作的图中。它们可以用于计算输入值，也可以在计算中修改
#对于各种机器学习中应用，一般都会有模型参数，可以用Variable表示
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#我们赋予tf.Variable不同的初值来创建不同的Variable：在这里我们使用全零的张量来初始化W,b
#W的维度是[784,10],b的维度是[10]

#模型：
y = tf.nn.softmax(tf.matmul(x,W)+b)

#为了更好的训练模型，定义一个指标来评估这个迷行(一般使用成本(cost)或者损失(loss))
#然后尽量最小化这个指标，，在这里使用"交叉熵"(ccross-entropy)
#交叉熵是用来衡量我们的预测用于描述真相的低效性

#为了计算交叉熵，我们添加一个新的占位符用于输入正确值：
y_ = tf.placeholder("float",[None,10])

#计算交叉熵：
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#这里的tf.log()是计算每个元素的对数

'''
注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。
对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。
'''
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：
init = tf.initialize_all_variables()
#现在我们可以在一个Session里面启动我们的模型，并且初始化变量：
sess = tf.Session()
sess.run(init)

#训练模型，这里我们让模型循环训练1000次！
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))