'''
######################使用一般的是模型进行训练###############################
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#tensorflow依赖于一个高效从C++后端进行计算，与后端的连接叫做Session
#一般而言，使用Tensorflow程序的流程就是创建一个图，然后再session中启动它
#通过InteractiveSession类，可以更加灵活的构建你的代码。它能够在你运行图的时候，
# 插入一些计算图，这些计算图由某些operation构成的，这对于工作中交互式环境中的人们来说非诚便利

import tensorflow as tf
sess = tf.InteractiveSession()
#计算图，为了在python中进行高效的计算，我们通常会使用Numpy这种库
#但是会产生一些消耗，Tensorflow是要求我们描述一个交互操作图，然后完全将其运行python外部
#这个与Theano或者Torch的做法相似。
#由此可知python代码的目的是来构建这个在外部运行的计算图，以及安排计算图的那一部分的应该被运行

#占位符
#通过输入图像和目标函数输出来构建节点，来开始构建计算图
x = tf.placeholder('float',shape=[None,784])
y_ = tf.placeholder('float',shape=[None,10])

#这里的x,y_并不是特定的值，相反，他们都只是一个占位符，可以在Tensorf运行某个计算时根据该占位符输入具有体的值
#为模型定义一个权重w和偏置b。可以将它们当做额外的输入量，但是Tensorflow有一个更好的处理方式，变量
#一个变量代表着Tensorflow计算图中的一个值，能够在计算过程中使用，甚至进行修改，在机器学习过程中模型参数一般使用Variable
#来表示
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#调用tf.Variable的时候传入初始值。在这个例子里，我们把W和b都初始化为零向量。W是一个784x10的矩阵
#（因为我们有784个特征和10个输出值）。b是一个10维的向量（因为我们有10个分类）

#变量需要通过seesion初始化后，才能在session中使用。这一初始化步骤为，为初始值指定具体值（本例当中是全为零），
# 并将其分配给每个变量,可以一次性为所有变量完成此操作。
sess.run(tf.initialize_all_variables())

#类别预测和损失函数
#现在我们可以实现我们的回归模型了，这只需要这一行
y = tf.nn.softmax(tf.matmul(x,W) + b)

#可以很容易为训练过程中指定最小化误差的损失函数，我们的损失函数式目标类别和预测类别之间的
#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了。我们计算的交叉熵是指整个minibatch的。
#训练模型
#因为TensorFlow知道整个计算图，它可以使用自动微分法找到对于各个变量的损失的梯度值。
# TensorFlow有大量内置的优化算法 ，在这里，我们用最速下降法让交叉熵下降，步长为0.01.

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#返回的train_step操作对象，在运行时会使用梯度下降来更新参数。因此，
# 整个模型的训练可以通过反复地运行train_step来完成。
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
#每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，
# 并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
#注意，在计算图中，你可以用feed_dict来替代任何张量，并不仅限于替换占位符。

#评估模型
#首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，
# 它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，
# 因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
# 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

#这里返回一个布尔数组，为了计算我们分类的准确率，我们将布尔值转换为浮点数来代表对错，然后去取平均值
# [True,False,True,True]变为[1,0,1,1],计算出平均值为0.75

accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
#最后，我们可以计算出在测试数据上的准确率，大约为91%

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
'''
######################使用卷积模型网络进行训练###############################
'''
#构造一个多层卷积网络
#在MNIST上只有91%的准确率，很糟糕，，那么使用复杂的模型：卷积神经网络改善效果

#权重初始化
#为了创建这个模型，我们需要创建大量的权重和偏置项。这个模型中权重在初始化时应该
#加入少量的噪声来打破对称性以及避免0梯度。由于我们使用的ReLUctant神经元，因此比较好的做法使用一个较小的正数来初始化偏置项，
#以及避免神经元节点输出恒为0的问题(dead neurons).为了不在简历模型的时候反复做初始化操作，我们定义两个函数用于初始化。

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积和池化
#TensorFlow在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大？在这个实例里，我们会一直使用vanilla版本。
# 我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。我们的池化用简单传统
# 的2x2大小的模板做max pooling。为了代码更简洁，我们把这部分抽象成一个函数。

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层卷积
#现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。卷积的权重张量
# 形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都
# 有一个对应的偏置量。
#
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

#为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里
# 的通道数为1，如果是rgb彩色图，则为3)。

x_image = tf.reshape(x,[-1,28,28,1])
#我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
#为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
#现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层
# 输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
#为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
# 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的
# 输出外，还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。

keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
#最后，我们添加一个softmax层，就像前面的单层softmax regression一样。

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练和评估模型
#这个模型的效果如何呢？
#为了进行训练和评估，我们使用与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，
# 只是我们会用更加复杂的ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob
# 来控制dropout比例。然后每100次迭代输出一次日志。
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
sess.run(tf.initialize_all_variables())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print ('step %d, training accuracy %g'%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print('test accuracy %g'%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))