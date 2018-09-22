import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST数据集相关的常数
INPUT_MODE = 784  #输入层的节点数。对于MNIST的数据集，这个等于图片的像素
OUTPUT_NODE = 10  #输出层的节点数。这个等于类别的数目，在MNIST数据集中需要区分的是0~9的10个数字

#配置神经网络的参数
LAYER1_NODE = 500  #隐藏层的节点数，这里使用只有一个隐藏层的网络结构作为样例，这里使用的500个隐藏节点
BATCH_SIZE = 100   #一个训练batch中的训练数据个数。数字越小时，训练过程约接近随机梯度下降：数字越大时，
                   #训练过程约接近梯度下降
LEARNING_RATE_BASE = 0.8    #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率

REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000       #训练轮数
MOVING_AVERAGE_DECAY = 0.99  #华东平均衰减率

#一个辅助函数，给定神经网络的输入和所有参数，计算神经额昂了的前向传播结果，在这里定义了一个使用ReLUctant
#激活函数的三层全连接神经网络，通过加入隐藏层实现了多层网络结构
#通过ReLU激活函数实现了去线性化，在这个函数中支持传入用于计算参数平均值的类
#这样方便在测试时使用欢动平均模型

def inferance(inut_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:#当没有提供华东平均类时，直接使用参数当前的取值
        layer1 = tf.nn.relu(tf.matmul(inut_tensor,weights1)+biases1)
        # 计算隐藏层的向前传播结果，这里使用了ReLU激活函数

        # 计算输出层的前向传播结果，因为在计算损失函数时会一并计算softmax函数
        # 所以这里不需要加入激活函数，而且不加入softmax不会影响预测结果，因为预测时
        # 使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后分类结果
        # 计算没有影响，于是在计算整个神经网络的前向传播时可以不加入最后的softmax层
        return tf.matmul(layer1,weights2)+ biases2

    else: #首先使用avg_class.average函数来计算得出变量的华东平均值
        #然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(inut_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

#模型训练过程
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_MODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y-input')

    #生成隐藏性的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_MODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    #生成参数输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))

    #计算在当前的参数下的神经网络的向前的传播结果，这里给出用于计算华东平均的类为None
    #所以函数不不会使用参数的华东平均值
    y = inferance(x,None,weights1,biases1,weights2,biases2)

    #定义存储训练的轮数的变量。这个歌变量不需要计算滑动平均值，所以这里指定的这个变量为
    #不可训练的变量(trainable = False).在使用TensorFlow训练神经网络时
    #一般会代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0,trainable=False)

    #给给定的滑动平均衰减率和训练轮数的变量，初试化的滑动平均类。
    #给定训练的轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #在所有代表神经网络参数的变量上使用华东平均，其他的辅助（biru global_step）就不需要了
    #tf.trainable_variables返回的就是图上的集合
    #graphkeys.trainable_variable中的元素，这个集合的元素就是所有的指定的trainable = False参数

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用了平滑平均之后的向前传播的结果。
    #取值，而是会维护一个音质变量来记录其滑动平均值，所以当需要使用这个滑动平均值的时候
    #需要明确调用average函数
    average_y  = inferance(x,variable_averages,weights1,biases1,weights2,biases2)

    #计算交叉熵作为刻画预测值和真实值之间的差距的损失函数，这里使用了tensorflow中提供的
    # sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。当分类问题只有一个正确是答案
    #时，可以使用这个歌函数来加速交叉熵的计算。MNIST问题的图片上只包含0~9中的数字，所以使用这个函数来计算交叉熵
    #当分类问题只有一个正确答案是，可以使用这个函数来计算交叉损失函数，这个函数
    #标准答案是一个长度为10的一个数组，而该函数需要提供的是一个正确的答案，所以需要使用tf.argmax函数来得到正确答案
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    #计算当前的batch中的所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数。
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失。一般只计算神经网络边上的权重的正则损失，而不使偏置项
    regularization = regularizer(weights1)+regularizer(weights2)
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,   #基础的学习率，随着迭代的进行，更新变量是使用的
                              #学习率在这个歌基础上递减
        global_step,          #当前迭代的轮数
        mnist.train.num_example/BATCH_SIZE,#过完所有的训练的数据需要的迭代次数
        LEARNING_RATE_DECAY    #学习率的衰减率
    )
    #使用tf.train.GradientDescentOptimalizer优化算法来优化损失函数。注意这里的损失函数
    #包含了交叉熵损失和L2正则化损失。
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

    #在训练的神经网络模型的时候，每一过数据即需要通过反向传播来更新神经网络中的模型的参数
    #又要更新每一个参数的滑动平均值，为了一次完成多个操作，TensorFlow提供了
    #tf.control_dependencies和tf.group两种机制，下面两行程序和
    #train_op = tf.group(train_step,Variables_averages_op)是等价的
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    #检验使用的滑动平均模型神经网络前向传播结果是否正确。
    # tf.argmax(average_y,1)计算每一个样例的预测答案。其中average_y是一个batch_size*10的二维数组
    #表示一个样例的前向出阿伯结果，tf.argmax的第二个参数"1"表示选取最大值的操作仅仅在第一个维度中进行，
    # 也就是说，只在每一行选取最大值对应的下标。于是得到的结果是一个长度为batch
    #的一维数组，这个一维数组中的值表示了每一个样例对应的数字识别结果
    #tf.equal判断两个张量的每一维是否相等，如果相等就返回True,否则就返回false
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #这个运算首先讲一个bool型的数值转换为实数型，然后计算平均值，这个平均值，就是模型子这个一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
