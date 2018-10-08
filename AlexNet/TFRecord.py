#图片数据集转换为Tensorflow专用格式
'''
图片和文件夹的相关的数据处理，使用的是tensorflow的数据转化的完成的数据
'''
import os
import io
import string
import numpy as np
import tensorflow as tf

def get_file(file_dir):
    '''
    1.读取数据集文件的位置，
    2.根据文件夹名称的不同将处于不同的文件夹中的图片标签设为0或者1，如果有更多的分类的话可以根据这个格式来设置更多的标签类
    3.使用创建的数组对所读取的文件位置和标签进行保存，使用numpy对数组的调整核重构后对应的文件位置火热文件标签的矩阵
    :param file_dir:
    :return :
    '''
    images = []#图片数据集
    temp = []#缓存区
    for root,sub_floders,files in os.walk(file_dir):#os.walk(file_dir)解析了一串的文件的路径
        # root 根目录,sub_floders 子目录文件夹,files 文件
        #图片的目录
        for name in files:#所有的文件
            images.append(os.path.join(root,name))#连接两个(或更多)路径
            #获取10个子目录的文件夹的名字
        for name in sub_floders:#
            temp.append(os.path.join(root,name))#root
            pass
            #print(files)##打印相关的文件数据集
            pass
    pass
    #temp
    #根据文件夹的名字来指定10个标签,,
    labels = []
    for one_folder in temp: #temp
        n_img = len(os.listdir(one_folder)) #
        letter = one_folder.split('\\')[-1] #拿到最后一个文件夹的名称,同文件夹的名称获得标记的数据labels
        if letter == 'cat':#
            labels = np.append(labels,n_img*[0])#相当于temp = [temp,app]，平移矩阵元素#
        else:#
            labels = np.append(labels,n_img*[1])#相当于数据的
        pass #
    #shuffle 将序列的所有元素随机排序。
    temp = np.array([images,labels])#数据
    temp = temp.transpose()#得到的数据
    np.random.shuffle(temp)#将其随机打散

    image_list = list(temp[:,0])#特征数据集
    label_list = list(temp[:,1])#标签数据集
    label_list = [int(float(i)) for i in label_list]#这个是python特有的一种很好的结构操作
    #返回相应的数据集
    return image_list,label_list

def int64_feature(value):#特征值转换为64为的数据
    return tf.train.Feature(int64_list=tf.train.Int64List(value = value))#

def bytes_featrue(value):#特征值的数据转换为字节类型的数据
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = value))

def convert_to_tfrecord(image_list,labels_list,save_dir,name):
    '''
    一个tf.train.Example包含着若干的数据特征(Features),而Features中又包含着Feature字典,更进一步说明，任何一步的细节
    说明，任何一个Feature中又包含着FloatList,或者ByteList,或者Int64List,这三种数据格式之一。TFReacord就是通过一个包
    含着二进制的文件的数据文件，将数据和标签进行保存以便于Tensorflow
    :param image_list:获取图片的数据集
    :param labels_list:对应的标签
    :param save_dir: 存储路径
    :param name:
    :return :
    '''
    filename = os.path.join(save_dir,name+'.tfrecords')##得到数据集保存的位置
    n_samples = len(labels_list)#得到数据积集的长度
    writer = tf.python_io.TFRecordWriter(filename)#获取得到数据集的保存的位置
    print('\n Transform start ......')
    for i in np.arange(0,n_samples):##
        try:
            image = io.imread(image_list[i])#image 的类型必须是array！  获取图像数据
            image_raw = image.tostring()#将其转换为string类型,为什么要转化为string类型
            label = int(labels_list[i])##这里将标签数据转换为整数类型
            ##设置统一的格式输入
            example = tf.train.Example(featrues=tf.train.Features(feature={'label':int64_feature(label),
                                                                           'image_raw':bytes_featrue(image_raw)}))
            writer.write(example.SerializeToString())#将其转换为string的序列流
        except IOError as e:
            print('do not read：',image[i])
    writer.close()
    print('\n Transform done!')
    pass

#训练时数据集的提取
def read_and_decode(tfrecords_file,batch_size):
    '''
    训练时的数据集的提取
    1.定义一个数据的队列，将文件推入到队列中
    2.队列根据文件名读取数据，Decoder将读出的数据解码（readerhe writer是不一样的，reader符号化的，
       reader是要在session中run()函数才能读取的，而writer是可以的）
    3.如果要讲数据打印或者进一步执行，那么就要使用专门的函数tf.train.shuffle_batch来执行。
      shuffle_batch()函数用于从TFRecord 中读取数据。保证每次读取数据和标签的内容都是同步的不会造成不匹配的现象；
    :param tfrecords_file: 输入的frecords文件。
    :param batch_size:批处理块的大小
    :return:
    '''
    filename_queue = tf.train.string_input_producer([tfrecords_file])##数据产生队列
                                                                     #输入管道队列的输出字符串(如文件名)。
                                                                     # 注意:如果' num_epochs '不是' None '，
                                                                     # 这个函数将创建本地计数器“times”。
                                                                     # 使用' local_variables_initializer() '来初始化本地变量。
    reader = tf.TFRecordReader()#创建一个reader
    _,serialized_example = reader.read(filename_queue)##返回阅读器生成的下一个记录(键，值)对。
                                                       # 如果有必要，是否会将工作单元从队列中取出(例如，什么时候阅读器需要从一个新文件开始读取，
                                                       # 因为它已经有了完成前一个文件)。
    img_featrues = tf.parse_single_example(serialized_example,#
                                           features={
                                               'label':tf.FixedLenFeature([],tf.int64),  #
                                               'image_raw':tf.FixedLenFeature([],string),#
                                           })
    image = tf.decode_raw(img_featrues['image_raw'],tf.uint8)#转换为训练数据集
    image = tf.reshape(image,[227,227,3])#weight和width,和channels
    label = tf.cast(img_featrues['label'],tf.int32)#将张量转换为一种新类型（将label的数据类型转换为32位整型数据）
    image_batch,label_batch = tf.train.suffle_batch([image,label],
                                                    batch_size = batch_size,#每次弹出的元素的大小
                                                    min_after_dequeue = 100,#指出队列操作之后还可以提供随机
                                                                            # 采样出批量的数据的样本池的大小
                                                    num_threads = 64,#所用的线程的数目
                                                    capcity = 200)##队列能容纳的最大的元素 capcity > min_after_meam
                                                    ##一般推荐：min_after_mean+(num_threads+a small safty margin)*batch_size
    #batch进行读取的时候，Tensorflow构建一个队列queues和QueueRunners。
    #suffle_batch的作用就是构建一个读取数据的队列，不断的把单个元素送入到队列中，为了保证队列不陷入停滞状态，
    # 从而专门启动一个QueueRunners线程来完成，当队列中个的个数达到batch_size和min_after_dequeue的总和之后，
    # 队列会随机将barch_size个元素弹出，
    ##suffle_batch的作用将解码完毕的样本加入到一个队列中，按需弹出batch_size大小的样本。
    #这里要特别注意
    return image_batch,tf.reshape(label_batch,[batch_size])#输入的batch是对读取的batch的尺寸进行设置，
                                                            # 如果大小不合适会影响模型的训练速度
    pass

#图片的四肢数据转化为TensorFlow专用格式
def get_batch(image_list,label_list,img_width,img_height, batch_size, capacity):
    '''
    在工程上，除了将数据集转化为专用的数据格式外，还有一种常用的方法。
    将读取的数据集格式转化为专用个的格式，每次直接读取其中生成的batch值
    注：这里根据不同的硬件来配置这6个参数。
    :param image_list:图片列表
    :param label_list:标签列表
    :param img_width:图像的宽度
    :param img_height:图像的高度
    :param batch_size: 批处理模块的大小
    :param capacity :
    :return image_batch,label_batch:
    '''
    image = tf.cast(image_list,tf.string)   #数据集的类型转换
    label = tf.cast(label_list,tf.int32)    #标签集的数据类型的转换

    input_queue = tf.train.slice_input_producer([image,label])#生成“tensor_list”中每个“张量”的切片。
                                                              # 使用队列实现——队列的“QueueRunner”
                                                              # 添加到当前的“Graph”的QUEUE_RUNNER集合中。
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])            #数据的大小没有完全得到的全部的数据
    image = tf.image.decode_jpeg(image_contents,channels=3)  #读取图片jpeg

    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)#作物和/或垫一幅图像的目标宽度和高度。
                                                                              # 将图像大小调整为目标宽度和高度裁剪或用0均匀填充图像。
    image = tf.image.per_image_standardization(image)#将图片标准化
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size=batch_size,
                                             num_threads=64,
                                             capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
    pass

#标签重构和模型存储
def onehot(labels):
    '''
    one-hot编码
    :param labels:
    :return:
    '''
    n_sample = len(labels)
    n_class = max(labels)+1
    onehot_labels = np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels] = 1
    return onehot_labels

#数据的大小是
def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):
    '''
    一般而言，Batch_normalization 用在矩阵计算之前，因为卷积网络经管卷积后得到了一系列的特征图，
    在卷积网络中每个特征图可以看作一个特征处理，对于每个卷积后的特征图都只有一对可以学的参数，
    同时求取所有的样本的特征图所有的神经元的均值，方差，然后对这一个特征图进行归一化。
    :param inputs: 输入的数据集
    :param is_training: 是否输入的是训练集
    :param is_conv_out: 是否卷积
    :param decay: 卷机值
    :return:
    '''

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))#
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))#
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)  ##滑动均值
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)   ##滑动方差

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,#输入的数据文件
                                             batch_mean, #批量数据均值
                                             batch_var, #批量数据方差
                                             beta, #待训练的参数
                                             scale, #待训练的参数
                                             0.001) #方差编译系数
            ##B_N正则化的数据
    else:
        return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, 0.001)
