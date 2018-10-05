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

    :param file_dir:
    :return:
    '''
    images = []
    temp = []
    for root,sub_floders,files in os.walk(file_dir):#os.walk(file_dir)解析了一串的文件的路径
        #图片的目录
        for name in files:
            images.append(os.path.join(root,name))
            #获取10个子目录的文件夹的名字
            for name in sub_floders:
                temp.append(os.path.join(root,name))
                pass
            print(files)
            pass
    pass
    #根据文件夹的名字来指定10个标签,,
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('\\')[-1] #拿到最后一个文件夹的名称,同文件夹的名称获得标记的数据labels
        if letter == 'cat':
            labels = np.append(labels,n_img*[0])#相当于temp = [temp,app]
        else:
            labels = np.append(labels,n_img*[1])
        pass
    #shuffle
    temp = np.array([images,labels])
    temp = temp.transpose()#
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]

    return image_list,label_list

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value = value))

def bytes_featrue(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = value))

def convert_to_tfrecord(image_list,labels_list,save_dir,name):
    '''
    :param image_list:获取图片的数据集
    :param labels_list:对应的标签
    :param save_dir: 存储路径
    :param name:
    :return:
    '''
    filename = os.path.join(save_dir,name+'.tfrecords')
    n_samples = len(labels_list)
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start ......')
    for i in np.arange(0,n_samples):
        try:
            image = io.imread(image_list[i])#image 的类型必须是array！
            image_raw = image.tostring()
            label = int(labels_list[i])##这里到底是list还是其他的什么
            example = tf.train.Example(featrues=tf.train.Features(feature={'label':int64_feature(label), 'image_raw':bytes_featrue(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('不能读取：',image[i])
    writer.close()
    print('\n Transform done!')
    pass

#训练时数据集的提取
def read_and_decode(tfrecords_file,batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])#数据产生
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    img_featrues = tf.parse_single_example(serialized_example,\
                                           features={
                                               'label':tf.FixedLenFeature([],tf.int64),
                                               'image_raw':tf.FixedLenFeature([],string),
                                           })
    image = tf.decode_raw(img_featrues['image_raw'],tf.uint8)
    image = tf.reshape(image,[227,227,3])
    label = tf.cast(img_featrues['label'],tf.int32)
    image_batch,label_batch = tf.train.suffle_batch([image,label],\
                                                    batch_size = batch_size,\
                                                    min_after_dequeue = 100,\
                                                    num_threads = 64,\
                                                    capcity = 200)
    return image_batch,tf.reshape(label_batch,[batch_size])#输入的batch是对读取的batch的尺寸进行设置，
                                                            # 如果大小不合适会影响模型的训练速度
    pass

#图片的四肢数据转化为TensorFlow专用格式
def get_batch(image_list,label_list,img_width,img_height, batch_size, capacity):
    '''
    :param image_list:图片列表
    :param label_list:标签列表
    :param img_width:
    :param img_height:
    :param batch_size:
    :param capacity:
    :return:
    '''
    image = tf.cast(image_list,tf.string)
    label = tf.cast(label_list,tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    image = tf.image.per_image_standardization(image)#将图片标准化
    image_batch,label_batch = tf.train.batch([image,label],\
                                             batch_size=batch_size,\
                                             num_threads=64,\
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
    :param inputs:
    :param is_training:
    :param is_conv_out:
    :param decay:
    :return:
    '''

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)
