#数据集处理
import os
import cv2
def rebuild(dif):
    '''
    在这里导入的是图片集的根目录，os对于数据集所在的文件进行读取，
    之后在一个for循环下重新构建了所有的数据的路径，图片重新构建后
    写入到给定的位置，
    :param dif:
    :return:
    '''
    for root,dirs,files in os.walk(os):
        for file in files:
            filepath = os.path.join(root,file)
            try:
                image = cv2.imread(filepath)
                dim = (227,227)#这个是可以变化的
                resized = cv2.resize(image,dim)#变化数据集维度
                #path = "E:\\cat_and_dog\\dog_r\\"+file
                path = ".\\dataset_kaggledogvscat\\train"+file
                #E:\计算机视觉\Python\AlexNet\dataset_kaggledogvscat\train
                cv2.imwrite(path,resized)
            except:#当出现坏的图片数据的时候，会抛出相关的异常，并且删除图片
                print(filepath)
                os.remove(filepath)
        pass
        cv2.waitKey(0) #退出
    pass
