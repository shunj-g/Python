'''
使用SVM库来进行手写体识别——样例
'''
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
labeled_images = pd.read_csv('mnist/train.csv')
images = labeled_images.iloc[0:42000,1:]
labels = labeled_images.iloc[0:42000,:1]


train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img)
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

test_images = pd.DataFrame(test_images)
train_images = pd.DataFrame(train_images)
#这里之所以报错就是因为没有使用pandas进行数据格式是转化
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].as_matrix().reshape((28,28))
print(img)
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])

#整个支持向量机的模型的数据的训练
plt.hist(train_images.iloc[i])
clf = svm.SVC()#分类器的类别    这里的clf是分类的
clf.fit(train_images, train_labels.values.ravel()) #训练支持向量机模型
clf.score(test_images,test_labels)#返回给定测试数据和标签的平均精度。

test_data=pd.read_csv('mnist/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:28000])

'''
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
'''
test_data=pd.read_csv('mnist/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:28000])
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
