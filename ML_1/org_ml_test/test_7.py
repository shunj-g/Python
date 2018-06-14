import numpy as np
from sklearn import datasets
'''
  @author  gsj
  使用Numpy来将random.permutation()函数打乱数据集的所有的数据元素
  使用K-近邻算法分类 K-neighbor-Classfication
'''
np.random.seed(0)
iris = datasets.load_iris()
x = iris.data
y = iris.target
i = np.random.permutation(len(iris.data))#生成150的随机数
x_train = x[i[:-10]]
y_train = y[i[:-10]]
x_test = x[i[:-10]]
y_test = y[i[:-10]]

#导入K邻近包
from  sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',
                     metric_params=None,n_neighbors=5,p=2,weights='uniform')
ans = knn.predict(x_test)
print(ans)
print(y_test)