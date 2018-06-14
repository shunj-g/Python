'''
  @author  gsj
  使用Numpy来将random.permutation()函数打乱数据集的所有的数据元素
  使用K-近邻算法分类 K-neighbor-Classfication
'''
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target

x_min,x_max = x[:,0].min()-.5,x[:,0].max() + .5
y_min,y_max = x[:,1].min()-.5,x[:,1].max() + .5

#mESH
#MESH
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'],N=3)#color
h = .05
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
knn = KNeighborsClassifier()
knn.fit(x,y)
Z = knn.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light,alpha=0.5)#具有网眼的坐标图

#plot
plt.scatter(x[:,0],x[:,1],c=y)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.show()