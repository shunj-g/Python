from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
iris = datasets.load_iris()

#以花瓣作为：2,：3
#以花蕊作为：0:1
x = iris.data[:,1]#x-axis - sepal length
y = iris.data[:,2]#Y-axis - sepal length

#有(3,2)=共有3种情况
spacies = iris.target #Spacies
x_min,x_max = x.min() - 0.5,x.max() + 0.5
y_min,y_max = y.min() - 0.5,y.max() + 0.5

#SCATTERPLOT
#plt.figure()
#plt.title('Iris DataSet - Classfication By Sepal Size')
#plt.scatter(x,y,c = spacies)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
#plt.xlim(x_min,x_max)
#plt.ylim(y_min,y_max)
#plt.xticks()
#plt.yticks()
#plt.show()



print('======================')
print(iris.data)
print('======================')
print(iris.target)
print('======================')
print(iris.target_names)

#将四维数据转化为三维数据
#主成分分析法-----》奇异矩阵进行分解操作
#进行降维操作
x_reduced = PCA(n_components=3).fit_transform(iris.data)#降为3维
print(x_reduced)
fig = plt.figure()#获取图像
fig.show()
ax = Axes3D(fig)
ax.set_title('Iris DataSet by PCA',size=14)
#降维为3维后进行散点图
ax.scatter(x_reduced[:,0],x_reduced[:,1],x_reduced[:,2],c = spacies)
ax.set_xlabel('First eigenvector')
ax.set_ylabel('Second eigenvector')
ax.set_zlabel('Third eigenvector')
ax.w_xaxis.set_ticklabels(())
ax.w_yaxis.set_ticklabels(())
ax.w_zaxis.set_ticklabels(())

plt.show()






