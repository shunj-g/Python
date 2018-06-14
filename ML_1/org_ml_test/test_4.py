import numpy as np
import pandas as pd

#a1 = np.ones((3,3))
#a2 = np.arange(0,12).reshape(3,4)
#A = np.arange(0,10).reshape(1,10)
#B = np.arange(0,10).reshape(10,1)
#a3 = np.linspace(0,10,5)
#a4 = np.array((3.3,4.5,1.2,5.7,0.3))
#a4.sum()
#print(a4.sum())
#print(a4.min())
#print(a2)
#A = np.arange(16).reshape(4,4)
#b = np.arange(4)

#m = np.arange(6).reshape(3,2,1)
#print(m)
#n = np.arange(6).reshape(3,1,2)
#print(n)
#print(m+n)  #进行矩阵相加时，要讲两个数组进行广播
################
################
#def foo(x):
#    return x/2
#a5 = np.apply_along_axis(foo,axis=1,arr=a2)#np.mean
#[A1,A2,A3] = np.split(a2,[1,3],axis = 1)

#print(A1,A2,A3)
#print(A)
#print(B)
#C = B+A
#print(A)
#print(b)
#print(A+b)
#structured = np.array([(1,'First',0.5,1+2j),(2,'Second',1.3,2-2j),
                      # (3,'Third',0.8,1+3j)],dtype=('i2,a6,f4,c8'))
#print(structured['f1'])

#arr = np.array([1,2 ,3,4])
#s3 = pd.Series(arr)
#print(s3)
#s = pd.Series([12,-4,7,9],index=['a','b','c','d'])
#默认的标签是1,2,3,4
#加入的标签是可以

#print(s.values)
#定义DataFrame对象(data)
#DataFram是神来之笔---->堪比结构化的数据库
data = {'color': ['blue','green','yellow','red','write'],
        'object':['ball','pen','pencil','paper','mug'],
        'price': [1.2,1.0,0.6,0.9,1.7]}

frame = pd.DataFrame(data)
#print(frame)

frame = pd.DataFrame(np.arange(16).reshape((4,4)),
                     index=['red','blue','yellow','white'],#行索引
                     columns=['ball','pen','pencil','peper'])#列索引

#print(frame)
#frame2 = pd.DataFrame(data,columns=['object','price'])
#print(frame2[frame2.isin([1.0,'pen'])])
#def f(x):
#    return pd.Series([x.min(),x.max()],index=['min','max'])

#func = lambda x: x.max() - x.min()#lambda表达式
ser = pd.Series([5,0,3,8,4],#数据内容
                index=['red','blue','yellow','white','green'])#数据column
#print(ser.rank())
#对于索引的排序：使用sort_index()
#def f(x):
#     return pd.Series([x.min(),x.max()],index=['min','max'])
#print(frame.sort_index(ascending=True))#一般设置为False就是表明从大到小排列
#frme表明是二维表
#print(frame.sort_index(axis=1))

#Series的对象的排序，一般使用rank()，没有order()
#print(ser.order())

seq2 = pd.Series([3,4,3,4,5,4,3,2],['2006','2007','2008','2009','2010','2011','2012','2013'])
seq = pd.Series([1,2,3,4,4,3,2,1],['2006','2007','2008','2009','2010','2011','2012','2013'])
#print(seq.corr(seq2))#相关性
#print(seq.cov(seq2))#协方差

frame2 = pd.DataFrame([[1,4,3,6],[4,5,6,1],[3,3,1,5],[4,1,6,4]],
                     index=['red','blue','yellow','white'],#行索引
                     columns=['ball','pen','pencil','peper'])#列索引
#print(frame2.corr())
#print(frame2.corrwith(frame))

ser1 = pd.Series([5,0,np.nan,8,4],#数据内容
                index=['red','blue','yellow','white','green'])#数据column

import pymongo as mongo
import pymysql as mysql
import matplotlib.pyplot as plt#导入包
#print(ser1[ser1.notnull()])###
#print(ser1.dropna())       #how='all'

Xmin=0
Xmax=5
Ymin=0
Ymax=20
plt.axis(Xmin=Xmin,Xmax=Xmax,Ymin=Ymin,Ymax=Ymax)
plt.title('My first plot',fontsize=20,fontname='Times New Roman')
plt.xlabel('counting',color='gray')
plt.ylabel('square values',color='gray')
plt.text(1,1.5,'First')
plt.text(2,4.5,'Second')
plt.text(3,9.5,'Third')
plt.text(4,16.5,'Fourth')
plt.text(1.1,12,r'$y = x^2$',fontsize=20,bbox={'facecolor':'yellow','alpha':0.5})
plt.grid(True)
x = np.arange(-2*np.pi,2*np.pi,0.01)
y = np.sin(3*x)/x
plt.plot(x,y,'r')
#plt.plot([1,2,3,4],[1,2,6,16],'r-o')
plt.legend(['First series','Second series','Third series'],loc = 2)
plt.show()

pop = np.random.randint(0,100,100)
n,bins,patches = plt.hist(pop,bins=20)
plt.show()

index = np.arange(5)
values = [5,7,3,4,6]
std=[0.8,1,0.4,0.9,1.3]
plt.title('A Bar Chart')
##bar是垂直的 barh是平行的
plt.barh(index,values,yerr=std,error_kw={'ecolor':'0.1',
                                        'capsize':6},alpha=0.7,label='First')
plt.yticks(index+0.4,['A','B','C','D','E'])
plt.legend(loc=2)
plt.show()
#user = 'root'
#pwd = ''
#host = '127.0.0.1'
#db = 'taotao'
#mydb = mysql.connect(user=user,password=pwd,host=host,database=db)
#print(pd.__version__)