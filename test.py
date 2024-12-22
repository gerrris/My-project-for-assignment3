
import numpy as np 
# 引入numpy模块
import pandas as pd
import tensorflow as tf
import torch
# 定义预处理函数
def preprocess(x,y):
    # 对输入数据缩放
    x=tf.reshape(x,[-1,1,256,8])
    print(x.numpy())
    print(x.shape)
    y=tf.cast(y,dtype=tf.int32)
    y=y+30
    y=tf.one_hot(y,depth=61)

    return x,y
filename='D:/Myfile/课题组工作/MIMO_DOA/my_work/Dataset/all_data_snaps_256.csv'#csv数据为复数
num_rows = 1  # 这里设置你想要读取的行数
# 使用pandas读取CSV文件
df = pd.read_csv(filename, skiprows=79999, nrows=num_rows, header=None)
# 将选定行转换为Numpy数组
array = df.to_numpy()
mapping = np.vectorize(lambda t:complex(t.replace('i','j')))
all_data= mapping(array)
all_data = np.real(all_data)# 获得全部的实部数据

# 再进行第2步：分别读取目标角度和回波矩阵
x_data = [] # 回波矩阵
y_data = [] # 目标角度
for i in range(all_data.shape[0]):
    featrure = all_data[i,1:]
    target = all_data[i,0]
    x_data.append(list(featrure))
    y_data.append(target)

x_data = np.array(x_data)
y_data = np.array(y_data)

x_data = torch.tensor(x_data)
y_data = torch.tensor(y_data)

[x,y]=preprocess(x_data,y_data)




 
