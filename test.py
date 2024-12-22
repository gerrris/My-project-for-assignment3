import pandas as pd

# 假设我们要读取前1000行数据
file_path = 'D:/mydata.csv'#csv数据为复数
num_rows = 1000  # 这里设置你想要读取的行数

# 使用pandas读取CSV文件
df = pd.read_csv(file_path)

# 使用iloc选择前num_rows行
selected_rows = df.iloc[:num_rows]

# 将选定行转换为Numpy数组
array = selected_rows.to_numpy()

# 输出结果，查看数组
print(array.shape[0],array.shape[1])
print(type(array))




 
