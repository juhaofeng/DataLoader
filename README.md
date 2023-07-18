# DataLoader
- 采用torch的DataLoader加载自己的数据集 数据集包括图片集、验证集、图片标签和验证集标签
- 代码中定义FlowerDataset类继承Dataset ，其必须重写 __init__，__getitem__，__len__函数
- __init__ ： 构造函数 传入目录地址  root_dir ： img集地址  ann_file： 标签地址
- __getitem__ ： 可以根据id值获得所对应的图片和对应的标签
-  __len__ ： 获得图片集的长度


### 仅上传部分训练集图片 如果您需要更多的训练集，请联系我邮箱 j186258@163.com
