raw文件到原子文件user, item, rating...
    各个数据集继承base_dataset.py，自定义自己的数据读取方式

原子文件到训练文件
    使用preprocess.py对原子文件进行转化，如自定义分割比例，remapid，筛选用户交互数目等