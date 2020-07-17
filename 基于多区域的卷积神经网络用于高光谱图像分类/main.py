'''
@Author: your name
@Date: 2018-05-09 11:06:14
@LastEditTime: 2020-07-07 16:41:31
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \undefinedc:\Users\李春娣\Desktop\学校\实验室\论文\论文学习\2020.6.24\20200514235605008619\main.py
'''
import os #导入操作系统接口模块

train_KEY = ['data', 'XR', 'XL', 'XU', 'XB', 'XC']
test_KEY = ['hsi', 'XR', 'XL', 'XU', 'XB', 'XC']
ksize1 = [11, 7, 7, 11, 11, 3]
ksize2 = [11, 11, 11, 7, 7, 3]
num_result = 1
for str_train_key, str_test_key, ksize_1, ksize_2 in zip(train_KEY, test_KEY, ksize1, ksize2): # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    os.system('python HSI_multi_SPA.py --mode 0 --train_KEY {} --test_KEY {} --ksize1 {} --ksize2 {}'
              .format(str_train_key, str_test_key, ksize_1, ksize_2))  # os.system连续执行多条语句
    os.system('python HSI_multi_SPA.py --mode 1 --train_KEY {} --test_KEY {} --ksize1 {} --ksize2 {} > result{}'
              .format(str_train_key, str_test_key, ksize_1, ksize_2, num_result))# format 格式化字符串的函数，str.format()，它增强了字符串格式化的功能。
    num_result += 1


full_net_ = [False]
right_net_ = [True]
left_net_ = [False]
up_net_ = [True]
bottom_net_ = [True]
center_net_ = [True]

num_result = 11
for full_net, right_net, left_net, up_net, bottom_net, center_net in zip(full_net_, right_net_, left_net_, up_net_, bottom_net_, center_net_):
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    print num_result
    os.system('python HSI_multiSPA_union.py --mode 0 --full_net_ {} --right_net_ {} --left_net_ {} --up_net_ {} --bottom_net_ {} --center_net_ {}'
              .format(full_net, right_net, left_net, up_net, bottom_net, center_net))  # format 格式化字符串的函数，str.format()，它增强了字符串格式化的功能。
    os.system('python HSI_multiSPA_union.py --mode 1 --full_net_ {} --right_net_ {} --left_net_ {} --up_net_ {} --bottom_net_ {} --center_net_ {} > result{}'
              .format(full_net, right_net, left_net, up_net, bottom_net, center_net, num_result))
    num_result += 1
