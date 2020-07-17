'''
@Author: your name
@Date: 2020-07-15 08:52:51
@LastEditTime: 2020-07-15 20:39:35
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \undefinedc:\Users\李春娣\Desktop\学校\实验室\论文\论文学习\2020.7.14&7.21\LDAM-DRW\losses.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F #包含 torch.nn 库中所有函数，同时包含大量 loss 和 activation function
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()
# FocalLoss 继承父类Module 将自定义操作封装成nn.Module类
class FocalLoss(nn.Module):
    #传入权重和γ值
    def __init__(self, weight=None, gamma=0.):
        #调用父类构造函数
        super(FocalLoss, self).__init__()
        assert gamma >= 0 # assert断言函数：如果表达式为假，触发异常；如果表达式为真，不执行任何操作。
        self.gamma = gamma
        self.weight = weight
    #实现forward函数，该函数为默认执行的函数，即计算过程，并将输出返回
    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0 # assert断言函数：如果表达式为假，触发异常；如果表达式为真，不执行任何操作。
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)#返回一个填充了标量值0的张量，其大小与之相同 x
        index.scatter_(1, target.data.view(-1, 1), 1)
        #scatter() 和 scatter_() 的作用是一样的，只不过 scatter() 不会直接修改原来的 Tensor，而 scatter_() 会
        #PyTorch 中，一般函数加下划线代表直接在原来的 Tensor 上修改
        #scatter(dim, index, src) 的参数有 3 个dim：沿着哪个维度进行索引 index：用来 scatter 的元素索引 src：用来 scatter 的源元素，可以是一个标量或一个张量
        index_float = index.type(torch.cuda.FloatTensor) 
        # Pytorch中的tensor又包括CPU上的数据类型和GPU上的数据类型，一般GPU上的Tensor是CPU上的Tensor加cuda()函数得到。
        #一般系统默认是torch.FloatTensor类型。例如data = torch.Tensor(2,3)是一个2*3的张量，类型为FloatTensor;data.cuda()就转换为GPU的张量类型，torch.cuda.FloatTensor类型。

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) #.matmul 矩阵相乘
        batch_m = batch_m.view((-1, 1)) #view()函数作用是将一个多行的Tensor,拼接成一行。
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x) #torch.where(condition, input, other)返回input或是other中满足condtion的元素。
        return F.cross_entropy(self.s*output, target, weight=self.weight)