# A Fast Dense Spectral–Spatial Convolution Network Framework for Hyperspectral Images Classification #
## 摘要 ##
- FDSSC框架使用不同的卷积核大小分别提取光谱和空间特征，并使用“valid”的卷积方法来降低高维。为了提高速度和防止过度拟合，FDSSC框架使用了动态学习速率、参数校正线性单元、批处理规范化和dropout层。
## 1. Introduction ##
## 2. Proposed Framework ##
### 2.1. Extracting Spectral and Spatial Features Separately from HSI ###
- （1）V是在第i层的第j个特征cube在位置（x,y,z）的值
- Figure 1（a）2D卷积（b）3D卷积操作
### 2.2 Going Deeper with Densely-Connected Structures ###
#### 2.2.1. Densely-Connected Structure ####
- (2)传统卷积操作
- （3）densely-connected structure
- Figure 2. Example of a DenseNet with four composite layers (l = 4)
#### 2.2.2. Separately Learning Deeper Spectral and Spatial Features ####
- （4）第l层的输入 D1（.）卷积层
- （5）输入的特征映射的数量
-  Figure 3 密集光谱块的结构 在密集的谱块和空间块中，填充三维卷积层的方法是“相同的”，这是输出尺寸恒定（r×r×b）的原因。然而，在减少维度层时，用于填充三维卷积层的方法对于改变特征映射的大小是“有效的”。
- Figure 4 降维层的结构 通过两个具有“valid”填充和reshape的三维卷积层特征映射为s×s×1，减小了数据块的空间大小、通道数目和数据块的高维性
- （6）密集空间块中第lth层的卷积层的输出
-  Figure 5 密集空间块结构
### 2.3 Going Faster and Preventing Overfitting There ###
- 大量的训练参数，训练时间长和过拟合
- （7）PReLU 激活函数 在ReLU的基础上引入少量的参数ai是一个可学习的参数，确定斜率的负部分
- （8）PReLU在更新ai时采用动量法 µ是动量，lr是学习率。更新ai时，不应使用权重衰减，因为ai可能趋于零。ai=0.25作为初始值。虽然ReLU是一个有用的非线性函数，但它阻碍了反向传播，而PReLU使模型更快地收敛
- 在训练模型时，还使用了早期停止和动态学习率。早停是指在经过一定的epoch（如本文中的50个）之后，如果损失不再减少，训练过程将提前停止。这样既减少了训练时间，又防止了过拟合。我们采用了一个可变的学习速率，因为当结果接近一个最优值时，步长应该减小。在初始学习率较高的情况下，当经过一定数量的周期（如本文中的10个周期）后精度不提高时，学习率将减半。如果精度在一定的时间段后不再增加，学习率将再次降低一半，并将循环，直到小于设定的最小学习率。在本文中，最小学习率被设置为0，也就是说，学习率循环直到达到最大的epoch数。
- 采用dropout层防止过拟合
### 2.4 Fast Dense Spectral–Spatial Convolution Framework ###
#### 2.4.1 目标函数 ####
- （9）FDSSC框架的预测值,liy
