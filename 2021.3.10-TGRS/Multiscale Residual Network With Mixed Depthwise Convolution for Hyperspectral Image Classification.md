# Multiscale Residual Network With Mixed Depthwise Convolution for Hyperspectral Image Classification.md
- Manuscript received February 21, 2020; revised June 2, 2020; accepted July 5, 2020.
## 摘要
- 卷积神经网络（CNN）在现代遥感图像处理中的应用越来越广泛，在高光谱图像（HSI）分类中表现出突出的能力。然而，现有的基于CNN的HSI分类方法大多只考虑单尺度特征提取，忽略了一些重要的精细信息，不能保证获取最优的空间特征。此外，许多最新的方法需要调整大量的网络参数，这将导致很高的计算成本。
- 针对上述两个问题，提出了一种新的多尺度残差网络（MSRN）用于HSI分类。具体地说，提出的MSRN引入了深度可分离卷积（DSC），并用混合深度卷积（MDConv）代替DSC中的普通深度卷积，在一次深度卷积操作中混合了多个内核大小。混合深度卷积DSC（MDSConv）不仅可以从每个特征图中探索不同尺度的特征，而且可以大大减少网络中的可学习参数。另外，用MDSConv层代替普通残差块中的卷积层，设计了多尺度残差块。MRB用作拟定MSRN的主要单元。此外，为了进一步增强特征表示能力，该网络在级联的两个mrb上增加了一个高级快捷连接（HSC）来聚合低层特征和高层特征。