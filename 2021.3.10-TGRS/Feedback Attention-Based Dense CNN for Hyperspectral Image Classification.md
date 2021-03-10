# Feedback Attention-Based Dense CNN for Hyperspectral Image Classification
- Manuscript received August 10, 2020; revised January 12, 2021; accepted February 6, 2021
## 摘要
- 近年来，基于卷积神经网络（CNN）的高光谱图像分类（HSIC）方法不断发展。然而，高复杂度、信息冗余和低效率的描述仍然是当前HSIC网络的主要障碍。
- 为了解决上述问题，我们提出了一个空间谱密集的CNN框架，并在本文中提出了一种反馈注意机制FADCNN。该结构将光谱空间特征以紧凑的连接方式组合在一起，通过两个独立的密集CNN网络独立地提取足够的信息。具体地说，首次提出了反馈注意模块，利用稠密模型高层的语义知识来增强注意图，并考虑多尺度空间信息来增强空间注意模块。为了进一步提高特征表示的计算效率和识别率，设计了波段注意模块，强调了参与分类训练的波段的权重。此外，为了在特征挖掘网络中更好地细化空间光谱特征，对空间光谱特征进行了深入的集成和挖掘。
