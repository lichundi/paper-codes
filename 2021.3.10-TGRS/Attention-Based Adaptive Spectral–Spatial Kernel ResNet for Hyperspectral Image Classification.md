# Attention-Based Adaptive Spectral–Spatial Kernel ResNet for Hyperspectral Image Classification
- Manuscript received November 7, 2020; accepted December 3, 2020
## 摘要
- 高光谱图像（HSI）提供了丰富的光谱-空间信息，具有成百上千个相邻的窄带。由于噪声和频带相关性的存在，信息谱-空间核特征的选择带来了挑战。这通常是通过使用卷积神经网络（CNNs）和具有固定大小的感受野（RF）来解决的。然而，当使用前向和后向传播来优化网络时，这些解决方案不能使神经元有效地调整射频大小和交叉信道依赖性。
- 本文提出了一种基于注意力的自适应谱空间核改进残差网络（A2S2K ResNet），该网络以端到端训练的方式捕获识别光谱空间特征。特别地，该网络学习选择性的三维卷积核，利用改进的三维重构块联合提取光谱-空间特征，并采用有效的特征重校准（EFR）机制提高分类性能。
