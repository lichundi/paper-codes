# Hyperspectral Image Classification With Multiattention Fusion Network
- Manuscript received October 7, 2020; revised December 31, 2020; accepted January 13, 2021.
## 摘要
- 高光谱图像（HSI）有数百个连续的波段，其中包含大量的冗余信息。此外，高光谱立方体的空间分片往往包含一些与中心像素类别不同的像素，通常称为干涉像素。这种干扰像素的存在对提取更多的鉴别信息有负面影响。
- 本文提出了一种用于HSI分类的多注意融合网络（MAFN）。与现有的方法相比，MAFN分别采用了带注意模块（BAM）和空间注意模块（SAM）来减轻冗余带和空间干扰像素的影响。这样，MAFN结合多注意和多级融合机制，实现了特征重用，从不同层次获取互补信息，从而提取出更具代表性的特征
- source code is available at https://github.com/Li-ZK/MAFN-2021.
- BAM MAFN采用BAM来解决频带冗余问题，提高了系统的整体性能。
- SAM用于提取与中心像素类别一致的区域特征，从而提取更具辨别的特征。
- 本文还采用了多级特征融合的方法来提取低、中、高层的不同特征。最后，将多注意和多级特征融合机制相结合，实现特征重用，从不同层次获取补充信息，从而提取出更具代表性的特征。
