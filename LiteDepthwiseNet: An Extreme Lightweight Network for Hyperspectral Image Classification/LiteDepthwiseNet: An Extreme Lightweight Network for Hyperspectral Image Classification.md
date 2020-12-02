## 摘要 ## 
- 基于三维深度卷积，LiteDepthwiseNet可以将标准卷积分解为深度卷积和点态卷积，以最小的参数获得较高的分类性能。
- 去除了原始三维深度卷积中的ReLU层和Batch Normalization layer，显著改善了模型在小数据集上的过拟合现象。
- 采用focal loss作为损失函数，提高了模型对困难样本和不平衡数据的关注度，其训练性能明显优于交叉熵损失和平衡交叉熵损失
## INTRODUCTION ##
- 上述DL模型通常需要大量的训练样本和网络参数，而且计算量也较高
- HSI在分类中的标记成本和可用性都非常有限。此外，这些标记数据的类别分布是不均匀的。
- 轻量级的神经网络LiteDenseNet被提出用于HSI分类。与传统的3D-CNN结构不同，LiteDenseNet采用了三维双向密集结构，采用了群卷积，大大减少了参数的数目和计算成本。然而，群卷积切断了不同通道的连接，并可能导致精度的损失。
- 一个非常轻量级的HSI分类任务网络LiteDepthwiseNet优点：
- 1、用三维深度卷积代替群卷积。三维深卷积中的点态卷积可以连接所有的高光谱通道，相应的网络结构具有全通道接收场，更适合于HSI分类任务。基于这种新的网络结构，LiteDepthwiseNet只涉及很少的参数和触发器
- 2、引入焦点损失（FL）代替主流交叉熵损失（CEL）作为损失函数。它增加了对小样本和难分类样本的关注，有助于提高模型的最终性能。
- 3、去除了原始三维深卷积网络的中间激活层和归一化层，增强了模型的线性度，减少了HSI训练样本较少时的过拟合现象。
## RELATED WORK ##
### A. 2D 深度可分离卷积 ###
- Fig. 1 标准卷积
- Fig. 2深度可分离卷积，标准卷积分解为深度卷积和点态卷积两个过程，在一定条件下可以显著降低二维卷积的计算量和参数个数。
### B. 交叉熵损失函数 ###
- （1）标准交叉熵损失函数 p表示类别中样本的预测概率，y表示样本是否正确分类，y是一个等于0或1的指示变量。可以观察到，当y为1时，p越接近1，损失越小；如果y为0，p越接近0，则损失越小，这符合优化的方向。
- （2）我们引入分段函数F，简化表达式
- （3）交叉熵损失函数另一种形式
- （4）下平衡CEL（BCEL），为了处理类别不平衡的数据，一个在标准交叉熵损失函数中加入系数α∈[0，1]，以平衡不同类别样本的权重
### C. LiteDenseNet ###
- Fig. 3 The 3D two-way dense layer 在对输入数据进行1×1×1的卷积变换后，顶层采用两层3×3×3的卷积来捕获全局特征，得到尺寸为（9×9×97，12）的输出；底层采用3×3×3的卷积核捕捉局部特征，得到尺寸为（9×9×97，12）的输出。对双向密集层的输出进行级联运算，生成尺寸为（9×9×97，48）的三维立方体。
- Fig. 4: Group Convolution  为了减少参数的数目和计算量，LiteDenseNet使用了群卷积来代替常规卷积。 正常卷积层的信道大小与输入端相同，而群卷积将原始卷积核和输入端按信道维数分成3组。将输入信号分成h×w×l和通道cF/3三组。三组分别与cF/3通道进行正常卷积。因此，通道尺寸由h×w×l×cF减小到h×w×l×（cF/3）。这样，群卷积在保持输入和输出大小的同时，显著地减少了参数数目和计算开销。
## III. THE PROPOSED LITEDEPTHWISENET ##
### A. Modified 3D Depthwise Convonlution ###
- Fig. 5: Group convolution with 3 groups 群卷积永久性地切断了信道间的某些连接，从而在一定程度上降低了分类性能。
- Fig. 6: 3D depthwise convolution. Fig. 深度可分离卷积代替群卷积，其中点态卷积可以连接所有信道。
- Fig. 7: Modified 3D depthwise convolution.由于HSIs训练样本数量有限，往往会对非线性能力较强的模型产生过拟合。因此，我们提出了一种改进的三维深度卷积用于HSI分类。主要调整是去除了原三维深卷积网络的中间激活层和归一化层，增强了模型的线性度，减少了过拟合现象。同时，大大减少了参数的数目和计算量。
### B. Focal Loss ###
- (5)由于大多数HSI具有不平衡的数据类别[5]，使用标准CEL函数进行训练将导致小样本类别的分类精度较低。BCEL函数通常用于处理这种情况。然而，BCEL函数也可能集中在易于分类的样本上。为了提高分类性能，该模型应兼顾小样本和难分类样本。
### C. The Framework of the LiteDepthwiseNet ###
- Fig. 8: The LiteDepthwiseNet architecture 
- 具体来说，在我们的网络框架中，批处理规范化
-（BN）和ReLU激活（ReLU）在每个点卷积和普通卷积之后添加。为了简洁起见，在下面的描述中将不指定它。当一个输入，一个尺寸为（9×9×200，1）的三维立方体输入到尺寸为（1×1×7，24）的3D-CNN层中，我们得到一个尺寸为（9×9×97，24）的输出。接下来，我们将输出分别放入两个分支中。右分支（分支1）对输入进行分组卷积，并将输入分成三组，每组是一个三维立方体，大小为（9×9×97，8）。然后，我们将它们放入一个尺寸为（1×1×1，16）的3D-CNN层中。到目前为止，我们仍然保留了LiteDenseNet的原始设计，因为将第一层和第二层修改为3D深度卷积会增加参数的数量。然后，将三个三维cnn的输出结果按通道尺寸进行叠加，得到尺寸为（9×9×97，48）的输出。后续模块均由三维深卷积构成，按通道尺寸将（9×9×97，48）分为48个独立的3D块，并使用48个（3×3×3，1）尺寸的3D-CNN滤波器进行一对一的卷积运算，输出尺寸为（9×9×97，48）。我们将其输入到尺寸为（1×1×1，12）的3D-CNN中进行逐点线性组合，弥补了群卷积的不足。输入通道数和输出通道数都要继续卷积。至此，右分支的设计就完成了。左分支（分支2）将通过群卷积获得的输入输入馈入3D深度卷积。最后，我们将三个通道的输出，即未进入任何支路的输入（9×9×97，24）、右支路的输出（9×9×97，12）和左支路的输出（9×9×97，12）相联系，得到大小为（9×9×97，48）的输出，并将其送入最后通过一个全局平均池层和一个完全连通层得到最终结果。
### D.计算成本和参数数量对比 ###
- TABLE II:
- Fig. 9 Parameters of each algorithm
- Fig. 10: FLOPs of each algorithm 
## IV. 实验结果 ##
### A. Data Description ###
- TABLE III:IP数据集 但对于IP数据，训练率太低，这将导致某些类别的训练样本缺失。为了确保每个类别中至少有5个样本，我们使用5%的IP数据进行训练。
- TABLE IV：UP数据集 0.5%的UP数据进行训练
- TABLE V: PC数据集 0.1%的PC数据用于训练
### B. Classification Maps and Categorized Results ###
- TABLE VI  Fig. 11 IP
- Table VII  Fig. 12 UP
- Table VIII Fig. 13 PC 
### Investigation of the Effects of Different Loss Functions ###
- Table IX 三个数据集上不同损失函数下所提出算法的实验结果
### D. Investigation of the γ of Focal Loss ###
- Fig. 14: The OA, AA, and Kappa of LiteDepthwiseNet with different values of γ on the IP dataset.
- Fig. 15: The OA, AA, and Kappa of LiteDepthwiseNet with different values of γ on the UP dataset.
- Fig. 16: The OA, AA, and Kappa of LiteDepthwiseNet with different values of γ on the PC dataset.
- γ值对AA的影响尤为显著，因为AA对小样本类别（通常是难分类样本）的准确性特别敏感，调整γ可以显著提高对难分类样本的关注度。
