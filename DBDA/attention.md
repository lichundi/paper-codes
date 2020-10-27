# 摘要 #
- 序列转导模型主要是基于复杂的递归或卷积神经网络，包括一个编码器和一个解码器。本文提出一个新的简单的网络结构Transformer完全基于注意力机制，不需要递归和卷积。
- 在两个机器翻译任务上的实验表明，该模型的结果较好，并行性更高，所需的训练时间大大减少。在WMT 2014英语-德语翻译任务中达到了28.4 BLEU，比现有的最佳结果提高了超过2个BLEU。在8个GPU上训练3.5天，建立一个最先进的模型41.8BLEU。
# 引言 #
- 
# 模型结构 #
- 绝大部分的序列处理模型都采用Encoder-Decoder结构，其中Encoder将输入序列（x1,x2,…,xn），然后Decoder生成一个输出序列（y1,y2,…,yn）,每个时刻输出一个结果。从框架中我们可以知道Transformer模型延续了这个模型。
- Transformer其实这就是一个Seq2Seq模型（ Seq2Seq模型是输出的长度不确定时采用的模型，这种情况一般是在机器翻译的任务中出现，将一句中文翻译成英文，那么这句英文的长度有可能会比中文短，也有可能会比中文长，所以输出的长度就不确定了。），左边一个encoder把输入读进去，右边一个decoder得到输出：
- Transformer=Transformer Encoder + Transformer Decoder。
## 3.1 编码器和解码器堆栈 ##
- Encoder：Encoder有N=6层，每层包括两个Sub-Layers:第一个Sub-Layer是Multi-Headed Self-Attentionmechanism，用来计算输入的Self-Attention；第二个Sub-Layer是简单的全连接网络。在每个Sub-Layer我们都模拟了残差网络，每个Sub-Layer的输出都是：
- LayerNorm(x + Sublayer(x))，其中Sublayer(x) 是由子层本身实现的函数。为了方便这些残差连接，模型中的所有子层以及嵌入层产生的输出维度都为dmodel = 512。
- Decoder：Decoder也是N=6层，每层包括3个Sub-Layers：
- (1) 第一个是Masked multi-head Self-Attention，也是计算输入的Self-Attention，但是因为是生成过程，因此在时刻 i 的时候，大于 i 的时刻都没有结果，只有小于 i 的时刻有结果，因此需要做Mask。
- (2) 第二个Sub-Layer是全连接网络，与Encoder相同。
- (3) 第三个Sub-Layer是对Encoder的输入进行Attention计算。同时Decoder中的Self-Attention层需要进行修改，因为只能获取到当前时刻之前的输入，因此只对时刻 t 之前的时刻输入进行Attention计算，这也称为Mask操作。
## 3.2 Attention ##
- 注意函数可以描述为将一个查询和一组键值对映射到一个输出，其中查询、键、值和输出都是向量。输出被计算为值的加权和，其中分配给每个值的权重由查询与相应键的兼容性函数计算。
### 3.2.1 放缩点积Attention（Scaled Dot-Product Attention） ###
- Figure 2:输入包括dk维的查询和键，以及dv维的值。我们使用所有的键计算查询的点积，将每个键除以√dk，并应用softmax函数来获得值的权重。在实践中，我们同时计算一组查询的注意函数，并将其压缩到矩阵Q中。键和值也被打包到矩阵K和V中。我们将输出矩阵计算为：(1)
- 相较于加法Attention，在实践中点积Attention速度更快，更节省空间，因为其使用矩阵乘法实现计算过程。当d_k较小时，加法Attention和点积Attention性能相近；但d_k较大时，加法Attention性能优于不带缩放的点积Attention。原因：对于很大的 d_k，点积幅度增大，将softmax函数推向具有极小梯度的区域，为抵消这种影响，在Transformer中点积缩小1/√dk倍
### 3.2.2 多头Attention（Multi-Head Attention）###
- Multi-Head Attention就是把Scaled Dot-Product Attention的过程做H次，然后把输出Z合起来。
- 它使得模型可以关注不同位置。虽然在上面的例子中，z1 包含了一点其他位置的编码，但当前位置的单词还是占主要作用， 当我们想知道“The animal didn’t cross the street because it was too tired” 中 it 的含义时，这时就需要关注到其他位置。这个机制为注意层提供了多个“表示子空间”（representation Subspaces）。下面我们将具体介绍：
- (1)经过 Multi-Headed，我们会得到和 heads 数目一样多的 Query / Key / Value 权重矩阵组.论文中用了8个，那么每个Encoder/Decoder我们都会得到 8 个集合。这些集合都是随机初始化的，经过训练之后，每个集合会将input Embeddings （或者来自较低编码器/解码器的向量）投影到不同的表示子空间中。
- (2)我们重读记忆八次相似的操作，得到八个Zi矩阵。简单来说，就是定义8组权重矩阵，每个单词会做8次上面的Self-Attention的计算这样每个单词就会得到8个不同的加权求和Z。
- (3)feed-forward处只能接受一个矩阵，所以需要将这八个矩阵压缩成一个矩阵。方法就是先将八个矩阵连接起来，然后乘以一个额外的权重矩阵W0。为了使得输出与输入结构对标 乘以一个线性W0 得到最终的Z
###  3.2.2 Transformer中的Attention ###
- 在Transformer中以3种方式使用Multi-Head Attention
- 1、在“Encoder-Decoder Attention”层，query来自上面的解码器层，key和value来自编码器的输出。这允许解码器中的每个位置能关注到输入序列中的所有位置。这模仿Seq2Seq模型中典型的Encoder-Decoder的attention机制
- 2、Encoder包含self-attention层。在self-attention层中，所有的key、value和query来自同一个地方，在这里是Encoder中前一层的输出。编码器中的每个位置都可以关注编码器上一层的所有位置。
- 3、Decoder中的self-attention层允许解码器中的每个位置都关注解码器中直到并包括该位置的所有位置（即Decoder中每个位置关注该位置及前面所有位置的信息）。我们需要防止解码器中的向左信息流来保持自回归属性。通过屏蔽softmax的输入中所有不合法连接的值（设置为-∞），我们在缩放版的点积attention中实现。
## 3.3 Position-wise Feed-Forward Networks ##
- - 编码器和解码器中的每一层都包含一个完全连接的前馈网络，该前馈网络分别应用于每个位置。这包括两个线性变换，中间有一个ReLU激活。
## 3.4 嵌入和Softmax ## 
- 与其他序列转导模型类似，我们使用学习到的嵌入将输入词符和输出词符转换为维度为dmodel的向量。我们还使用普通的线性变换和softmax函数将解码器输出转换为预测的下一个词符的概率。在我们的模型中，两个嵌入层之间和pre-softmax线性变换共享相同的权重矩阵。在嵌入层中，我们将这些权重乘以1/√dmodel
## 3.5 位置编码 ##
- 因为Transformer中不包含循环和卷积，为了让模型利用序列的顺序，需要添加位置信息（即使用位置编码表示序列的顺序）。因此将"位置编码"添加到编码器和解码器堆栈底部的输入嵌入中。位置编码和嵌入的维度dmodel相同，因此两者可以相加。
- pos是位置，i是维度，即位置编码的每一个维度对应一个正弦曲线。使用该函数的原因是：很容易学习到对相对位置的关注，因此对确定的偏移k，PEpos+k可以表示为PEpos的线性函数。
## 4 使用Self-Attention的原因 ##
- 使用Self-Attention主要解决三个问题
- 1、每层计算的总复杂度
- 2、可以并行的计算量，以所需的最小顺序操作的数量来衡量
- 3、网络中长距离依赖之间的路径长度。影响学习这种依赖性能力的一个关键因素是前向和后向信号必须在网络中传播的路径长度。输入和输出序列中任意位置组合之间的这些路径越短，学习远距离依赖性就越容易。
- 因此，我们还比较了由不同图层类型组成的网络中任意两个输入和输出位置之间的最大路径长度。
- 表1：不同层类型的最大路径长度、每层复杂性和最小顺序操作数。n 为序列的长度，d 为表示的维度，k 为卷积的核的大小，r 为受限self-attention中邻域的大小。
