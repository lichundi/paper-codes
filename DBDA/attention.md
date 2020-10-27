# 摘要 #
- 序列转导模型主要是基于复杂的递归或卷积神经网络，包括一个编码器和一个解码器。本文提出一个新的简单的网络结构Transformer完全基于注意力机制，不需要递归和卷积。
- 在两个机器翻译任务上的实验表明，该模型的结果较好，并行性更高，所需的训练时间大大减少。在WMT 2014英语-德语翻译任务中达到了28.4 BLEU，比现有的最佳结果提高了超过2个BLEU。在8个GPU上训练3.5天，建立一个最先进的模型41.8BLEU。
# 引言 #
- 


# 模型结构 #
- 大多数竞争性神经序列转导模型都有编解码器结构[5,2,35]。这里，编码器将符号表示的输入序列（x1，…，xn）映射到连续表示序列z=（z1，…，zn）。给定z，解码器然后一次生成一个符号的输出序列（y1，…，ym）。在每一步中，模型都是自回归的[10]，在生成下一步时，使用先前生成的符号作为附加输入
- Transformer其实这就是一个Seq2Seq模型（ Seq2Seq模型是输出的长度不确定时采用的模型，这种情况一般是在机器翻译的任务中出现，将一句中文翻译成英文，那么这句英文的长度有可能会比中文短，也有可能会比中文长，所以输出的长度就不确定了。），左边一个encoder把输入读进去，右边一个decoder得到输出：
- Transformer=Transformer Encoder + Transformer Decoder。
## 3.1 编码器和解码器堆栈 ##
- Encoder：编码器由N = 6 个完全相同的层堆叠而成。每一层都有两个子层。第一个子层是一个multi-head self-attention机制，第二个子层是一个简单的、位置完全连接的前馈网络。我们对每个子层再采用一个残差连接，接着进行层标准化。也就是说，每个子层的输出是LayerNorm(x + Sublayer(x))，其中Sublayer(x) 是由子层本身实现的函数。为了方便这些残差连接，模型中的所有子层以及嵌入层产生的输出维度都为dmodel = 512。
- Decoder：解码器同样由N = 6 个完全相同的层堆叠而成。 除了每个编码器层中的两个子层之外，解码器还插入第三个子层，该层对Encoder堆栈的输出执行multi-head attention。 与编码器类似，我们在每个子层再采用残差连接，然后进行层标准化。我们还修改解码器堆栈中的self-attention子层，以防止位置关注到后面的位置。这种掩码结合将输出嵌入偏移一个位置，确保对位置的预测 i 只能依赖小于i 的已知输出。
## 3.2 Attention ##
- 注意函数可以描述为将一个查询和一组键值对映射到一个输出，其中查询、键、值和输出都是向量。输出被计算为值的加权和，其中分配给每个值的权重由查询与相应键的兼容性函数计算。
### 3.2.1 放缩点积Attention（Scaled Dot-Product Attention） ###
- Figure 2:输入包括dk维的查询和键，以及dv维的值。我们使用所有的键计算查询的点积，将每个键除以√dk，并应用softmax函数来获得值的权重。在实践中，我们同时计算一组查询的注意函数，并将其压缩到矩阵Q中。键和值也被打包到矩阵K和V中。我们将输出矩阵计算为：(1)
- 相较于加法Attention，在实践中点积Attention速度更快，更节省空间，因为其使用矩阵乘法实现计算过程。当d_k较小时，加法Attention和点积Attention性能相近；但d_k较大时，加法Attention性能优于不带缩放的点积Attention。原因：对于很大的 d_k，点积幅度增大，将softmax函数推向具有极小梯度的区域，为抵消这种影响，在Transformer中点积缩小1/√dk倍
### 3.2.2 多头Attention（Multi-Head Attention）###
###  3.2.2 Transformer中的Attention ###
- 在Transformer中以3种方式使用Multi-Head Attention
- 1、在“Encoder-Decoder Attention”层，query来自上面的解码器层，key和value来自编码器的输出。这允许解码器中的每个位置能关注到输入序列中的所有位置。这模仿Seq2Seq模型中典型的Encoder-Decoder的attention机制
- 2、Encoder包含self-attention层。在self-attention层中，所有的key、value和query来自同一个地方，在这里是Encoder中前一层的输出。编码器中的每个位置都可以关注编码器上一层的所有位置。
- 3、Decoder中的self-attention层允许解码器中的每个位置都关注解码器中直到并包括该位置的所有位置（即Decoder中每个位置关注该位置及前面所有位置的信息）。我们需要防止解码器中的向左信息流来保持自回归属性。通过屏蔽softmax的输入中所有不合法连接的值（设置为-∞），我们在缩放版的点积attention中实现。
## 3.3 Position-wise Feed-Forward Networks ##
## 3.4 嵌入和Softmax ## 
- 与其他序列转导模型类似，我们使用学习到的嵌入将输入词符和输出词符转换为维度为\large d_{model}的向量。我们还使用普通的线性变换和softmax函数将解码器输出转换为预测的下一个词符的概率。在我们的模型中，两个嵌入层之间和pre-softmax线性变换共享相同的权重矩阵。在嵌入层中，我们将这些权重乘以\large \sqrt{d_{model}}

