# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : transform_tf.py
# Time    : 2019/5/16 0016 下午 5:08
© 2019 Ming. All rights reserved. Powered by King
"""


import tensorflow as tf
import numpy as np




# 模型构建

class Transformer(object):
    """
    Transformer Encoder 用于文本分类
    """

    def __init__(self, num_heads, learning_rate, vocab_size, sequence_length, embed_dim, num_blocks, filters,
                 word_vec=None):

        # 定义模型的输入
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, sequence_length], name="input_y")
        self.dropout = tf.placeholder(tf.float32, name="dropout_prob")
        self.init = tf.initializers.he_normal()
        self.num_heads = num_heads
        self.lr = learning_rate
        #编码器参数
        self.word_vec = word_vec
        self.vocab_size = vocab_size  #self.vocab_size 包含 'unk'
        self.embed_dim = embed_dim
        self.seq_len = sequence_length
        self.filters = filters #feed_forward conv1d的第一层参数，因为解码层要用encoder的结果，第二层filter一般为en_embed_dim
        self.num_blocks = num_blocks #堆叠层数


    def word_embedding(self, inputs):
        # [vocab_size,embed_size],初始化hekaiming_init
        if self.word_vec == None:
            Embedding = tf.get_variable("word_embed", shape=[self.vocab_size, self.embed_dim],
                                             initializer=self.init, trainable=True)
            Embedding = tf.concat((tf.zeros(shape=[1, self.embed_dim]),
                                   Embedding[1:, :]), 0)
        else:
            Embedding = tf.get_variable("word_embed", shape=[self.vocab_size, self.embed_dim],
                                             initializer=tf.constant_initializer(self.word_vec), trainable=True)

        # 利用词嵌入矩阵将输入的句子中的词转换成词向量，维度[batch_size, sequence_length, embed_dim]
        embedded_words = tf.nn.embedding_lookup(Embedding, inputs)
        return embedded_words


    def position_embedding(self, inputs, masking=True):
        '''
        位置信息
        K.ones_like生成与另一个张量shape相同的全1张量(batc_size, 80)
        K.cumsum 在给定轴上求张量的累积和
        K.not_equal逐元素判不等关系，返回布尔张量
        两个tensor相乘是逐元素相乘
        pos 累积位置axis=1，表示x轴每行都是1-max_len
        mask 为有元素为1，没有为0
        0 在word——embeding中代表‘unk’
        '''
        mask = tf.cast(tf.not_equal(inputs, 0), 'int32')
        pos = tf.cumsum(tf.ones_like(inputs, 'int32'), 1)
        pos = pos - 1  #[0,1,2,...,seq_len] 位置索引
        pos_index = pos * mask #若第一和第三位置为0,则[0,1,0,3,...,seq_len]

        #生成位置向量查询表 [self.seq_len_index, self.embed_dim]
        # 根据正弦和余弦函数来获得每个位置上的embedding的 pos /(1000_(2i /d_model)
        positionEmbedding = np.array(
            [[pos / np.power(10000, (i - i % 2) / self.embed_dim) for i in range(self.embed_dim)]
             for pos in range(self.seq_len)])

        # 然后根据奇偶性分别用sin和cos函数来包装  第一个位置为零向量
        # PE(pos, 2i) = sin( pos /(1000_(2i /d_model) )
        # PE(pos, 2i+1) = cos( pos /(1000_(2i /d_model))
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])
        if masking:
            positionEmbedding[0, :] = 0  # 第一个位置为零向量即mask
        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, pos)
        if masking:
            positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, pos_index)
        return positionEmbedded



    def mask(self, inputs, queries=None, keys=None, type=None):
        """Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (N, T_q, T_k)
        queries: 3d tensor. (N, T_q, d)
        keys: 3d tensor. (N, T_k, d)
        e.g.,
        >> queries = tf.constant([[[1.],
                                   [2.],
                                   [0.]]], tf.float32) # (1, 3, 1)
        >> keys = tf.constant([[[4.],
                         [0.]]], tf.float32)  # (1, 2, 1)
        >> inputs = tf.constant([[[4., 0.],
                                  [8., 0.],
                                  [0., 0.]]], tf.float32)
        >> mask(inputs, queries, keys, "key")
        array([[[ 4.0000000e+00, -4.2949673e+09],
                [ 8.0000000e+00, -4.2949673e+09],
                [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
        >> inputs = tf.constant([[[1., 0.],
                                 [1., 0.],
                                [1., 0.]]], tf.float32)
        >> mask(inputs, queries, keys, "query")
        array([[[1., 0.],
            [1., 0.],
            [0., 0.]]], dtype=float32)
        key masking:让key值为0的单元对应的score极小，这样加权计算value的时候相当于对结果不造成影响。
        首先用abs取绝对值，即其值只能为0（一开始的keys值第三个维度值全部为0，reduce_sum加起来之后为0）
        然后再用一个reduce_sum(tf.abs(keys), axis=-1))将最后一个维度上的值加起来，keys的shape也从[N, T_k, d]变为[N,T_k]。
        然后用到了tf.sign(x, name=None)，该函数返回符号 y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0，
        sign会将原tensor对应的每个值变为-1,0,或者1。则经此操作，得到key_masks,有两个值，0或者1。0代表原先的keys第三维度所有值都为0，
        反之则为1，我们要mask的就是这些为0的key。然后一个扩维tf.expand_dims(masks, 1)变成 [N, 1, T_k],
        然后又是一个tf.tile操作，使其变为[N, T_q, T_k]。由于每个queries都要对应这些keys，而mask的key对每个queries都是mask的。
        而之前的key_masks只相当于一份mask，所以扩充之前key_masks的维度，在中间加上一个维度大小为queries的序列长度。
        然后利用tile函数复制相同的mask值即可。然后定义一个和outputs同shape的paddings，该tensor每个值都设定的极小。
        用where函数比较，当对应位置的key_masks值为0也就是需要mask时，outputs的该值（attention score）设置为极小的值（利用paddings实现），
        否则保留原来的outputs值。 经过以上key mask操作之后outputs的shape仍为 (N, T_q, T_k)，只是对应mask了的key的score变为很小的值。

        causality参数告知我们是否屏蔽未来序列的信息（解码器self attention的时候不能看到自己之后的那些信息），这里即causality为True时的屏蔽操作。
        该部分实现还是比较巧妙的，利用了一个三角阵的构思来实现。下面详细介绍。 首先定义一个和outputs后两维的shape相同shape（T_q,T_k）的一个张量（矩阵）。
        然后将该矩阵转为三角阵tril。三角阵中，对于每一个T_q,凡是那些大于它角标的T_k值全都为0，这样作为mask就可以让query只取它之前的
        key（self attention中query即key）。由于该规律适用于所有query，接下来仍用tile扩展堆叠其第一个维度，构成masks，shape为(N, T_q,T_k).
        之后两行代码进行paddings，和之前key mask的过程一样就不多说了。 以上操作就可以当不需要来自未来的key值时将未来位置的key的score设置为极小。
        之后一行代码outputs = tf.nn.softmax(outputs) # (N, T_q, T_k) 将attention score了利用softmax转化为加起来为1的权值，很简单。

        Query Masking,所谓要被mask的内容，就是本身不携带信息或者暂时禁止利用其信息的内容。这里query mask也是要将那些初始值为0的queryies
        （比如一开始句子被PAD填充的那些位置作为query） mask住。代码前三行和key mask的方式大同小异，只是扩展维度等是在最后一个维度展开的。
        操作之后形成的query_masks的shape为[N, T_q, T_k]。 第四行代码直接用outputs的值和query_masks相乘。这里的outputs是之前已经softmax之后的权值。
        所以此步之后，需要mask的权值会乘以0，不需要mask的乘以之前取的正数的sign为1所以权值不变。实现了query_masks的目的。
        outputs的shape应该和query_masks 的shape一样，为(N, T_q, T_k)。

        """

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding的输入为0时，
        # 计算出来的权重应该也是0，由于在transformer结构中引入了位置向量，当padding和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。虽然在query中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
        # query = keys，因此只要一方为0，计算出的权重就为0。
        # padding mask 在所有的scaled—dot-product attention 里面都需要用到，
        # 而sequence mask 只有在decoder的self-attention里面用到。
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            # Generate masks # 将每一时序上的向量中的值相加取平均值,让最后一维embed_dim相加，如果
            masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
            # 利用tf，tile进行张量扩张， 维度[batch_size * num_heads, keys_len] keys_len = keys 的序列长度
            masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
            # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
            masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

            # Apply masks to inputs # tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
            paddings = tf.ones_like(inputs) * padding_num
            # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
            # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
        elif type in ("q", "query", "queries"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
            masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
            masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

            # Apply masks to inputs
            outputs = inputs * masks
        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

            paddings = tf.ones_like(masks) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
        else:
            print("Check if you entered type correctly!")

        return outputs

    def scaled_dot_product_attention(self, Q, K, V, causality=False, training=True,
                                     scope="scaled_dot_product_attention"):
        '''See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            #  计算keys和query之间的点积，维度[batch_size * num_heads, queries_len, embed_dim/num_heads],
            #  后两维是query和 embed_dim/num_heads 的序列长度
            #  tf.matmul 对于高维矩阵，第0维代表batch_size,后两维做矩阵运算 tf.transpose()交换张量位置（矩阵转置）

            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale # 对计算的点积进行缩放处理，除以向量长度的根号值
            outputs /= d_k ** 0.5

            # key masking   softmax 之前做mask mask =-2 ** 32 + 1 防止soft_max
            outputs = self.mask(outputs, Q, K, type="key")

            # causality or future blinding masking
            if causality:
                outputs = self.mask(outputs, type="future")
            # softmax
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # query masking softmax 之后做mask,mask=0
            outputs = self.mask(outputs, Q, K, type="query")
            # dropout
            outputs = tf.layers.dropout(outputs, rate=self.dropout, training=training)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs


    def multihead_attention(self, query, keys, values, num_heads, num_units=None,
                                            causality=False, scope="multihead_attention"):
        #  因为keys是加上了position embedding的，其中不存在padding为0的值,一般要计算mask
        #  num_units 一般是 num_heads的整数倍

        if num_units is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            num_units = query.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
            # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
            # Q, K, V的维度都是[batch_size, sequence_length, num_units(embedding_size)]
            Q = tf.layers.dense(query, num_units, activation=tf.nn.relu)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
            V = tf.layers.dense(values, num_units, activation=tf.nn.relu)

            # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
            # Q_, K_, V_ 的维度都是[batch_size * num_heads, sequence_length, embedding_size/num_heads]
            # tf.split(Q, num_heads, axis=-1) 维度为： [num_heads, batch_size, sequence_length, embedding_size/num_heads]
            Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
            K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
            V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)

            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_, causality=causality, training=True)

            # Restore shape# 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

            # 对每个subLayers建立残差连接，即H(x) = F(x) + x
            outputs += query
            # normalization 层
            outputs = self.layer_normalization(outputs)

        return outputs



    def layer_normalization(self, inputs, scope="layer_normalization"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # LayerNorm层和BN层有所不同
            epsilon = 1e-8

            inputsShape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]

            paramsShape = inputsShape[-1:]

            # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
            # mean, variance的维度都是[batch_size, sequence_len, 1]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

            beta = tf.Variable(tf.zeros(paramsShape))

            gamma = tf.Variable(tf.ones(paramsShape))
            normalized = (inputs - mean) / ((variance + epsilon) ** .5)

            outputs = gamma * normalized + beta

        return outputs




    def feed_forward(self, inputs, filters, scope="feed_forward"):
        # 在这里的前向传播采用卷积神经网络
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # 内层
            params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # 外层
            params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}

            # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
            # 维度[batch_size, sequence_length, embedding_size]
            outputs = tf.layers.conv1d(**params)

            # 残差连接
            outputs += inputs

            # 归一化处理
            outputs = self.layer_normalization(outputs)

        return outputs


    def encoder_layer(self):

        with tf.variable_scope('en_word_embedding', reuse=tf.AUTO_REUSE):
            # embeding layer ,对输入的句子进行word_embeding编码,是否引入训练好的word_vec
            embedded_words = self.word_embedding(self.input_x)
        with tf.variable_scope('en_pos_embedding', reuse=tf.AUTO_REUSE):
            #位置向量编码：维度[batch_size, sequence_length, embed_dim]
            embedded_postion = self.position_embedding(self.input_x, masking=True)
        #论文中将位置向量和词向量相加
        embedding_words_pos = tf.add(embedded_words, embedded_postion)
        #dropout——layer
        embed_words_pos = tf.nn.dropout(embedding_words_pos, keep_prob=self.dropou)
        with tf.variable_scope("Encoding_layer", reuse=tf.AUTO_REUSE):
            for i in range(self.num_blocks):
                with tf.name_scope("Encoding_block-{}".format(i + 1)):
                    # 维度[batch_size, sequence_length, embedding_size]
                    multi_headAtt = self.multihead_attention(embed_words_pos, embed_words_pos, embed_words_pos,
                                                                    self.num_heads, num_units=None, causality=False)
                    # 维度[batch_size, sequence_length, embedding_size]
                    enc = self.feed_forward(multi_headAtt, filters=[self.filters, self.embed_dim])
        outputs = enc
        return outputs


    def decode_layer(self):
        '''
        self.de_filters 一般等于self.en_filters
        '''

        with tf.variable_scope('de_word_embedding', reuse=tf.AUTO_REUSE):
            # embeding layer ,对输入的句子进行word_embeding编码,是否引入训练好的word_vec
            embedded_words = self.word_embedding(self.input_y)
        with tf.variable_scope('de_pos_embedding', reuse=tf.AUTO_REUSE):
            #位置向量编码：维度[batch_size, sequence_length, embed_dim]
            embedded_postion = self.position_embedding(self.input_y, masking=True)
        #论文中将位置向量和词向量相加
        embedding_words_pos = tf.add(embedded_words, embedded_postion)
        #dropout——layer
        embed_words_pos = tf.nn.dropout(embedding_words_pos, keep_prob=self.dropout)
        with tf.variable_scope("Decoding_layer", reuse=tf.AUTO_REUSE):
            for i in range(self.num_blocks):
                with tf.name_scope("Decoding_block-{}".format(i + 1)):
                    # 维度[batch_size, sequence_length, embedding_size]
                    multi_headAtt = self.multihead_attention(embed_words_pos, embed_words_pos, embed_words_pos,
                                        self.num_heads, num_units=None, causality=True, scope="self_attention")

                    multi_headAtt1 = self.multihead_attention(multi_headAtt, self.encoder_layer, self.encoder_layer,
                                        self.num_heads, num_units=None, causality=False, scope="vanilla_attention")


                    # 维度[batch_size, sequence_length, embedding_size]
                    dec = self.feed_forward(multi_headAtt1, filters=[self.filters, self.embed_dim])

        #对decode出来的东西进行解码
        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(embedded_words)  # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights)  # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1)) # (N, T2)返回每一列索引最大

        return logits, y_hat

    def label_smoothing(self, inputs, epsilon=0.1):
        '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
        inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
        epsilon: Smoothing rate.

        For example,

        ```
        import tensorflow as tf
        inputs = tf.convert_to_tensor([[[0, 0, 1],
           [0, 1, 0],
           [1, 0, 0]],
          [[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0]]], tf.float32)

        outputs = label_smoothing(inputs)

        with tf.Session() as sess:
            print(sess.run([outputs]))

        >>
        [array([[[ 0.03333334,  0.03333334,  0.93333334],
            [ 0.03333334,  0.93333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334]],
           [[ 0.93333334,  0.03333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334],
            [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
        ```
        '''
        V = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / V)



    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        enc_outputs = self.encoder_layer(xs)
        logits, preds = self.decode(ys, enc_outputs)

        # train scheme
        y_ = self.label_smoothing(tf.one_hot(ys, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(ys, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', self.lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        en_out = self.encode(xs, False)

        #logging.info("Inference graph is being built. Please be patient.")
        for _ in range(self.hp.maxlen2):
            logits, y_hat = self.decode(ys, en_out, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0] - 1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries