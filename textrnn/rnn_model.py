import tensorflow as tf

class TRNNConfig(object):
    '''RNN配置'''
    
    # 1. 模型参数
    vocab_size = 5000 # 词汇表大小
    
    embedding_dim = 64 # 词嵌入维度
    seq_length = 300 # 序列长度
    
    hidden_dim = 128 # 隐藏层维度
    rnn = 'gru' # 单元类型
    num_layers = 2 # 隐层数量
    
    num_classes = 10 # 类别数量
    
    # 2. 学习参数
    dropout_keep_prob = 0.8 # dropout保留比例（cnn: 0.5）
    learning_rate = 1e-2 # 学习率
    
    batch_size = 256 # 每批训练大小
    num_epochs = 10 # 全数据集迭代次数
    
    print_per_batch = 1 # 每多少轮输出结果
    save_per_batch = 10 # 每多少轮保存结果
    
class TextRNN(object):
    '''RNN文本分类模型'''
    
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        ##### 添加样本权重 #####
        self.sample_weights = tf.placeholder(tf.float32, [None, 1], name='sample_weights')
        ##### 添加样本权重 #####
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.rnn()

    def rnn(self):
        
        def lstm_cell(): # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
        
        def gru_cell(): # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)
        
        def dropout(): # 加dropout层
            if self.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            
        with tf.name_scope('rnn'):
            cells = [dropout() for i in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            
            _output, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _output[:, -1, :]
            
        with tf.name_scope('score'):
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        
        with tf.name_scope('optimize'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(self.cross_entropy)
            ##### 添加样本权重 #####
            self.cross_entropy_with_sample_weights = tf.multiply(self.sample_weights, self.cross_entropy)
            self.loss_with_sample_weights = tf.reduce_sum(self.cross_entropy_with_sample_weights)
            # self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss_with_sample_weights)
            ##### 添加样本权重 #####
            
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))