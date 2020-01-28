import numpy as np
import pandas as pd
import time
import functools
from collections import Counter
import tensorflow as tf
import tensorflow.keras as kr
import os
from sklearn.metrics import classification_report, confusion_matrix

def logtime(func):
    '''计时装饰器'''
    
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time1 = time.time()
        result = func(*args, **kw)
        time2 = time.time()
        print(func.__name__, 'time cost:', round(time2-time1, 4), 's')
        return result
    return wrapper

def build_vocab(train_dir, text_col, vocab_dir, vocab_size=5000):
    '''构建词汇表'''
    
    df = pd.read_csv(train_dir)
    texts = df[text_col].to_list()
    
    all_data = []
    for text in texts:
        all_data.extend(text)
        
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size-1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    
    with open(vocab_dir, 'w') as f:
        f.write('\n'.join(words))

def word_encode(vocab_dir):
    '''字符编码'''
    
    with open(vocab_dir, 'r') as f:
        words = [x.strip() for x in f.readlines()]

    word_to_id = dict(zip(words, range(len(words))))    
    
    return word_to_id
    
def label_encode():
    '''标签编码'''
    
    labels = ['危险驾驶', '交通肇事', '盗窃', '信用卡诈骗', 
              '容留他人吸毒', '故意伤害', '抢劫', '寻衅滋事', 
              '走私、贩卖、运输、制造毒品', '诈骗']
    
    label_to_id = dict(zip(labels, range(len(labels))))
    
    return label_to_id

def process_file(data_dir, text_col, label_col, word_to_id, label_to_id, seq_length):
    '''构建数据集'''

    df = pd.read_csv(data_dir)
    texts, labels = df[text_col].to_list(), df[label_col].to_list()
          
    text_id, label_id = [], []
    
    for i in range(len(texts)):
        text_id.append([word_to_id[_] for _ in texts[i] if _ in word_to_id])
        label_id.append(label_to_id[labels[i]])
    
    x_pad = kr.preprocessing.sequence.pad_sequences(text_id, seq_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(label_to_id))
    
    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    '''生成批次数据'''

    n = len(x)
    num_batch = int(n/batch_size) + 1
    idxs = np.random.permutation(np.arange(n))
    x_shuffle = x[idxs]
    y_shuffle = y[idxs]

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min((i+1) * batch_size, n)
        yield x_shuffle[start_idx:end_idx], y_shuffle[start_idx:end_idx]

@logtime
def train(model, x_train, y_train, x_dev, y_dev):
    '''训练'''
    
    tensorboard_dir = 'tensorboard/textrnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
        
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    save_dir = 'checkpoints/textrnn'
    save_path = os.path.join(save_dir, 'best_validation')
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    batch = 0 # 批次
    best_dev_acc = 0 # 验证集合最佳准确率
    best_batch = 0 # 最佳的batch
    early_stop = 1000 # 早停轮数

    flag = False
    for epoch in range(model.config.num_epochs):
        print('Epoch:', epoch + 1)
        
        batch_train = batch_iter(x_train, y_train, model.config.batch_size)

        for x_batch, y_batch in batch_train:

            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.keep_prob: model.config.dropout_keep_prob}
            sess.run(model.optim, feed_dict=feed_dict)

            if batch % model.config.save_per_batch == 0:

                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, batch)

            if batch % model.config.print_per_batch == 0:

                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = sess.run([model.loss, model.acc], feed_dict=feed_dict)

                feed_dict[model.input_x] = x_dev
                feed_dict[model.input_y] = y_dev

                loss_dev, acc_dev = sess.run([model.loss, model.acc], feed_dict=feed_dict)

                if acc_dev > best_dev_acc:
                    best_dev_acc = acc_dev
                    best_batch = batch
                    saver.save(sess=sess, save_path=save_path)

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}'
                print(msg.format(batch, loss_train, acc_train, loss_dev, acc_dev))


            batch += 1

            if batch - best_batch > early_stop:
            # if batch == 5:
                print('{0}轮没有提升，提前停止...'.format(batch))
                flag = True
                break

        if flag:
            break

@logtime
def test(model, x_test, y_test):
    '''预测，评估'''

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    save_dir = 'checkpoints/textrnn'
    save_path = os.path.join(save_dir, 'best_validation')
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=save_path)

    feed_dict = {model.input_x: x_test, model.input_y: y_test, model.keep_prob: 1.0}
    loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss, acc))

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = sess.run(model.y_pred_cls, feed_dict=feed_dict)

    msg_classification_report = classification_report(y_test_cls, y_pred_cls)
    print(msg_classification_report)

    msg_confusion_matrix = confusion_matrix(y_test_cls, y_pred_cls)
    print(msg_confusion_matrix)

    return y_pred_cls
