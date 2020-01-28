import os
import pandas as pd
from rnn_model import *
from functions import *
import warnings
warnings.filterwarnings('ignore')

base_dir = '../data'
train_dir = os.path.join(base_dir, 'train.csv')
dev_dir = os.path.join(base_dir, 'dev.csv')
test_dir = os.path.join(base_dir, 'test.csv')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

if __name__ == '__main__':

	config = TRNNConfig()

	if not os.path.exists(vocab_dir):
	    build_vocab(train_dir, 'fact', vocab_dir, config.vocab_size)

	word_to_id = word_encode(vocab_dir)
	label_to_id = label_encode()

	model = TextRNN(config)

	x_train, y_train = process_file(train_dir, 'fact', 'accusation', word_to_id, label_to_id, config.seq_length)
	x_dev, y_dev = process_file(dev_dir, 'fact', 'accusation', word_to_id, label_to_id, config.seq_length)
	x_test, y_test = process_file(test_dir, 'fact', 'accusation', word_to_id, label_to_id, config.seq_length)

	# train(model, x_train, y_train, x_dev, y_dev)
	y_pred_cls = test(model, x_test, y_test)

	labels = ['危险驾驶', '交通肇事', '盗窃', '信用卡诈骗', 
	          '容留他人吸毒', '故意伤害', '抢劫', '寻衅滋事', 
	          '走私、贩卖、运输、制造毒品', '诈骗']

	msg = ';'.join(['{0}:{1}'.format(str(i), labels[i]) for i in range(len(labels))])
	print(msg)

	label_map = {k:v for k,v in zip(range(len(labels)), labels)}
	test_df = pd.read_csv(test_dir)
	if 'pred' in test_df.columns:
	    del test_df['pred']
	test_df['pred'] = pd.Series(y_pred_cls).map(label_map)
	bad_case = test_df.loc[test_df['accusation'] != test_df['pred']][['accusation', 'pred', 'fact']]
	bad_case.to_csv('../data/bad_case.txt', index=None, encoding='utf-8', sep='\t')